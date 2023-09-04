import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path

from compel import Compel
from controlnet_aux import OpenposeDetector, LineartDetector, CannyDetector

def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)
    inference_config = OmegaConf.load(args.inference_config)

    config  = OmegaConf.load(args.config)
    samples = []
    
    sample_idx = 0
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
        
            ### >>> create validation pipeline >>> ###
            tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
            unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False
                
            processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            
            controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_openpose",
                    use_safetensors=False
                )

            """
            controlnet_names = [
                "lllyasviel/control_v11p_sd15_openpose",
                "lllyasviel/control_v11p_sd15_canny",
                "lllyasviel/control_v11p_sd15_lineart",
            ]
            controlnet_list = []
            for controlnet_name in controlnet_names:
                controlnet_list.append(ControlNetModel.from_pretrained(
                    controlnet_name,
                    use_safetensors=False
                ))
            controlnet = MultiControlNetModel(controlnet_list)
            
            processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
            canny = CannyDetector()
            """
           
            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs.DDIMScheduler)),
            ).to("cuda")

            # 1. unet ckpt
            # 1.1 motion module
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
            missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
            
            # 1.2 T2I
            if model_config.path != "":
                if model_config.path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.path)
                    pipeline.unet.load_state_dict(state_dict)
                    
                elif model_config.path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                            
                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)                
                    
                    # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                    
                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

            pipeline.to("cuda")
            ### <<< create validation pipeline <<< ###

            prompts      = model_config.prompt
            n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            
            init_image   = model_config.init_image if hasattr(model_config, 'init_image') else None
            last_image   = model_config.last_image if hasattr(model_config, 'last_image') else None
            
            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
            
            config[config_key].random_seed = []
            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                
                # manually set random seed for reproduction
                if random_seed != -1: torch.manual_seed(random_seed)
                else: torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())
                
                # if your image size is not 512x512, you need change the source code of controlnet_aux to return ori size
                openpose_image = processor(Image.open(init_image).convert('RGB').resize((args.W, args.H)), 
                                           include_face=True, 
                                           include_hand=True)
                
                use_compel = True
                
                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")
                if use_compel:
                    sample = pipeline(
                        prompt,
                        negative_prompt     = n_prompt,
                        init_image          = init_image,
                        last_image          = last_image,
                        num_inference_steps = model_config.steps,
                        guidance_scale      = model_config.guidance_scale,
                        width               = w,
                        height              = h,
                        video_length        = args.L,
                        fp16                = args.fp16,
                        control_image       = [openpose_image],
                    ).videos
                else:
                    conditioning = compel([prompt])
                    n_conditioning = compel([n_prompt])
                    sample = pipeline(
                        prompt_embeds=conditioning,
                        negative_prompt_embeds     = n_conditioning,
                        init_image          = init_image,
                        last_image          = last_image,
                        num_inference_steps = model_config.steps,
                        guidance_scale      = model_config.guidance_scale,
                        width               = w,
                        height              = h,
                        video_length        = args.L,
                        fp16                = args.fp16,
                        control_image       = [openpose_image],
                    ).videos
                samples.append(sample)

                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
                print(f"save to {savedir}/sample/{prompt}.gif")
                
                sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--fp16", action="store_true")
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    args = parser.parse_args()
    main(args)
