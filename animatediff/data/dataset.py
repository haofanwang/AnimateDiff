import decord
# decord.bridge.set_bridge('torch')

import os
import json
import yaml
import random
import numpy as np
from PIL import Image
from einops import rearrange
from typing import Callable, List, Optional, Union

from torch.utils.data import Dataset
from torchvision import utils
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as T

try:
    
    # https://huggingface.co/camenduru/big-lama/blob/main/big-lama/models/best.ckpt
    import sys
    sys.path.append('../../lama')
    
    from lama.saicinpainting.evaluation.utils import move_to_device
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    import torch
    from torch.utils.data._utils.collate import default_collate
    from lama.saicinpainting.training.data.datasets import make_default_val_dataset
    from lama.saicinpainting.training.trainers import load_checkpoint
    from lama.saicinpainting.evaluation.data import load_image, pad_img_to_modulo

    import cv2
    from omegaconf import OmegaConf
    
    model_dir = '../../lama/big-lama/models/best.ckpt'
    if not os.path.exists(model_dir):
        ckpts = "https://huggingface.co/camenduru/big-lama/blob/main/big-lama/models/best.ckpt"
        os.system(f"wget -O {ckpts} {model_dir}")
    
    use_lama = True
    
except Exception as e:
    print("lama is not found! Dewatermarking will be disabled!")
    use_lama = False
    
class MultiTuneAVideoDataset(Dataset):
    def __init__(
            self,
            meta_video_root: str = None,
            meta_path: str = None,
            video_path: Union[str, List[str]] = None,
            prompt: Union[str, List[str]] = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 16,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 16,
            maximum_count: int = 1000000,
            train_config_path = "/mnt/public02/usr/yanzhen1/workspace/AnimateDiff/lama/big-lama/config.yaml",
            checkpoint_path = "/mnt/public02/usr/yanzhen1/workspace/AnimateDiff/lama/big-lama/models/best.ckpt",
            mask_dir = "/mnt/public02/usr/yanzhen1/workspace/AnimateDiff/lama/test_data/modelscope-mask.jpg",
    ):
        
        metainfo = json.load(open(meta_path))
        self.video_path = []
        self.prompt = []
        count = 0
        for video_id in metainfo:
            if count > maximum_count:
                break
            local_path, prompt = metainfo[video_id]
            video_path = os.path.join(meta_video_root, local_path.split('./')[-1])
            if os.path.exists(video_path):
                self.video_path.append(video_path)
                self.prompt.append(prompt)
                count += 1
        
        self.prompt_ids = []

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        
        if use_lama:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
            with open(train_config_path, 'r') as f:
                train_config = OmegaConf.create(yaml.safe_load(f))
            train_config.training_model.predict_only = True
            train_config.visualizer.kind = 'noop'
            self.model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
            self.model.freeze()
            self.model.to(self.device)

            self.mask = cv2.imread(mask_dir, 0)
            self.mask = self.expandEdge(255-self.mask, k=7)
            self.mask = self.mask.astype('float32') / 255.0
        
    def __len__(self):
        return len(self.video_path)
    
    def expandEdge(self, mask, k=3, iternum=3):
        ele = np.ones((k, k), 'uint8')
        mask_ = mask.copy()
        for i in range(iternum):
            mask_ = cv2.dilate(mask_, ele, cv2.BORDER_DEFAULT)
        return mask_
    
    def inpaint_image(self, frame, target_size=(256,256)):
    
        height, width, _ = frame.shape
        cx, cy = width//2, height//2
        frame = frame[cy-height//2:cy+height//2, cx-height//2:cx+height//2, :]
        frame = cv2.resize(frame, (256,256))

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (2, 0, 1))
        frame = frame.astype('float32') / 255.0

        h, w = self.mask.shape[:2]
        with torch.no_grad():

            result = dict(image=frame, mask=self.mask[None, ...])

            result['image'] = pad_img_to_modulo(result['image'], 8)
            result['mask'] = pad_img_to_modulo(result['mask'], 8)

            batch = move_to_device(default_collate([result]), self.device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = self.model(batch)
            cur_res = batch["inpainted"][0,:,:h, :w].permute(1, 2, 0).detach().cpu().numpy()

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            #cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cur_res = cv2.resize(cur_res, target_size)
    
        return cur_res
    
    def __getitem__(self, index):
        
        # load and sample video frames
        try:
            if use_lama:
                vr = decord.VideoReader(self.video_path[index])
            else:
                vr = decord.VideoReader(self.video_path[index], width=self.width, height=self.height)
        except Exception as e:
            os.remove(self.video_path[index])
            return self.__getitem__(random.randint(0, len(self.video_path)-1))
        
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        if len(sample_index) < self.n_sample_frames:
            return self.__getitem__(random.randint(0, len(self.video_path)-1))
        
        video = vr.get_batch(sample_index)
        
        if use_lama:
            # lama inpainting
            new_video = []
            for frame in video.asnumpy():
                new_frame = self.inpaint_image(frame, target_size=(256,256))
                new_video.append(new_frame)
            video = np.stack(new_video, axis=0)
            video = torch.from_numpy(video).contiguous()
        else:
            video = torch.from_numpy(video.asnumpy()).contiguous()
            video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids[index]
        }
        
        return example
    