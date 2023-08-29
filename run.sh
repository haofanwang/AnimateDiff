apt-get update
apt-get install tmux
apt-get install libgl1

export https_proxy=10.7.4.2:3128

pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install diffusers==0.20.1 transformers==4.25.1 xformers==0.0.16 imageio==2.31.2 einops omegaconf safetensors accelerate decord compel

# lama
pip install pyyaml easydict scikit-image scikit-learn joblib matplotlib pandas tabulate webdataset albumentations==0.5.2 hydra-core==1.1.0


pip install numpy==1.22.3
pip install kornia==0.5.0 

pip install pytorch-lightning==2.0.6 opencv-python==3.4.11.45 

cd /mnt/public02/usr/yanzhen1/workspace/AnimateDiff

# finetune or training from scratch
accelerate launch train.py --config=./configs/training/finetuning.yaml

# training interpolation
accelerate launch train.py --config=./configs/training/training_interpolation.yaml --interpolation=True

# inference
python -m scripts.animate --config configs/prompts/1-ToonYou.yaml