# finetune or training from scratch
accelerate launch train.py --config=./configs/training/finetuning.yaml

# training interpolation
accelerate launch train.py --config=./configs/training/training_interpolation.yaml --interpolation=True

# inference
python -m scripts.animate --config configs/prompts/1-ToonYou.yaml