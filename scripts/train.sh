# QLoRA
python custom_train.py \
    --model_key 'meta-llama/Llama-3.1-8B' --val_split_ratio 0.1 \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 2e-5 \
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora --precision 'bf16-true' --epochs 10 --save_name test

# Quantized Adapter
python custom_train.py \
    --model_key 'meta-llama/Llama-3.1-8B' --val_split_ratio 0.1 \
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 2e-5\
    --train_adapter --use_qadapter --reduction_factor 256 --precision 'bf16-true'  --epochs 10 --save_name test