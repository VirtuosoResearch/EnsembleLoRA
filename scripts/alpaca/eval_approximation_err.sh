
python fast_estimate_eval_approximation_alpaca.py \
    --model_key ../llama/llama-3/Meta-Llama-3-8B-hf --train_lora --lora_rank 4 --lora_alpha 32 --precision 16\
    --batch_size 4 --project_gradients --project_dimension 100\
    --scale 0.1 --seed 0 --devices 0 1 2 --strategy fsdp --compute_pretrained_outputs
