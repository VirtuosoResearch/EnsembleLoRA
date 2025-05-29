# Evaluate gradients
python fast_estimate_eval_approximation.py \
    --model_key 'meta-llama/Llama-3.1-8B' --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --project_gradients --devices 0 --save_name test\
    --seed 0 --project_dimension 400 --compute_pretrained_outputs

# Evaluate errors
python fast_estimate_eval_approximation.py \
    --model_key 'meta-llama/Llama-3.1-8B' --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 4 --project_gradients --devices 0 --save_name test\
    --seed $seed --project_dimension 100 --scale $scale