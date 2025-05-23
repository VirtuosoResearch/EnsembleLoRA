python fast_estimate_compute_gradients_shakespeare.py --num_tasks 500\
    --model_key "meta-llama/Llama-3.2-1B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 80 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name shakespeare_task_500_llama1b --epochs 0 --precision "bf16-true" \
    --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 400