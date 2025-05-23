python fast_estimate_linear_regression_shakespeare.py --num_tasks 100\
    --model_key  "meta-llama/Llama-3.1-8B"\
    --devices 2 --batch_size 4 --inference_batch_size 4 --max_length 80 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name shakespeare_task_100_llama1b --epochs 0 --precision "bf16-true" \
    --compute_gradient_seed 0 --project_gradients_dim 200 --scale 0.3 --number_of_subsets 1000 --subset_size 10