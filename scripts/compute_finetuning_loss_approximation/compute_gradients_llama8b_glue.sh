python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name llama8b_10_tasks_dim_400_v2 --epochs 0 --precision "bf16-true" \
    --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 400

# python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
#     --model_key "meta-llama/Llama-3.2-3B"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --use_qlora --optimizer "paged_adamw_32bit"\
#     --save_name llama3b_10_tasks_dim_400 --epochs 0 --precision "bf16-true" \
#     --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 1000

# python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
#     --model_key "meta-llama/Llama-3.2-1B"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --use_qlora --optimizer "paged_adamw_32bit"\
#     --save_name llama1b_10_tasks_dim_400 --epochs 0 --precision "bf16-true" \
#     --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 1000

# python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
#     --model_key "meta-llama/Llama-3.1-8B"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --use_qlora --optimizer "paged_adamw_32bit"\
#     --save_name llama8b_10_tasks_dim_1000 --epochs 0 --precision "bf16-true" \
#     --compute_gradient_steps 10000000 --compute_gradient_seed 0 --project_gradients_dim 1000