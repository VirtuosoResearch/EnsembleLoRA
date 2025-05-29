python fast_estimate_compute_gradients_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag"\
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1\
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name "meaningful_save_name" --epochs 0 --precision "bf16-true" \
    --compute_gradient_seeds 0 --project_gradients_dim 400 --compute_gradient_steps 10000000 