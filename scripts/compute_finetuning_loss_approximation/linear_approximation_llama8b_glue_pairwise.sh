for downsample in 5000
do
for lambda in 2e4
do
python fast_estimate_linear_regression_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag" \
    --model_key  "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name llama8b_10_tasks_dim_400_v2 --epochs 0 --precision "bf16-true" \
    --compute_gradient_seed 0 --project_gradients_dim 400 \
    --regularization_lambda $lambda --downsample $downsample\
    --subset_size 0.2 --number_of_subsets 50   
done
done

# for lambda in 5e3 2e3 1e3
# do
# python fast_estimate_linear_regression_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag" \
#     --model_key  "meta-llama/Llama-3.2-3B"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --use_qlora --optimizer "paged_adamw_32bit"\
#     --save_name llama3b_10_tasks_dim_400 --epochs 0 --precision "bf16-true" \
#     --compute_gradient_seed 0 --project_gradients_dim 400 \
#     --regularization_lambda $lambda --load_sample_task_dir "llama8b_glue_10tasks_meta-llama-Llama-3.1-8B_lora_r_16_size_3.0_scale_0.3" --downsample $downsample
# done
# done

# for downsample in 5000 10000 20000
# do
# for lambda in 1e3 5e2 1e2 # 2e4 2e3 5e5 1e5
# do
# python fast_estimate_linear_regression_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag" \
#     --model_key  "meta-llama/Llama-3.1-8B"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --use_qlora --optimizer "paged_adamw_32bit"\
#     --save_name llama8b_10_tasks_dim_1000 --epochs 0 --precision "bf16-true" \
#     --compute_gradient_seed 0 --project_gradients_dim 1000 \
#     --regularization_lambda $lambda --load_sample_task_dir "llama8b_glue_10tasks_meta-llama-Llama-3.1-8B_lora_r_16_size_3.0_scale_0.3" --downsample $downsample
# done
# done

# for lambda in 5e3 2e3 
# do
# python fast_estimate_linear_regression_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag" \
#     --model_key  "meta-llama/Llama-3.2-1B"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --use_qlora --optimizer "paged_adamw_32bit"\
#     --save_name llama1b_10_tasks_dim_400 --epochs 0 --precision "bf16-true" \
#     --compute_gradient_seed 0 --project_gradients_dim 400 \
#     --regularization_lambda $lambda --load_sample_task_dir "llama8b_glue_10tasks_meta-llama-Llama-3.1-8B_lora_r_16_size_3.0_scale_0.3"
# done

# python fast_estimate_linear_regression_glue.py --task_names "cb" "rte" "copa" "wic" "wsc.fixed" "boolq" "multirc" "winogrande_debiased" "story_cloze" "hellaswag" \
#     --model_key  "meta-llama/Llama-3.1-8B"\
#     --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --use_qlora --optimizer "paged_adamw_32bit"\
#     --save_name llama8b_10_tasks_dim_400 --epochs 0 --precision "bf16-true" \
#     --compute_gradient_seed 0 --project_gradients_dim 400 \
#     --regularization_lambda 1e-6 --scale 0.1 --use_inv 

# python fast_estimate_linear_regression_glue.py --task_names "cb"\
#     --model_key  "meta-llama/Llama-3.1-8B"\
#     --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --lr 5e-5\
#     --train_lora --lora_rank 16 --lora_alpha 128 \
#     --use_qlora --optimizer "paged_adamw_32bit"\
#     --save_name llama8b_1_tasks_dim_400 --epochs 0 --precision "bf16-true" \
#     --compute_gradient_seed 0 --project_gradients_dim 400 \
#     --regularization_lambda 2000 --downsample 5000 --use_initialization