python evaluate_merged_model.py --task_names "cb" "rte" --model_key "meta-llama/Llama-3.2-1B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs  "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt"\
    "meta-llama-Llama-3.2-1B_rte_winogrande_debiased_story_cloze_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=8.pt"\
    "meta-llama-Llama-3.2-1B_copa_wic_wsc.fixed_boolq_lora_r_16_task_grouping_4_run_0/epoch_epoch=9.pt"\
    --merge_strategy "simple_ensemble"

python evaluate_merged_model.py --task_names "cb" "rte" --model_key "meta-llama/Llama-3.2-1B"\
    --devices 1 --batch_size 4 --inference_batch_size 4 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs  "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt"\
    "meta-llama-Llama-3.2-1B_rte_winogrande_debiased_story_cloze_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=8.pt"\
    "meta-llama-Llama-3.2-1B_copa_wic_wsc.fixed_boolq_lora_r_16_task_grouping_4_run_0/epoch_epoch=9.pt"\
    --merge_strategy "max_ensemble"

python evaluate_merged_model.py --task_names "cb" "rte" --model_key "meta-llama/Llama-3.2-1B"\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --save_name merging_tinyllama --precision "bf16-true" \
    --merge_model_dirs  "meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt"\
    "meta-llama-Llama-3.2-1B_rte_winogrande_debiased_story_cloze_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=8.pt"\
    "meta-llama-Llama-3.2-1B_copa_wic_wsc.fixed_boolq_lora_r_16_task_grouping_4_run_0/epoch_epoch=9.pt"\
    --merge_strategy "simple_ensemble" --use_qlora --optimizer "paged_adamw_32bit"
