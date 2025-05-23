task_names=("cb" "multirc" "rte" "winogrande_debiased" "story_cloze" "hellaswag"  "copa" "wic" "wsc.fixed" "boolq") 
checkpoints=(
    "meta-llama-Llama-3.2-1B_cb_downsampled_single_task_adapter_run_0/epoch_epoch=4.pt" # 
    "meta-llama-Llama-3.2-1B_multirc_downsampled_single_task_adapter_run_0/epoch_epoch=4.pt" # 
    "meta-llama-Llama-3.2-1B_rte_downsampled_single_task_adapter_run_0/epoch_epoch=5.pt" # 
    "meta-llama-Llama-3.2-1B_winogrande_debiased_downsampled_single_task_adapter_run_0/epoch_epoch=8.pt" # 
    "meta-llama-Llama-3.2-1B_story_cloze_downsampled_single_task_adapter_run_0/epoch_epoch=6.pt" # 
    "meta-llama-Llama-3.2-1B_hellaswag_downsampled_single_task_adapter_run_0/epoch_epoch=4.pt" # 
    "meta-llama-Llama-3.2-1B_copa_downsampled_single_task_adapter_run_0/epoch_epoch=7.pt" #
    "meta-llama-Llama-3.2-1B_wic_downsampled_single_task_adapter_run_0/epoch_epoch=7.pt" # 
    "meta-llama-Llama-3.2-1B_wsc.fixed_downsampled_single_task_adapter_run_0/epoch_epoch=0.pt" # 
    "meta-llama-Llama-3.2-1B_boolq_downsampled_single_task_adapter_run_0/epoch_epoch=9.pt" # 
)
length=${#task_names[@]}

for j in 0 2 4 6 8; do
for ((i = j+1; i < $length; i++)); do
python evaluate_merged_model.py --task_names "${task_names[$j]}" "${task_names[$i]}" --model_key "meta-llama/Llama-3.2-1B"\
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100\
    --devices 0 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 1 --epochs 0\
    --train_adapter --reduction_factor 128 \
    --save_name merging_llama1b_adapter --precision "bf16-true" \
    --merge_model_dirs "${checkpoints[$j]}" "${checkpoints[$i]}"\
    --merge_strategy "averaging" --merge_scale 0.8
done
done