python custom_train_glue_mtl.py --task_names "cb" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtft_qadapter --epochs 10 --write_results --load_model_dir "meta-llama-Llama-3.1-8B_cb_multirc_rte_winogrande_debiased_story_cloze_hellaswag_copa_wic_wsc.fixed_boolq_downsampled_mtl_qadapter_run_0/epoch_epoch=7.pt"

python custom_train_glue_mtl.py --task_names "multirc" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtft_qadapter --epochs 10 --write_results --load_model_dir "meta-llama-Llama-3.1-8B_cb_multirc_rte_winogrande_debiased_story_cloze_hellaswag_copa_wic_wsc.fixed_boolq_downsampled_mtl_qadapter_run_0/epoch_epoch=7.pt"

python custom_train_glue_mtl.py --task_names "rte" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtft_qadapter --epochs 10 --write_results --load_model_dir "meta-llama-Llama-3.1-8B_cb_multirc_rte_winogrande_debiased_story_cloze_hellaswag_copa_wic_wsc.fixed_boolq_downsampled_mtl_qadapter_run_0/epoch_epoch=7.pt"

python custom_train_glue_mtl.py --task_names "copa" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtft_qadapter --epochs 10 --write_results --load_model_dir "meta-llama-Llama-3.1-8B_cb_multirc_rte_winogrande_debiased_story_cloze_hellaswag_copa_wic_wsc.fixed_boolq_downsampled_mtl_qadapter_run_0/epoch_epoch=7.pt"

python custom_train_glue_mtl.py --task_names "boolq" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtft_qadapter --epochs 10 --write_results --load_model_dir "meta-llama-Llama-3.1-8B_cb_multirc_rte_winogrande_debiased_story_cloze_hellaswag_copa_wic_wsc.fixed_boolq_downsampled_mtl_qadapter_run_0/epoch_epoch=7.pt"

python custom_train_glue_mtl.py --task_names "wic" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtft_qadapter --epochs 10 --write_results --load_model_dir "meta-llama-Llama-3.1-8B_cb_multirc_rte_winogrande_debiased_story_cloze_hellaswag_copa_wic_wsc.fixed_boolq_downsampled_mtl_qadapter_run_0/epoch_epoch=7.pt"

python custom_train_glue_mtl.py --task_names "winogrande_debiased" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtft_qadapter --epochs 10 --write_results --load_model_dir "meta-llama-Llama-3.1-8B_cb_multirc_rte_winogrande_debiased_story_cloze_hellaswag_copa_wic_wsc.fixed_boolq_downsampled_mtl_qadapter_run_0/epoch_epoch=7.pt"

python custom_train_glue_mtl.py --task_names "hellaswag" \
    --downsample_ratio 1e-3 --minimum_samples 500 --minimum_samples_validation 100 \
    --model_key "meta-llama/Llama-3.1-8B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 2 --lr 5e-5\
    --train_adapter --use_qadapter --reduction_factor 128 --precision "bf16-true" \
    --save_name downsampled_mtft_qadapter --epochs 10 --write_results --load_model_dir "meta-llama-Llama-3.1-8B_cb_multirc_rte_winogrande_debiased_story_cloze_hellaswag_copa_wic_wsc.fixed_boolq_downsampled_mtl_qadapter_run_0/epoch_epoch=7.pt"