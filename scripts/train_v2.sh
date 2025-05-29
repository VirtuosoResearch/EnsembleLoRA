python custom_train.py --model_key 'meta-llama/Llama-3.1-8B' --val_split_ratio 0.1 \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 3 --task_names "cb" "wic" "wsc.fixed" "boolq"\
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora \
    --save_name grouped --epochs 10

python custom_train.py --model_key 'meta-llama/Llama-3.1-8B' --val_split_ratio 0.1 \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 3 --task_names "copa" "multirc" "rte"\
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora \
    --save_name grouped --epochs 10

python custom_train.py --model_key 'meta-llama/Llama-3.1-8B' --val_split_ratio 0.1 \
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 512 --runs 3 --task_names "winogrande_debiased" "story_cloze" "hellaswag"\
    --train_lora --lora_rank 4 --lora_alpha 32 --use_qlora \
    --save_name grouped --epochs 10