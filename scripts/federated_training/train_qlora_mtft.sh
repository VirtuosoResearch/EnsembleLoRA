for i in {1..500}
do
python custom_train_shakespeare.py --num_tasks 100 --task_idxes $i\
    --model_key "meta-llama/Llama-3.2-1B"\
    --devices 1 --batch_size 8 --inference_batch_size 8 --max_length 80 --runs 1 --lr 2e-4 --precision "bf16-true"\
    --train_lora --lora_rank 16 --lora_alpha 128 \
    --use_qlora --optimizer "paged_adamw_32bit"\
    --save_name shakespeare_meft_200 --epochs 10 --write_results --load_model_dir "quantized-finetuning/external_lightning_logs/meta-llama-Llama-3.2-1B_shakespeare_num_tasks_100_lora_r_16_shakespeare_run_0/epoch_epoch=9.pt"
done