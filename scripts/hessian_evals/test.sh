python compute_hessian_traces.py --task_name "cb" \
      --model_key "meta-llama/Llama-3.1-8B"\
      --devices 1 --batch_size 1 --inference_batch_size 1 --max_length 512\
      --train_lora --lora_rank 16 --lora_alpha 128 --precision 'bf16-true'\
      --save_name test_hessian --num_samples 100 --load_model_dir "meta-llama-Llama-3.1-8B_cb_lora_r_16_varying_rank_run_0/epoch_epoch=2.pt"