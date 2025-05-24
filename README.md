# Quantized Fine-tuning

## Usage of Adapter

training adapter script: `~/Quantized-boosting-code/quantized-finetuning/scripts/train_glue/train_adapter.sh`

Using ```Bottleneck Adapter```

There are two parts in Bottleneck Adapter: mh_adapter, output_adapter.

Make ```reduction_factor``` larger if you want to reduce the number of adapter parameter. 

## Instruction dataset and Alpaca

Data: `~/in-context-learning/chain-of-thought-finetuning/data/alpaca_data`

Link: https://github.com/HazyResearch/skill-it/tree/main/aux_data 

Package: Pytorch Nightly; Pytorch Lightning

```
conda env create -f environment.yml
```

#### Step 1

Train a model on all data: `./scripts/alpaca/train_alpaca.sh`

Use `custom_train_alpaca.py`
```
python custom_train_alpaca.py --model_key $model_key\
    --lr 5e-5 --batch_size 1 --max_length 256 --epochs 10\
    --train_lora --lora_rank 4 --lora_alpha 32\
    --strategy auto --devices 0 1 2 --runs 1 --accumulate 1 --precision "bf16-true" 
```
Main parameters: 
- `model_key`: huggingface directories: `../llama/llama-3/Meta-Llama-3-8B-hf`, `google/gemma-2b`, `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`, `flan_t5_base`, ...
- `save_name`: name for saving the checkpoints
- `load_model_dir`: the name of the directory of the checkpoint to be loaded
- `write_results`: write the loss results to a csv file

#### Step 2

Compute and project gradients on all training samples: `./script/alpaca/eval_approximation_err.sh`

Run `fast_estimate_eval_approximation_alpaca.py` 
```
python fast_estimate_eval_approximation_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100 \
    --devices 0 1 2 --strategy auto --save_name $save_name --compute_pretrained_outputs
```
- Main parameters: 
  - `--compute_pretrained_outputs` will compute the gradients and outputs at the initialization
  - `--model_key`
  - `--save_name`
  - `--downsample`: downsampling the dataset
  - `--num_batches_gradients`: number of batches to compute gradients. `num_batches_gradients` * `batch_size` = `downsample`

Note: Before `load_model_dir`, use `export_hf_model` to save models as huggingface model checkpoints. 

### Step 3: Compute Linear Approximation Error

Estimate linear regression models on subsets of clusters: `./script/alpaca/eval_approximation_err.sh`

Use `fast_estimate_eval_approximation_alpaca.py` 
```
python fast_estimate_eval_approximation_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 2 --max_length 256 --project_gradients --project_dimension 100\
    --scale 0.2 --seed $seed --devices 0 1 2 --strategy auto
```
Main parameters:
- `--project_gradients` 
- `--project_dimension`
- `--scale`: control the fine-tuned distance
- `--save_name`

```
save_name = ("Instruction_{}".format(model_key) if args.train_instruction else "Alpaca_{}".format(model_key)) + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                ("_{}".format(args.save_name) if args.save_name != "none" else "")

gradient_dir = save_name + f"_dim_{args.project_dimension}_run_{args.run}" + ("_pretrained" if args.load_model_dir is None else "")
```

Gradients and outputs are saved to `./gradients/{gradient_dir}/`


### Step 4: Estimate Fine-tune Model Parameters

```
python fast_estimate_linear_regression_alpaca.py \
    --model_key $model_key --train_lora --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size 16 --project_gradients --project_dimension 200 --devices 2\
    --load_model_dir $load_dir\
    --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --number_of_subsets 1000 --subset_size 0.5 --save_name $save_name
```

Main parameters: 
- `--load_sample_task_dir`: if specified, will load previously sampled subsets
  - Otherwise, specify `--number_of_subsets` and `--subset_size`
- `--scale`: controls the norm of the fine-tuned model
- `--save_name`

```
save_name = "Alpaca_{}".format(model_key) + \
            (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
            (f"_{args.save_name}" if args.save_name else "") + \
            f"_dim_{args.project_dimension}_run_{args.run}" 
file_dir = os.path.join("./results/", save_name)
```

### Other Operations

Collect ground-truth empirical fine-tuned model losses. 

Use `sample_train_results_alpaca.py`
```
python sample_train_results_alpaca.py \
    --model_key $model_key --precision 32 --lr 5e-5\
    --load_sample_task_dir Alpaca_EleutherAI-gpt-neo-125M_lora_r_4 --devices 0 1 2 --subset_num 100\
    --save_name $save_name
```

