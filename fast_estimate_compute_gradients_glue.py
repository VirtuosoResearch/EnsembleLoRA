import argparse
import logging
import os
import wandb

from src.custom.glue_multitask_data_module import GLUEMultitaskDataModule
from src.custom.glue_multitask_model import GLUEMultitaskModel
# from src.lqlora_utils import lora_utils

from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy, _or_policy
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.trainer.states import RunningStage, TrainerFn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.nn import Embedding

from peft import get_peft_model, LoraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import SingleDeviceStrategy
import pandas as pd
from collections import defaultdict
import time

from torch._inductor.async_compile import AsyncCompile

logging.basicConfig(level=logging.INFO, force=True)
torch.set_float32_matmul_precision("high")

def get_trainable_parameters(model, removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "layer_norm", "embed_tokens", "norm"]):
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any([key in name for key in removing_keys]):
            continue
        params.append(param)
    return params

def initialize_model(args):
    if "gpt" in args.model_key or "Llama" in model_key \
        or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
        hf_key = args.model_key.replace("_", "-")
        tokenizer = AutoTokenizer.from_pretrained(hf_key)
        tokenizer.padding_side = 'right'
        if args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map={"": args.devices[0]}) #
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_key)
        model_type = "decoder"
        append_eos = True
    elif "flan" in model_key:
        hf_key = "google/{}".format(model_key.replace("_", "-"))
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
        tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
        model_type = "encoder_decoder"
        append_eos = False  # t5 tokenizers already append eos
    else:
        raise NotImplementedError(args.model_key)
    
    if args.use_3bit or args.use_2bit:
        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            use_gradient_checkpointing=True)

        lora_utils.transform_lora_layers(
            lpq=False,
            model=model,
            model_name="nf3" if args.use_3bit else "nf2",
            device=f"cuda:{args.devices[0]}")
        model.to(f"cuda:{args.devices[0]}")        

    elif args.train_lora:
        if args.model_key == "gpt2": # for gpt2, we generally use full model
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["c_attn", "c_proj", "c_fc"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif args.model_key == "EleutherAI/gpt-neox-20b":
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query_key_value"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif "flan" in args.model_key:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q", "k", "v"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        else:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    return model, tokenizer, hf_key, model_type, append_eos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_names", type=str, nargs="+", default=["cb", "rte", "copa", "wic", "wsc.fixed", "boolq", "multirc", "winogrande_debiased", "story_cloze", "hellaswag"])
    parser.add_argument("--model_key", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--task_idxes", type=int, nargs="+", default=None)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    parser.add_argument("--downsample_ratio", type=float, default=0.5)
    parser.add_argument("--minimum_samples", type=int, default=500)
    parser.add_argument("--minimum_samples_validation", type=int, default=200)


    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--use_3bit", action="store_true")
    parser.add_argument("--use_2bit", action="store_true")

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--load_model_dir", type=str, default="test")
    parser.add_argument("--generate_output", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")


    # compute gradient arguments
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--compute_gradient_steps", type=int, default=1e7)
    parser.add_argument("--compute_gradients_seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--project_gradients_dim", type=int, default=200)

    args = parser.parse_args()
    args.enable_checkpointing = not args.disable_checkpointing
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    ''' Constants '''
    model_key = args.model_key.replace("/", "-").replace("..", "")
    load_model_dir = args.load_model_dir
    save_name = (f"{args.save_name}_{model_key}" if args.save_name else "") + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") 
    seed_str = "_".join([str(seed) for seed in args.compute_gradients_seeds])
    gradients_dir = save_name + f"_dim_{args.project_gradients_dim}_seed_{seed_str}" # + ("_pretrained" if not os.path.exists(load_model_dir) else "")
    file_dir = os.path.join("./results/", save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    if not os.path.exists("external_lightning_logs"):
            raise Exception("external_lightning_logs/ does not exist")
    default_root_dir = os.path.join("external_lightning_logs", 
                                    f"{model_key}_" + \
                                    "_".join(args.task_names) + \
                                    (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                                    (f"_{args.save_name}" if args.save_name else "")
                                    )
    ''' Constants '''

    metrics = {}
    model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)

    if os.path.exists(load_model_dir):
        returned_keys = model.load_state_dict(torch.load(load_model_dir, map_location="cpu"), strict=False)
        print(f"Loaded from {load_model_dir}: {returned_keys}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = args.batch_size
    if args.inference_batch_size is None:
        inference_batch_size = batch_size
    else:
        inference_batch_size = args.inference_batch_size
    
    data_module = GLUEMultitaskDataModule(
            task_names=args.task_names,
            tokenizer=tokenizer,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            max_input_length=args.max_length,
            val_split_ratio=args.val_split_ratio,
            downsample_ratio=args.downsample_ratio,
            minimum_samples=args.minimum_samples,
            minimum_samples_validation=args.minimum_samples_validation)
    data_module.setup(stage="fit")

    task_answer_choices = {}
    for task_name in args.task_names:
        answer_choices = data_module.task_to_templates[task_name].answer_choices.split("|||")
        # process the answer choices, different models tokenize them differently
        if "gpt" in args.model_key: 
            answer_choices = [" " + choice.strip() for choice in answer_choices] 
            answer_choices = [tokenizer([choice])["input_ids"][0][0] for choice in answer_choices]; answer_choices.sort()
        elif "TinyLlama" in args.model_key or ("CodeLlama" in args.model_key):
            answer_choices = [choice.strip() for choice in answer_choices]
            answer_choices = [tokenizer([choice])["input_ids"][0][1] for choice in answer_choices]; answer_choices.sort()
        elif "Llama-3" in args.model_key:
            answer_choices = [" " + choice.strip() for choice in answer_choices] 
            answer_choices = [tokenizer([choice])["input_ids"][0][1] for choice in answer_choices]; answer_choices.sort()
        else:
            answer_choices = [" " + choice.strip() for choice in answer_choices] 
            answer_choices = [tokenizer([choice])["input_ids"][0][0] for choice in answer_choices]; answer_choices.sort()
        task_answer_choices[task_name] = answer_choices
    lm = GLUEMultitaskModel(model, tokenizer, model_type, use_cpu_offload=False,
                    lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb, 
                    optimizer=args.optimizer, generate_output=args.generate_output, task_names=args.task_names, task_answer_choices=task_answer_choices,
                    compute_gradients=True, gradients_dir=gradients_dir,
                    project_gradients_dim=args.project_gradients_dim, compute_gradients_seeds=args.compute_gradients_seeds, 
                    compute_gradients_steps=args.compute_gradient_steps, start_step=args.start_step)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="accuracy_score",
        dirpath=default_root_dir,
        filename="epoch_{epoch}",
        save_top_k=(-1 if args.save_every_epoch else 1),
        mode="max",
    )
    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                        default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                        accumulate_grad_batches=args.accumulate, precision=args.precision,
                        enable_checkpointing=args.enable_checkpointing,
                        callbacks=[checkpoint_callback], use_distributed_sampler=False, inference_mode=False
                        )
    # save initial weights
    if args.train_lora:
        if not os.path.exists(os.path.join("gradients", gradients_dir)):
            os.makedirs(os.path.join("gradients", gradients_dir))
        model_path = os.path.join("gradients", gradients_dir) + "/initial_weights.pt"
        state_dict = model.state_dict()
        state_dict = {k: v.clone() for k, v in state_dict.items() if "lora" in k}
        torch.save(state_dict, model_path)
    
    start_time = time.time()
    outputs = trainer.predict(lm, dataloaders=data_module.train_dataloader())
    end_time = time.time()
    print("Time for computing gradients & outputs", end_time - start_time)
