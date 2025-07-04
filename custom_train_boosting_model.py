import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import get_peft_model, LoraConfig

import argparse
import logging
import os
import wandb

from src.custom.glue_multitask_model import GLUEMultitaskModel
from src.custom.glue_multitask_data_module import GLUEMultitaskDataModule

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
import pandas as pd
from collections import defaultdict
import time

from src.merging_utils.ensemble import EnsembleAdapterModule, EnsembleLoRAModule
from adapters import AutoAdapterModel, DoubleSeqBnConfig

def initialize_model(args):
    model_key = args.model_key.replace("/", "-").replace("..", "")
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
    
    if args.train_adapter:
        
        if args.use_qadapter:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4' 
            )

            model = AutoAdapterModel.from_pretrained(
                hf_key, 
                quantization_config=quantization_config, 
                torch_dtype=torch.bfloat16, 
                device_map={"": args.devices[0]}
            )
        
        else: model = AutoAdapterModel.from_pretrained(hf_key)

        bottleneck_config = DoubleSeqBnConfig(
            mh_adapter=True,    
            output_adapter=True,    
            reduction_factor=args.reduction_factor,     
            non_linearity="relu",
            dropout=0.1,     
        )

        model.add_adapter(adapter_name="seq_bn",config=bottleneck_config)
        model.adapter_to("seq_bn", f"cuda:{args.devices[0]}", torch.bfloat16)
        model.heads.default[0].weight.data = model.heads.default[0].weight.data.to(torch.bfloat16)

        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

        model.set_active_adapters("seq_bn")
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params_count = sum(p.numel() for p in model.parameters())

        print(f"Trainable parameters: {trainable_params_count} || All parameters: {all_params_count} || ratio: {trainable_params_count/all_params_count}")
        print("-"*20,"Bottleneck_Adapter","-"*20)

    
    if args.use_3bit or args.use_2bit:
        from src.lqlora_utils import lora_utils
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, hf_key, model_type, append_eos

''' Linear Approximation: linear regression '''
def generate_state_dict(model, state_dict, coef, device="cpu", removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings", "quant", "absmax"]):
    new_state_dict = {}; cur_len = 0
    for key, param in model.named_parameters():
        param_len = param.numel()
        if any([rkey in key for rkey in removing_keys]):
            continue
        else:
            if "lora" in key or "adapter" in key:
                new_state_dict[key] = state_dict[key].clone().to(device) + \
                    torch.Tensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
                cur_len += param_len
    return new_state_dict

def solve_linear_regression(lm, state_dict, gradients, outputs, residuals, alpha=1.0):
    from sklearn.linear_model import Ridge

    y = (residuals - outputs)
    reg = Ridge(alpha=alpha).fit(gradients, y)

    coef = reg.coef_
    project_matrix = lm.project_matrix
    coef = project_matrix @ coef.flatten()
    print("L2 norm before scaling", np.linalg.norm(coef))

    # evaluate task performances
    new_state_dict = generate_state_dict(lm.model, state_dict, coef, device=lm.model.device)
    return new_state_dict


def aggregate_predictions_adaboost(outputs):
    error = 0; sum_weights = 0; masks = []
    for batch in outputs:        
        # skipping this batch if no labels are present
        if len(batch["label_ids"]) == 0:
            masks.append(batch["masks"].clone())
            continue

        mask = batch["masks"].clone()

        # mask indicates where the prediction is incorrect
        mask[mask>0] = ~torch.Tensor(batch["label_ids"] == batch["pred_ids"]).type(torch.bool).view(-1)
        error += batch["weights"][mask].sum()
        sum_weights += batch["weights"].sum()
        masks.append(mask)
    
    error = error/sum_weights # the sum of weights on the incorrect predictions over the sum of all weights
    masks = torch.cat(masks)
    return error, masks

def aggregate_predictions_gradient_boosting(outputs):

    def onehot_encoding(label, n_classes):
        """Conduct one-hot encoding on a label vector."""
        label = label.view(-1)
        onehot = torch.zeros(label.size(0), n_classes).float().to(label.device)
        onehot.scatter_(1, label.view(-1, 1), 1)

        return onehot

    all_residuals = []; masks = [] # residuals per sample (if multiple positions in one sample, we sum them up)
    for batch in outputs:
        labels = batch["labels"][:, 1:].contiguous()
        correct_class_probs = batch["probs"]
        mask = batch["masks"].clone()
        residuals = np.zeros(len(labels))

        is_label_mask = labels != -100
        valid_labels = labels[is_label_mask].view(-1)
        if len(residuals) == 1:
            residuals = (1 - correct_class_probs).cpu().numpy() if mask[0] else np.zeros(len(labels))
        else:
            residuals[mask] = (1 - correct_class_probs).cpu().numpy()
        
        all_residuals.append(residuals)
        masks.append(mask)
    all_residuals = np.concatenate(all_residuals, axis=0)
    masks = torch.cat(masks)
    return all_residuals, masks

def aggregate_metrics(outputs, task_names):
    from sklearn.metrics import accuracy_score, f1_score
    """
    Gather outputs from all GPUs and save validation predictions as a CompletionDataset and
    log validation metrics.

    Note, `all_gather` *concatenates* tensors from all GPUs along the first dimension.
    """
    summary = {f"{task_name}_loss": 0 for task_name in task_names}
    summary.update({f"{task_name}_accuracy_score": 0 for task_name in task_names})
    summary.update({f"{task_name}_f1_score": 0 for task_name in task_names})
    
    # average loss
    losses = [output["loss"] for output in outputs]
    losses = torch.stack(losses)
    losses = losses[torch.isnan(losses) == False]
    summary.update({"loss": losses.mean().item()})

    task_counts = {task_name: 0 for task_name in task_names}
    for batch in outputs:
        task_name = batch["task_name"]
        if len(batch["label_ids"]) == 0:
            continue
        summary[f"{task_name}_loss"] += batch["loss"].item()*len(batch["label_ids"]) if torch.isnan(batch["loss"]) == False else 0
        summary[f"{task_name}_accuracy_score"] += accuracy_score(batch["label_ids"], batch["pred_ids"])*len(batch["label_ids"])*100
        summary[f"{task_name}_f1_score"] += f1_score(batch["label_ids"], batch["pred_ids"], average="macro")*len(batch["label_ids"])*100
        task_counts[task_name] += len(batch["label_ids"])
    
    for task_name in task_names:
        summary[f"{task_name}_loss"] = (summary[f"{task_name}_loss"]/task_counts[task_name]) if task_counts[task_name] > 0 else 0
        summary[f"{task_name}_accuracy_score"] = (summary[f"{task_name}_accuracy_score"]/task_counts[task_name]) if task_counts[task_name] > 0 else 0
        summary[f"{task_name}_f1_score"] = (summary[f"{task_name}_f1_score"]/task_counts[task_name]) if task_counts[task_name] > 0 else 0

    # average accuracy and f1 score
    summary.update({"accuracy_score": np.mean([summary[f"{task_name}_accuracy_score"] for task_name in task_names])})
    summary.update({"f1_score": np.mean([summary[f"{task_name}_f1_score"] for task_name in task_names])})

    # Log metrics
    print(summary)
    return summary

def main(args):
    # Initialize the model
    model_key = args.model_key.replace("/", "-").replace("..", "")
    save_name = (f"{args.save_name}_{model_key}" if args.save_name else "") + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") 
    seed_str = "_".join([str(seed) for seed in args.compute_gradients_seeds])
    gradients_dir = save_name + f"_dim_{args.project_gradients_dim}_seed_{seed_str}"

    model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # initialize the data module
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
        elif "Llama-3" in args.model_key:
            answer_choices = [" " + choice.strip() for choice in answer_choices] 
            answer_choices = [tokenizer([choice])["input_ids"][0][1] for choice in answer_choices]; answer_choices.sort()
        elif "TinyLlama" in args.model_key:
            answer_choices = [choice.strip() for choice in answer_choices]
            answer_choices = [tokenizer([choice])["input_ids"][0][1] for choice in answer_choices]; answer_choices.sort()
        else:
            answer_choices = [" " + choice.strip() for choice in answer_choices] 
            answer_choices = [tokenizer([choice])["input_ids"][0][0] for choice in answer_choices]; answer_choices.sort()
        task_answer_choices[task_name] = answer_choices


    def get_lora_state_dict(model):
        state_dict = {}
        for name, weight in model.state_dict().items():
            if "lora" in name or "adapter" in name:
                state_dict[name] = weight.detach().clone().to(f"cuda:{args.devices[0]}")
        return state_dict

    initial_lora_state_dict = torch.load(os.path.join(os.path.join("gradients", gradients_dir), "initial_weights.pt"), map_location="cpu")

    lm = GLUEMultitaskModel(model, tokenizer, model_type, use_cpu_offload=False,
                    lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb, 
                    optimizer=args.optimizer, generate_output=args.generate_output, task_names=args.task_names, task_answer_choices=task_answer_choices,
                    compute_gradients=True, gradients_dir=gradients_dir, 
                    project_gradients_dim=args.project_gradients_dim, compute_gradients_seeds=args.compute_gradients_seeds)

    if not os.path.exists("external_lightning_logs"):
        raise Exception("external_lightning_logs/ does not exist")
    default_root_dir = os.path.join("external_lightning_logs", 
                                "gradient_boosting" if args.train_gradient_boosting else "adaboosting",
                                f"_{model_key}_" + \
                                "_".join(args.task_names) + \
                                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                                (f"_{args.save_name}" if args.save_name else "")
                                )

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=default_root_dir,
        filename="epoch_{epoch}",
        save_top_k=(-1 if args.save_every_epoch else 1),
        mode="min",
    )

    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                        default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                        accumulate_grad_batches=args.accumulate, precision=args.precision,
                        enable_checkpointing=True,
                        callbacks=[checkpoint_callback]
                        )
    
    # Before estimation
    tasks_to_lengths = [len(task_list) for task_list in data_module.multitask_train_sampler._train_data_list]
    tasks_to_indices = [list(range(sum(tasks_to_lengths[:i]), sum(tasks_to_lengths[:i+1]))) for i, task_name in enumerate(args.task_names)]

    ''' In-place save '''
    from lightning_fabric.utilities.cloud_io import _load as pl_load
    def save_trained_model(checkpoint_dir, appendix):
        checkpoint = pl_load(checkpoint_dir, map_location=lm.device)
        state_dict = checkpoint["state_dict"]
        state_dict = {k[6:]: v for k, v in state_dict.items() if "lora" in k}
        save_path_dir = checkpoint_dir.replace(".ckpt", "") + appendix + ".pt"
        torch.save(state_dict, save_path_dir)
    ''' In-place save '''

    ''' First Load a ensemble of models '''
    lm.fit_least_square = False
    load_model_dirs = []
    for load_model_dir in args.load_model_dirs:
        load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
        # checkpoint = torch.load(load_model_dir, map_location="cpu")
        # model.load_state_dict(checkpoint, strict=False)
        load_model_dirs.append(load_model_dir)
    if args.load_task_weights is not None:
        with open(args.load_task_weights, "r") as f:
            weights = [[float(item) for item in line.strip().split()] for line in f.readlines()]
    else:
        weights = [1.0/len(load_model_dirs) for _ in range(len(load_model_dirs))]

    # only using the ensemble model for evaluation now
    _ensemble_model = EnsembleLoRAModule(model, load_model_dirs, weights, args.task_names) if args.train_lora else \
                      EnsembleAdapterModule(model, load_model_dirs, weights, args.task_names)
    ensemble_model = GLUEMultitaskModel(_ensemble_model, tokenizer, model_type, use_cpu_offload=False,
                    lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb, use_ensemble=True,
                    optimizer=args.optimizer, generate_output=args.generate_output, task_names=args.task_names, task_answer_choices=task_answer_choices)
    # initial validation
    trainer.validate(ensemble_model, dataloaders=data_module.val_dataloader())

    ''' Gradient Boosting '''
    if args.train_gradient_boosting:
        gradient_boosting_lr = args.gradient_boosting_lr
        for est_idx in range(args.n_estimators):

            # Compute residuals: use the correct class probabilities to compute the residuals
            outputs = trainer.predict(ensemble_model, dataloaders=data_module.train_dataloader())
            aggregate_metrics(outputs, data_module.task_names)
            residuals, masks = aggregate_predictions_gradient_boosting(outputs)
            data_module.update_residuals(residuals, masks)

            # Train a new model
            lm.lr = 5e-4
            lm.fit_least_square = True
            model.load_state_dict(initial_lora_state_dict, strict=False) # re-initialize the adapter weights
            if args.fit_linear_regression:
                # Load gradients 
                def load_gradient_features(seed, task_name, gradient_idxes):
                    cur_gradients_dir = gradients_dir.replace(f"seed_{args.compute_gradients_seeds[0]}", f"seed_{seed}")

                    gradients = []
                    for filename in os.listdir(f"./gradients/{cur_gradients_dir}/{task_name}"):
                        if filename.endswith("_gradients.npy"):
                            gradients.append(np.load(f"./gradients/{cur_gradients_dir}/{task_name}/{filename}"))
                    gradients = np.concatenate(gradients, axis=0)
                    return gradients
                
                def load_outputs(seed, task_name, gradient_idxes):
                    cur_gradients_dir = gradients_dir.replace(f"seed_{args.compute_gradients_seeds[0]}", f"seed_{seed}")

                    outputs = []
                    for filename in os.listdir(f"./gradients/{cur_gradients_dir}/{task_name}"):
                        if filename.endswith("_outputs.npy"):
                            outputs.append(np.load(f"./gradients/{cur_gradients_dir}/{task_name}/{filename}"))
                    outputs = np.concatenate(outputs, axis=0)
                    outputs = outputs[:, 0]
                    outputs = outputs[outputs != 0] # in case the outputs are cut off by the maximum length
                    return outputs
                
                all_gradients = []; all_outputs = []
                for task_idx in range(len(args.task_names)):
                    task_gradients = load_gradient_features(args.compute_gradients_seeds[0], args.task_names[task_idx], tasks_to_indices[task_idx])
                    task_outputs = load_outputs(args.compute_gradients_seeds[0], args.task_names[task_idx], tasks_to_indices[task_idx])
                    all_gradients.append(task_gradients); all_outputs.append(task_outputs)
                all_gradients = np.concatenate(all_gradients, axis=0)
                all_outputs = np.concatenate(all_outputs, axis=0) if all_outputs[0] is not None else None
                
                new_state_dict = solve_linear_regression(lm, initial_lora_state_dict, all_gradients, all_outputs, residuals, alpha=args.lr_alpha)
                task_names_str = "_".join(args.task_names)
                checkpoint_dir =  os.path.join(default_root_dir, f"gradient_boosting_{task_names_str}_iteration_{est_idx}" + ".pt")
                torch.save(new_state_dict, checkpoint_dir)
            else:
                checkpoint_callback = ModelCheckpoint(
                    monitor="loss",
                    dirpath=default_root_dir,
                    filename="epoch_{epoch}",
                    save_top_k=(-1 if args.save_every_epoch else 1),
                    mode="min",
                ) # initialize a new checkpoint callback
                trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                                default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                                accumulate_grad_batches=args.accumulate, precision=args.precision,
                                enable_checkpointing=True,
                                callbacks=[checkpoint_callback]
                                )
                trainer.fit(lm, data_module)

                # Save the trained model and write an ensemble to evaluate the predictions
                task_names_str = "_".join(args.task_names)
                appendix = f"_gradient_boosting_{task_names_str}_iteration_{est_idx}"
                checkpoint_dir = checkpoint_callback.best_model_path.replace(".ckpt", "") + appendix + ".pt"
                save_trained_model(checkpoint_callback.best_model_path, appendix)
                os.remove(checkpoint_callback.best_model_path)

            _ensemble_model.add_adapter(checkpoint_dir, gradient_boosting_lr)
            trainer.validate(ensemble_model, dataloaders=data_module.val_dataloader())
    elif args.train_adaboosting:
        for est_idx in range(args.n_estimators):
            # Compute the prediction errors of the current ensemble model
            outputs = trainer.predict(ensemble_model, dataloaders=data_module.train_dataloader())
            aggregate_metrics(outputs, data_module.task_names)
            error, masks = aggregate_predictions_adaboost(outputs)
            alpha = np.math.log((1-error)/error)
            data_module.update_weights(alpha, masks)

            # Train a new model
            lm.use_sample_weights = True
            checkpoint_callback = ModelCheckpoint(
                monitor="loss",
                dirpath=default_root_dir,
                filename="epoch_{epoch}",
                save_top_k=(-1 if args.save_every_epoch else 1),
                mode="min",
            ) # initialize a new checkpoint callback
            trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                            default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                            accumulate_grad_batches=args.accumulate, precision=args.precision,
                            enable_checkpointing=True,
                            callbacks=[checkpoint_callback]
                            )
            trainer.fit(lm, data_module)
            
            # Save the trained model and write an ensemble to evaluate the predictions
            task_names_str = "_".join(args.task_names)
            appendix = f"_adaboosting_{task_names_str}_iteration_{est_idx}"
            checkpoint_dir = checkpoint_callback.best_model_path.replace(".ckpt", "") + appendix + ".pt"
            save_trained_model(checkpoint_callback.best_model_path, appendix)
            os.remove(checkpoint_callback.best_model_path)

            _ensemble_model.add_adapter(checkpoint_dir, alpha)
            trainer.validate(ensemble_model, dataloaders=data_module.val_dataloader())
    else:
        raise Exception("Please specify a boosting algorithm to train")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_names", type=str, nargs="+", default=["cb"])
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--use_3bit", action="store_true")
    parser.add_argument("--use_2bit", action="store_true")
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    parser.add_argument("--downsample_ratio", type=float, default=0.5)
    parser.add_argument("--minimum_samples", type=int, default=500)
    parser.add_argument("--minimum_samples_validation", type=int, default=200)

    parser.add_argument("--train_adapter", action="store_true")
    parser.add_argument("--reduction_factor", type=int, default=128)
    parser.add_argument("--use_qadapter", action="store_true")

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    parser.add_argument("--save_name", type=str, default=None)

    parser.add_argument("--load_model_dirs", type=str, nargs="+", default=["meta-llama-Llama-3.2-1B_cb_multirc_lora_r_16_task_grouping_2_run_0/epoch_epoch=3.pt"])
    parser.add_argument("--load_task_weights", type=str, default=None, help="Path to the task weights file to load")
    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--generate_output", action="store_true")

    # Boosting arguments
    parser.add_argument("--train_gradient_boosting", action="store_true")
    parser.add_argument("--train_adaboosting", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=10)
    parser.add_argument("--gradient_boosting_lr", type=float, default=0.2)
    parser.add_argument("--fit_linear_regression", action="store_true")
    parser.add_argument("--lr_alpha", type=float, default=10) # alpha for the ridge regression
    parser.add_argument("--compute_gradients_seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--project_gradients_dim", type=int, default=400)
    args = parser.parse_args()

    main(args)