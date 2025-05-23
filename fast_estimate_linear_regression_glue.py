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
from sklearn.linear_model import LogisticRegression
import pandas as pd
from collections import defaultdict
import time

from torch._inductor.async_compile import AsyncCompile
from src.merging_utils.ensemble import EnsembleAdapterModule, EnsembleLoRAModule

logging.basicConfig(level=logging.INFO, force=True)
torch.set_float32_matmul_precision("high")

def add_result_to_csv(result_datapoint, file_name):
    for key, val in result_datapoint.items():
        result_datapoint[key] = [val, ]
    
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        tmp_df = pd.DataFrame(result_datapoint)
        result_df = pd.concat([result_df, tmp_df], ignore_index = True)
        result_df.to_csv(file_name)
    else:
        result_df = pd.DataFrame(result_datapoint)  
        result_df.to_csv(file_name) 

def get_trainable_parameters(model, removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "layer_norm", "embed_tokens", "norm", "quant", "absmax"]):
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any([key in name for key in removing_keys]):
            continue
        params.append(param)
    return params

def compute_norm(state_dict, use_lora = True, removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings", "quant", "absmax"]):
    norm = 0
    for key, val in state_dict.items():
        if use_lora:
            if "lora" in key:
                norm += val.clone().square().sum().item()
        else:
            if any([rkey in key for rkey in removing_keys]):
                    continue
            norm += val.clone().square().sum().item()
    return np.sqrt(norm)

def generate_state_dict(model, state_dict, coef, device="cpu", removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings", "quant", "absmax"]):
    new_state_dict = {}; cur_len = 0
    for key, param in model.named_parameters():
        if not param.requires_grad: continue
        param_len = param.numel()
        if any([rkey in key for rkey in removing_keys]):
            continue
            # new_state_dict[key] = state_dict[key].clone()
        else:
            assert "lora" in key
            new_state_dict[key] = state_dict[key].clone().to(device) + \
                torch.Tensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
            cur_len += param_len
    return new_state_dict

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
    parser.add_argument("--task_names", type=str, nargs="+", default=["cola"])
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
    parser.add_argument("--task_idxes", type=int, nargs="+", default=None)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--downsample_ratio", type=float, default=1.0)
    parser.add_argument("--minimum_samples", type=int, default=1e6)
    parser.add_argument("--minimum_samples_validation", type=int, default=1e6)

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
    parser.add_argument("--compute_gradients_seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--project_gradients_dim", type=int, default=200)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--regularization_lambda", type=float, default=1)
    parser.add_argument("--number_of_subsets", type=int, default=100)
    parser.add_argument("--subset_size", type=float, default=0.5)
    parser.add_argument("--load_sample_task_dir", type=str, default=None)
    parser.add_argument("--downsample", type=int, default=5000)
    parser.add_argument("--use_initialization", action="store_true")

    parser.add_argument("--use_ensemble", action="store_true")
    parser.add_argument("--num_repeat", type=int, default=1)

    args = parser.parse_args()
    args.enable_checkpointing = not args.disable_checkpointing
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    ''' Constants '''
    model_key = args.model_key.replace("/", "-").replace("..", "")
    save_name = (f"{args.save_name}_{model_key}" if args.save_name else "") + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") 
    gradients_dir = save_name + f"_dim_{args.project_gradients_dim}_seed_{args.compute_gradients_seeds[0]}" # + ("_pretrained" if not os.path.exists(args.load_model_dir) else "")
    load_model_dir = os.path.join(os.path.join("gradients", gradients_dir), "initial_weights.pt")
    file_dir = os.path.join("./results/", save_name + \
                            f"_size_{args.subset_size}_scale_{args.scale}_lambda_{args.regularization_lambda}_downsample_{args.downsample}" + \
                            (f"_pretrained" if args.use_initialization else ""))
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    sampled_task_dir = save_name + f"_size_{args.subset_size}_scale_{args.scale}"
    
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

    # load the initial weights!
    print(load_model_dir)
    if os.path.exists(load_model_dir):
        returned_keys = model.load_state_dict(torch.load(load_model_dir, map_location="cpu"), strict=False)
        print(f"Loaded the initial weights from {load_model_dir}!")

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
            val_split_ratio=0,
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

    # for task in data_module.task_to_templates:
    #     train_dataset = data_module.task_to_train_datasets[task]
    #     for answer_choice in data_module.task_to_templates[task].answer_choices.split("|||"):
    #         answer_choice = answer_choice.strip()
    #         print("Answer: ", answer_choice, "Num samples:", np.sum([1 for output in train_dataset["output"] if output == answer_choice]))

    #     print("Validation")
    #     valid_dataset = data_module.task_to_valid_datasets[task]
    #     for answer_choice in data_module.task_to_templates[task].answer_choices.split("|||"):
    #         answer_choice = answer_choice.strip()
    #         print("Answer: ", answer_choice, "Num samples:", np.sum([1 for output in valid_dataset["output"] if output == answer_choice]))

    #     print("Test")
    #     test_dataset = data_module.task_to_test_datasets[task]
    #     for answer_choice in data_module.task_to_templates[task].answer_choices.split("|||"):
    #         answer_choice = answer_choice.strip()
    #         print("Answer: ", answer_choice, "Num samples:", np.sum([1 for output in test_dataset["output"] if output == answer_choice]))

    if args.use_ensemble:
        model = EnsembleLoRAModule(model, [], [])
    lm = GLUEMultitaskModel(model, tokenizer, model_type, use_cpu_offload=False,
                    lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb, 
                    optimizer=args.optimizer, generate_output=args.generate_output, task_names=args.task_names, task_answer_choices=task_answer_choices,
                    compute_gradients=True, gradients_dir=gradients_dir, 
                    project_gradients_dim=args.project_gradients_dim, compute_gradients_seeds=args.compute_gradients_seeds)
    
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
    
    # Before estimation
    tasks_to_lengths = [len(task_list) for task_list in data_module.multitask_train_sampler._train_data_list]
    tasks_to_indices = [list(range(sum(tasks_to_lengths[:i]), sum(tasks_to_lengths[:i+1]))) for i, task_name in enumerate(args.task_names)]
    
    if args.use_ensemble:
        state_dict = {key: val.clone() for key, val in lm.model.base_model.state_dict().items() if ("quant" not in key) and ("absmax" not in key)}
    else:
        state_dict = {key: val.clone() for key, val in lm.model.state_dict().items() if ("quant" not in key) and ("absmax" not in key)}
    pretrain_norm = compute_norm(state_dict)
    scale = pretrain_norm * args.scale
    print("Norm of the original model", pretrain_norm)

    project_matrix = lm.project_matrix 

    def customize_logistic_regression(gradients, outputs=None, labels=None, l2_strength=1e3):
        from scipy.optimize import minimize
        from sklearn.metrics import log_loss

        if outputs is not None:
            X = np.concatenate([gradients, -outputs.reshape(-1, 1)], axis=1)
        else:
            X = gradients

        def logistic_loss(variable_coefs):
            # Reinsert the fixed coefficient
            if outputs is not None:
                fixed_index = gradients.shape[1]; fixed_value = 1
                full_coefs = np.insert(variable_coefs, fixed_index, fixed_value)
            else:
                full_coefs = variable_coefs
            logits = X @ full_coefs.reshape(-1, 1)
            if labels is not None:
                probs = 1 / (1 + np.exp(-logits))
                loss = log_loss(labels, probs).mean()
            else:
                loss = np.log(1 + np.exp(-logits)).mean()

            # L2 penalty only on the variable coefficients
            l2_penalty = l2_strength * np.sum(variable_coefs ** 2)
            return loss + l2_penalty
        
        initial_guess = np.zeros(X.shape[1] - 1) if outputs is not None else np.zeros(X.shape[1])
        result = minimize(logistic_loss, initial_guess, method='BFGS', options={'maxiter': 10})
        print(result)

        # evaluate the trained model
        if labels is not None:
            accuracy = np.mean(np.round(1 / (1 + np.exp(-X @ result.x.reshape(-1, 1)))) == labels)
            print("Accuracy: ", accuracy)

        return result.x

    def fit_linear_model(gradients, outputs=None, labels=None, seed=0, use_customized_process=True):
        if use_customized_process:
            proj_coef = customize_logistic_regression(gradients, outputs=outputs, l2_strength=args.regularization_lambda)
            print("L2 norm before projection", np.linalg.norm(proj_coef))
        else:
            if labels is None:
                # randomly assign labels as 0 or 1
                labels = np.random.binomial(n=1, p=0.5, size=gradients.shape[0])
                # reverse the gradients for the 0 labels
                mask = np.copy(labels)
                mask[labels == 0] = -1
                mask = mask.reshape(-1, 1)
                gradients = gradients*mask
            else:
                ref_label = labels[0]
                origin_labels = np.copy(labels)
                labels[origin_labels == ref_label] = 1
                labels[origin_labels != ref_label] = -1

                mask = np.copy(labels)
                mask[labels == 0] = -1
                mask = mask.reshape(-1, 1)
                gradients = gradients*mask

            if outputs is None:
                # estimate parameters: train a logistic regression model
                clf = LogisticRegression(penalty='l2',  solver='lbfgs', C=1/args.regularization_lambda) 
                clf.fit(gradients, labels)
                print("Linear regression score: ", clf.score(gradients, labels))
                proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
                print("L2 norm before projection", np.linalg.norm(proj_coef))
            else:
                # concatenate outputs
                print("Also using outputs for linear regression")
                outputs = outputs*mask.flatten()
                gradients = np.concatenate([gradients, -outputs.reshape(-1, 1)], axis=1)
                # estimate parameters: train a logistic regression model
                clf = LogisticRegression(penalty='l2',  solver='lbfgs', C=1/args.regularization_lambda, fit_intercept=False) 
                clf.fit(gradients, labels)
                print("Linear regression score: ", clf.score(gradients, labels))
                proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)[:-1] # remove the last column corresponding to the output
                print("L2 norm before projection", np.linalg.norm(proj_coef))
            
        # convert the coefficients to the original space
        if seed != lm.current_project_seed:
            lm.initialize_project_matrix(seed=seed)
        project_matrix = lm.project_matrix
        coef = project_matrix @ proj_coef.flatten()
        print("L2 norm before scaling", np.linalg.norm(coef))

        # coef = coef * scale / np.linalg.norm(coef)
        # print("L2 norm after scaling", np.linalg.norm(coef))
        return coef

    def load_gradient_features(seed, gradient_idxes):
        cur_gradients_dir = gradients_dir.replace(f"seed_{args.compute_gradients_seeds[0]}", f"seed_{seed}")

        gradients = []
        for idx in gradient_idxes:
            if os.path.exists(f"./gradients/{cur_gradients_dir}/train_batch_{idx}_gradients.npy"):
                gradients.append(np.load(f"./gradients/{cur_gradients_dir}/train_batch_{idx}_gradients.npy"))
            else:
                print(f"Gradient for ./gradients/{cur_gradients_dir}/train_batch_{idx}_gradients.npy does not exist")
        gradients = np.concatenate(gradients, axis=0)
        return gradients

    def load_labels(seed, gradient_idxes):
        cur_gradients_dir = gradients_dir.replace(f"seed_{args.compute_gradients_seeds[0]}", f"seed_{seed}")

        labels = []
        for idx in gradient_idxes:
            if os.path.exists(f"./gradients/{cur_gradients_dir}/train_batch_{idx}_labels.npy"):
                labels.append(np.load(f"./gradients/{cur_gradients_dir}/train_batch_{idx}_labels.npy"))
            else:
                print(f"Gradient for ./gradients/{cur_gradients_dir}/train_batch_{idx}_labels.npy does not exist")
        labels = np.concatenate(labels, axis=0)
        return labels

    def load_outputs(seed, gradient_idxes):
        cur_gradients_dir = gradients_dir.replace(f"seed_{args.compute_gradients_seeds[0]}", f"seed_{seed}")

        outputs = []
        for idx in gradient_idxes:
            if os.path.exists(f"./gradients/{cur_gradients_dir}/train_batch_{idx}_outputs.npy"):
                outputs.append(np.load(f"./gradients/{cur_gradients_dir}/train_batch_{idx}_outputs.npy"))
            else:
                print(f"Gradient for ./gradients/{cur_gradients_dir}/train_batch_{idx}_outputs.npy does not exist")
        outputs = np.concatenate(outputs, axis=0)
        outputs = outputs[:, 0]
        return outputs

    def augment_gradients(gradients, num_copies=1):
        # augment the gradients by duplicating them
        augmented_gradients = [np.copy(gradients)]
        for i in range(num_copies):
            new_gradients_1 = np.copy(gradients); permute_1 = np.random.permutation(gradients.shape[0])
            new_gradients_2 = np.copy(gradients); permute_2 = np.random.permutation(gradients.shape[0])
            print("Permutation 1", permute_1); print("Permutation 2", permute_2)
            new_gradients = (new_gradients_1[permute_1] + new_gradients_2[permute_2]) / 2
            augmented_gradients.append(new_gradients)
        augmented_gradients = np.concatenate(augmented_gradients, axis=0)
        return augmented_gradients

    def gradient_based_estimation(task_idxes, repeat=1, seeds=[0]):
        if args.use_initialization:
            lm.model.load_state_dict(state_dict)
        else:
            gradient_idxes = []
            for task_idx in task_idxes:
                gradient_idxes += tasks_to_indices[task_idx]

            if args.use_ensemble:
                for trial in range(repeat):
                    # load gradients
                    all_gradients = load_gradient_features(seeds[trial], gradient_idxes)
                    all_outputs = load_outputs(seeds[trial], gradient_idxes)
                    if len(all_gradients) == 0:
                        return {}

                    # subsample gradients
                    rng = np.random.default_rng(trial)
                    if all_gradients.shape[0] > args.downsample:
                        choice_idxes = rng.choice(all_gradients.shape[0], args.downsample, replace=False)
                        gradients = all_gradients[choice_idxes]
                        outputs = all_outputs[choice_idxes]
                    else:
                        gradients = np.copy(all_gradients)
                        outputs = np.copy(all_outputs)

                    print("Number of gradients for logistic regression", gradients.shape)
                    coef = fit_linear_model(gradients, outputs=outputs, seed=seeds[trial])
                    new_state_dict = generate_state_dict(lm.model.base_model, state_dict, coef, device=lm.model.base_model.device)

                    lm.model.add_adapter(adapter="place_holder", weight=1/repeat, state_dict=new_state_dict)
                # evaluate task performances
                pretrain_state_dict = state_dict
                lm.model.base_model.load_state_dict(pretrain_state_dict)
            else:
                coefs = []
                for trial in range(repeat):
                    # load gradients
                    all_gradients = load_gradient_features(seeds[trial], gradient_idxes)
                    all_labels = load_labels(seeds[trial], gradient_idxes)
                    all_outputs = load_outputs(seeds[trial], gradient_idxes)
                    if len(all_gradients) == 0:
                        return {}
                    
                    # subsample gradients
                    rng = np.random.default_rng(4)
                    if all_gradients.shape[0] > args.downsample:
                        choice_idxes = rng.choice(all_gradients.shape[0], args.downsample, replace=False)
                        gradients = all_gradients[choice_idxes]
                        labels = all_labels[choice_idxes]
                        outputs = all_outputs[choice_idxes]
                    else:
                        gradients = np.copy(all_gradients)
                        outputs = np.copy(all_outputs)
                    
                    for i in range(2):
                        print("label distribution", np.sum(labels == i), "for label", i)

                    print("Number of gradients for logistic regression", gradients.shape)
                    coef = fit_linear_model(gradients, outputs=outputs, seed=seeds[trial])
                    coefs.append(coef)
                # average the coefficients
                coefs = np.array(coefs)
                coef = np.mean(coefs, axis=0)
                # coef = coef*scale / np.linalg.norm(coef)
                # print("L2 norm after scaling", np.linalg.norm(coef))
                
                # evaluate task performances
                new_state_dict = generate_state_dict(lm.model, state_dict, coef, device=lm.model.device)
                pretrain_state_dict = state_dict
                finetuned_state_dict = new_state_dict
                lm.model.load_state_dict(pretrain_state_dict)
                lm.model.load_state_dict(finetuned_state_dict, strict=False)

        # new data module 
        tmp_task_names = [args.task_names[idx] for idx in task_idxes]
        new_data_module = GLUEMultitaskDataModule(
            task_names=tmp_task_names,
            tokenizer=tokenizer,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            max_input_length=args.max_length,
            val_split_ratio=0.1,
            downsample_ratio=args.downsample_ratio,
            minimum_samples=args.minimum_samples,
            minimum_samples_validation=args.minimum_samples_validation)
        new_data_module.setup(stage="fit")

        summary = trainer.validate(lm, dataloaders=new_data_module.test_dataloader())[0]
        # summary = trainer.validate(lm, dataloaders=new_data_module.train_dataloader())[0]
        print(summary)
        return summary
    
    if args.load_sample_task_dir is not None:
        sampled_task_dir = os.path.join("./sampled_indices", "{}.txt".format(args.load_sample_task_dir))

        count = 0
        with open(sampled_task_dir, "r") as f:
            for line in f.readlines():
                task_idxes = [int(idx) for idx in line.strip().split()]
                task_idxes.sort()

                summary = gradient_based_estimation(task_idxes)
                if not summary:
                    continue

                # write results 
                for idx in task_idxes:
                    result_datapoint = {
                        "Task name": args.task_names[idx],
                        "Task index": idx,
                        "Task indices": " ".join([str(idx) for idx in task_idxes])
                    }
                    for key, val in summary.items():
                        if args.task_names[idx] in key:
                            tmp_key = key.removeprefix(args.task_names[idx] + "_")
                            result_datapoint[tmp_key] = val
                    file_name = os.path.join(file_dir, "results.csv")
                    add_result_to_csv(result_datapoint, file_name)
        # task_idxes = [0]
        # assert args.num_repeat == len(args.compute_gradients_seeds)
        # summary = gradient_based_estimation(task_idxes, repeat=args.num_repeat, seeds=args.compute_gradients_seeds)
    else:
        sampled_task_dir = os.path.join("./sampled_indices", "{}.txt".format(sampled_task_dir)); task_num = len(args.task_names)
        if not os.path.exists(sampled_task_dir):
            f = open(sampled_task_dir, "w")
            f.close()

        with open(sampled_task_dir, "r") as f:
            sampled_tasks = set()
            for line in f.readlines():
                sampled_tasks.add(line.rstrip("\n"))
        
        for _ in range(args.number_of_subsets):
            task_idxes = np.random.choice(task_num, int(args.subset_size*task_num), replace=False)
            task_idxes.sort()

            tmp_sampled_tasks = " ".join([str(idx) for idx in task_idxes])
            if tmp_sampled_tasks in sampled_tasks:
                continue
            summary = gradient_based_estimation(task_idxes)
            if not summary:
                continue

            # write results 
            for idx in task_idxes:
                result_datapoint = {
                    "Task name": args.task_names[idx],
                    "Task index": idx,
                    "Task indices": " ".join([str(idx) for idx in task_idxes])
                }
                for key, val in summary.items():
                    if args.task_names[idx] in key:
                        tmp_key = key.removeprefix(args.task_names[idx] + "_")
                        result_datapoint[tmp_key] = val
                file_name = os.path.join(file_dir, "results.csv")
                add_result_to_csv(result_datapoint, file_name)

            sampled_tasks.add(tmp_sampled_tasks)
            with open(sampled_task_dir, "a") as f:
                f.write(" ".join([str(idx) for idx in task_idxes]) + "\n")