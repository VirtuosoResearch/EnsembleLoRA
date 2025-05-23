# %%
import argparse
import logging
import os

from src.custom.data_module import DataModule
from src.data.completion_dataset import CompletionMetadata

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

from src.custom.model import Model
from torch.utils.data import DataLoader
import numpy as np

import time

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

model_key = "google/gemma-2b" # "EleutherAI/gpt-neo-1.3B"
device = 0

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
    )

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_key,
    quantization_config=quantization_config, 
    torch_dtype=torch.bfloat16,
    device_map={"": 2}
)
tokenizer = AutoTokenizer.from_pretrained(model_key, model_max_length=256)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
from src.custom.alpaca_data_module import AlpacaDataModule

data_module = AlpacaDataModule(tokenizer=tokenizer,
                               data_path="./data/alpaca_data/alpaca_final.pkl",
                               dev_split_path="./data/alpaca_data/alpaca_dev_split_map.pkl",
                               task_idxes=list(range(38)),
                               batch_size = 8,
                               inference_batch_size = 8,
                               context_length=256)
data_module.setup(stage="fit")
# %%
train_dataloader = data_module.train_dataloader()

for batch in train_dataloader:
    break

model_4bit.eval()
# model_4bit.to("cuda")
batch = {k: v.to("cuda:2") for k, v in batch.items()}
output_4bit = model_4bit(**batch)