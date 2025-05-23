# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.lqlora_utils import lora_utils
import torch

# Loads the base model (to CPU)
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

# %%
# Adds LoRA components, etc
model = lora_utils.prepare_model_for_lora(
    model=model,
    num_ranks=4,
    lora_alpha=32,
    lora_dropout=0.0,
    use_gradient_checkpointing=False)

# %%
# Applies LQ-LoRA to the model.
lora_utils.transform_lora_layers(
    lpq=False,
    model=model,
    model_name="nf3",
    device="cuda")

# %%
# %%
from src.custom.glue_data_module import GLUEDataModule

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

tokenizer.pad_token = tokenizer.eos_token

data_module = GLUEDataModule(
        task_name="cola",
        tokenizer=tokenizer,
        batch_size=8,
        inference_batch_size=None,
        max_input_length=256)
data_module.setup(stage="fit")


# %%
train_dataloader = data_module.train_dataloader()
for batch in train_dataloader:
    print(batch)
    break

# %%
for key, tensor in batch.items():
    batch[key] =  tensor.to("cuda:0")

# %%
model.to("cuda:0")
model(**batch)



# %%
# from models import allocation_utils as allocation_utils_LLaMA

# allocation_utils_LLaMA.create_qconfig_and_sensitivity_dict_LLaMA(
#                 identifier=f"llama-2-{model_size}/lpq-64/{data},budget={budget}")

# # %%
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type='nf4'
#     )
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-70B", quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map={"": 0}) #

# # %%
# import torch

# outputs = torch.load("./artifacts/llama-2-70b-1024.ilp.ranks-None.data-True.pth")
