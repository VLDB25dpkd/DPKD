from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer
import time



datasets = load_dataset("glue", "mnli", cache_dir="../openweb")
datasets = datasets["train"]
print(len(datasets))
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

def process_function(examples):
    return tokenizer(
        # examples["question"],
        examples["premise"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )


tokenized_data = datasets.map(process_function, batched=True)
tokenized_data.save_to_disk("./data/tokenized/mnli_p")


