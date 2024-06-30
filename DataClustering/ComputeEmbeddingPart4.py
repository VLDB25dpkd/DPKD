import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import (
    BertModel,
    BertForMaskedLM,
    BertForPreTraining,
    BertForSequenceClassification,
)
from transformers import AutoModel
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
import time


tokenized_data = load_from_disk("./tokenized_openweb")
print(len(tokenized_data))
tokenized_data = tokenized_data.select(range(6000000, len(tokenized_data)))
print(len(tokenized_data))
tokenized_data = tokenized_data.remove_columns(["text"])

datasets = load_dataset
embeddings = None
model = AutoModel.from_pretrained("bert-base-uncased")
model = model.to("cuda")
model.eval()

train_dataloader = DataLoader(tokenized_data, shuffle=False, batch_size=512)

embedding_to_idx = {}
idx = 6000000
print("compute embedding")
for i, batch in enumerate(tqdm(train_dataloader)):
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    hidden_embedding = output[0].mean(dim=1).cpu().numpy()
    for emdedding in hidden_embedding:
        embedding_to_idx[str(emdedding)] = idx
        idx += 1
    if embeddings is None:
        embeddings = hidden_embedding
    else:
        embeddings = np.concatenate((embeddings, hidden_embedding), axis=0)

np.save("./embeddings/openweb_embeddings4.npy", embeddings)
np.save("./dict/embedding_to_idx4.npy", embedding_to_idx)
