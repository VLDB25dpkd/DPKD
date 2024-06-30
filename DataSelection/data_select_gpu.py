from pickle import FALSE
from random import shuffle
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from datasets import load_from_disk, load_dataset
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def entropy(px):
    return -torch.sum(F.softmax(px, dim=1) * F.log_softmax(px, dim=1), dim=1)


model_path = "../classifier/model/dp/sst2_distilgpt2_classifier_eps1_epoch3"
# datasets=load_dataset("openwebtext",cache_dir="/data1/lwx/NLP/openweb")
# datasets=datasets["train"]

datasets = load_from_disk("../classifier/data/tokenized/openweb")
datasets.set_format("torch")
datasets = datasets.remove_columns(["text"])
# print(datasets[0])
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2ForSequenceClassification.from_pretrained(model_path)


# 读dic

embedding_to_ids = {}
embedding_to_ids.update(
    np.load("../DataQuanlity/dict/embedding_to_idx1.npy", allow_pickle=True).item()
)
print("dict1")
embedding_to_ids.update(
    np.load("../DataQuanlity/dict/embedding_to_idx2.npy", allow_pickle=True).item()
)
print("dict2")
embedding_to_ids.update(
    np.load("../DataQuanlity/dict/embedding_to_idx3.npy", allow_pickle=True).item()
)
print("dict3")
embedding_to_ids.update(
    np.load("../DataQuanlity/dict/embedding_to_idx4.npy", allow_pickle=True).item()
)
print("dict4")


print(len(embedding_to_ids))


def process_batch(batch):
    # Move the batch to the GPU
    # for key in batch.keys():
    #     batch[key] = batch[key].to(device)
    batch = {k: v.to("cuda") for k, v in batch.items()}
    # Run the model
    with torch.no_grad():
        outputs = model(**batch)
    return entropy(outputs.logits).tolist()


batch_size = 300
steps_per_save = 10000
device = torch.device("cuda")
model.to(device)
model = torch.nn.DataParallel(model)

for i in range(0, 10):
    embeddings = np.load("../DataQuanlity/random_cluster/cluster" + str(i) + ".npy")
    print(f"get embedding cluster {i}")

    # indices = [embedding_to_ids[str(embedding)] for embedding in embeddings]
    indices = []
    for j in tqdm(range(len(embeddings))):
        # for j in tqdm(range(10000)):
        indices.append(embedding_to_ids[str(embeddings[j])])

    # Get all texts for this cluster at once
    texts = datasets.select(indices)
    # texts = cluster_data
    print(f"get embedding cluster {i}")

    # Create a DataLoader to handle batching of inputs
    data_loader = DataLoader(texts, batch_size=batch_size, shuffle=FALSE)

    entropys = []
    for step, batch in enumerate(tqdm(data_loader)):
        # print(batch)
        entropys += process_batch(batch)
    np.save(f"./random/sst2/entropys/cluster{i}.npy", np.array(entropys))
    print(f"Finished processing cluster {i}")
