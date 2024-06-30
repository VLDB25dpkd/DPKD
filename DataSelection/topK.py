import numpy as np
from datasets import load_dataset
from tqdm import tqdm


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


for idx in range(0, 10):
    print("idx")
    print(idx)
    entropys = np.load("./random/sst2/entropys/cluster" + str(idx) + ".npy")
    print(entropys.shape)
    list_entropy = []
    i = 0
    for entropy in entropys:
        list_entropy.append([entropy, i])
        i += 1

    list_entropy = sorted(list_entropy, reverse=True)[:50000]
    embeddings = np.load("../DataQuanlity/random_cluster/cluster" + str(idx) + ".npy")
    print("embedding length")
    print(len(embeddings))

    indices = []
    print(len(list_entropy))
    for j in tqdm(range(len(list_entropy))):
        # for j in tqdm(range(10000)):
        indices.append(embedding_to_ids[str(embeddings[list_entropy[j][1]])])

    datasets = load_dataset("openwebtext", cache_dir="../openweb")
    datasets = datasets["train"]
    datasets = datasets.select(indices)

    print("save raw data")
    with open("./random/sst2/cluster" + str(idx) + ".txt", "a") as f:
        for i in tqdm(range(len(datasets))):
            f.write(str(datasets[i]["text"]).replace("\n", " ") + "\n")
