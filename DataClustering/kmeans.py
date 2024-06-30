from sklearn.cluster import KMeans
import numpy as np
import time

embeddings1 = np.load("./embeddings/openweb_embeddings1.npy")
embeddings2 = np.load("./embeddings/openweb_embeddings2.npy")
embeddings3 = np.load("./embeddings/openweb_embeddings3.npy")
embeddings4 = np.load("./embeddings/openweb_embeddings4.npy")
embeddings = np.vstack((embeddings1, embeddings2, embeddings3, embeddings4))

print(embeddings.shape)
start = time.time()
kmeans = KMeans(n_clusters=2, verbose=1)
res = kmeans.fit(embeddings)
end = time.time()
print(end - start)


labels = kmeans.labels_
print(labels)
for cluster_id in range(kmeans.n_clusters):
    cluster_data = embeddings[labels == cluster_id]
    print(f"Cluster {cluster_id + 1}:")
    print(cluster_data.shape)
    np.save("./kmeans_40/cluster" + str(cluster_id) + ".npy", cluster_data)
