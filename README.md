# CDS
The code for model compression via data selection

## Requirements:
  torch=2.1.2  
  datasets=2.18.0  
  opacus=1.4.0  
  scikit-learn=1.3.2  
  transformers=4.38.2  

## 1. Train domain classifier(DomainClassifier)
First, tokenize the public and private data:

    python prepare_data_glue.py
    python prepara_data_openweb.py
  
You should modify the dataset name and save_path in both .py file.  
Then, train the domain classifier with DPSGD:

    python train_classifier_gpu.py

You should modify the path of tokenized data and the save path of model in .py file.  

## 2. Clustering the embeddings(DataClustering)
Since the openweb dataset is large, we split it into 4 parts, and compute the embedding of each data as follow:

    python ComputeEmbeddingPart1.py
    python ComputeEmbeddingPart2.py
    python ComputeEmbeddingPart3.py
    python ComputeEmbeddingPart4.py

You should modify the path of tokenized data and the save path of embeddings.  
Then, clustering the emebddings:

    python Kmeans.py

You should modify the path of embeddings, cluster count of KMEANS and the save path of clustering result.

## 3. Select data via doamin classifier(DataSelction)
First, evaluate each data with domain classifier:

    python data_select_gpu.py

You should modify the path of domain classifier, clustering result and embeddings. And you should modify the cluster count in the main loop.

Then, select the topK data:

    python topK.py

You should modify the path of domain classifier, entropys and embeddings. And you should modify the cluster count in the main loop.
Finnaly, combine the topK data into a txt:

    python combine_data.py
You should modify the cluster count in the main loop and the path of topk data.

## 4. Compression the target model with selected data(distillation)
Since the data is DP selected from public data, we can directly use the distillation code of huggingface (https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) of compression the target model. 

You should use the combined .txt file of step 4 as the training data, and extract some layers' weight of teacher model to initialize the student model.

## 5. Compute the corresponding noise multiplier of DP finetuning(get_sigma.py)
    python get_sigma.py
You should modify the target_epsilon, dataszie (the size of dataset), batch size and epochs.

## 6. DP finetuning the student model with private data(Finetuning)
We use the finetuning code from huggingface(https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification), and we make a DP modification on it. You can use the run_glue_no_trainer.py in our repository to replace the original code in transoforers. And run the .py file with the .sh file we provided. Noting that you should replace the model path in .sh file with the actual path of the student model.

