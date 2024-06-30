
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from tqdm import tqdm
import os
import evaluate
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import torch.distributed as dist
import argparse

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

parser = argparse.ArgumentParser(
    description="Opacus MNIST Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
)
parser.add_argument(
    "--c",
    type=float,
    default=0.1,
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
dp_able=True



model = GPT2ForSequenceClassification.from_pretrained('distilgpt2', num_labels=2)
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
# model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
encoded_inputs1=load_from_disk("./data/tokenized/sst2").shuffle().select(range(50000))
encoded_inputs2=load_from_disk("./data/tokenized/openweb_100w").shuffle().select(range(200000))
inputs_ids=torch.cat((torch.tensor(encoded_inputs1['input_ids']),torch.tensor(encoded_inputs2['input_ids'])),0)
attention_masks=torch.cat((torch.tensor(encoded_inputs1['attention_mask']),torch.tensor(encoded_inputs2['attention_mask'])),0)
labels1=torch.ones([len(encoded_inputs1)],dtype=torch.long)
labels2=torch.zeros([len(encoded_inputs2)],dtype=torch.long)
labels=torch.cat((labels1,labels2),0)

print(labels.shape)
print(inputs_ids.shape)
print(attention_masks.shape)
# time.sleep(100)

dataset = torch.utils.data.TensorDataset(inputs_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=args.lr)

model.train()

if dp_able:
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        # noise_multiplier=,
        max_grad_norm=args.c,
        target_epsilon=1.0,
        epochs=3,
        target_delta=1024/250000,
        
    )

device="cuda:0"
# model=DataParallel(model)
model.to(device)
model.train()

for name, param in model.named_parameters():
    if "wpe" in name:
        print(f"Parameter Name: {name}, Shape: {param.shape}")
        param.requires_grad=False
        
for epoch in range(3):
    with BatchMemoryManager(
        data_loader=dataloader, 
        max_physical_batch_size=8, 
        optimizer=optimizer
    ) as memory_safe_data_loader:
        for batch in tqdm(memory_safe_data_loader):
            input_ids, attention_mask, target = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
            loss = outputs.loss
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()

model._module.save_pretrained("./model/dp/distilgpt2_classifier_eps1")



encoded_inputs1=load_from_disk("./data/tokenized/sst2").select(range(50000,60000))
encoded_inputs2=load_from_disk("./data/tokenized/openweb_100w").select(range(800000,900000))
inputs_ids=torch.cat((torch.tensor(encoded_inputs1['input_ids']),torch.tensor(encoded_inputs2['input_ids'])),0)
attention_masks=torch.cat((torch.tensor(encoded_inputs1['attention_mask']),torch.tensor(encoded_inputs2['attention_mask'])),0)
labels1=torch.ones([len(encoded_inputs1)],dtype=torch.long)
labels2=torch.zeros([len(encoded_inputs2)],dtype=torch.long)
labels=torch.cat((labels1,labels2),0)

print(labels.shape)
print(inputs_ids.shape)
print(attention_masks.shape)


test_dataset = torch.utils.data.TensorDataset(inputs_ids, attention_masks, labels)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
model.eval()
metric = evaluate.load("accuracy")
for batch in tqdm(test_dataloader):
    input_ids, attention_mask, target = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target = target.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    # print(predictions)
    # print(target)
    metric.add_batch(predictions=predictions, references=target)
score=metric.compute()
print(score)
file=open("./result.txt","a")
file.write(str(score))
file.close()

