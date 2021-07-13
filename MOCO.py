import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import PIL
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from torchvision.datasets import DatasetFolder, ImageFolder
from shuffle_batchnorm import ShuffleBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.datasets.utils import download_url
import tarfile
import hashlib
from model import MOCO, plot


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
print(f"num cuda devices: {torch.cuda.device_count()}")

# dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
# dataset_filename = dataset_url.split('/')[-1]
# dataset_foldername = dataset_filename.split('.')[0]
# data_path = './data'
# dataset_filepath = os.path.join(data_path,dataset_filename)
# dataset_folderpath = os.path.join(data_path,dataset_foldername)

# os.makedirs(data_path, exist_ok=True)

# download = False
# if not os.path.exists(dataset_filepath):
#     download = True
# else:
#     md5_hash = hashlib.md5()
#     file = open(dataset_filepath, "rb")
#     content = file.read()
#     md5_hash.update(content)
#     digest = md5_hash.hexdigest()
#     if digest != 'fe2fc210e6bb7c5664d602c3cd71e612':
#         download = True
# if download:
#     download_url(dataset_url, data_path)
#     with tarfile.open(dataset_filepath, 'r:gz') as tar:
#         tar.extractall(path=data_path)



train_root = "./data/imagenette2/train"
val_root = "./data/imagenette2/val"

    
    
def crop_transform(x):
    q = transform(x)
    k = transform(x)
    return [q, k]

transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2), 
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])



feature_dim = 128
K = 16000
t = 0.2
m = 0.999
model = MOCO(t, m, feature_dim).to(device)
checkpoint = torch.load("MOCOv2.pth")
model.load_state_dict(checkpoint['model_state_dict'])

lr = 0.001
BATCH_SIZE = 64
train_data = ImageFolder(root=train_root, transform=crop_transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size= BATCH_SIZE, drop_last=True, num_workers = 2, pin_memory=True)
optimizer = torch.optim.Adam(model.f_q.parameters(), lr=lr)

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

criterion = nn.CrossEntropyLoss()
EPOCHS = 500

batches_in_queue = K/BATCH_SIZE
queue =(F.normalize(torch.randn(feature_dim, K), dim=0)/10).to(device)
queue = checkpoint['queue']
loss_list=[]
labels = torch.zeros(BATCH_SIZE, dtype=torch.long).to(device)


best_loss = 100
cnt_since_best = 0
for epoch in tqdm(range(EPOCHS)):
    epoch_loss_list = []
    correct = total = total_loss = 0
    for i, batch in enumerate(train_loader):
        images, _ = batch
        total += BATCH_SIZE
        q, k = images[0].to(device), images[1].to(device)
        logits, k = model(q, k, queue)
        k = k.detach()
        predictions = torch.argmax(logits, dim = 1)
        correct += torch.sum(predictions == labels).item()
         # positives are the 0-th
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        epoch_loss_list.append(loss.item())
        optimizer.step()
        model.momentum_update()
        idx_replace = int(i%batches_in_queue)
        queue[:, idx_replace*BATCH_SIZE:(idx_replace+1)*BATCH_SIZE] = k.T
        
    epoch_loss = np.mean(epoch_loss_list)
    if epoch_loss < best_loss and epoch not in [0,1]:
        best_loss = epoch_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'queue': queue,
            },"MOCOv2.pth")
        cnt_since_best = 0
    else:
        cnt_since_best += 1
    if cnt_since_best == 25:
        lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"\n reduce learning rate to: {lr}")       
    loss_list.append(epoch_loss)
    plot(loss_list)
    print(f"epoch: {epoch}\n loss: {epoch_loss:.3f}\n accuracy: {correct/total:.3f}")

    
    
    
# Linear evaluation 


# train_eval_transform = transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
# 	])
# test_eval_transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
# 	])


# for param in model.parameters():
#     param.requires_grad = False

# model.f_q.fc = nn.Linear(2048, 10).to(device)
# model.f_q.fc.requires_grad = True
# # last_fc = nn.Linear(feature_dim, 10).to(device)
    
# train_data = ImageFolder(root=train_root, transform=train_eval_transform)
# val_data = ImageFolder(root=val_root,transform=test_eval_transform)
# train_loader = DataLoader(train_data, shuffle=True, batch_size= BATCH_SIZE,num_workers = 4, pin_memory=True)
# val_loader = DataLoader(val_data, shuffle=False, batch_size= BATCH_SIZE, num_workers = 4, pin_memory=True)
# lr = 0.003
# optimizer = torch.optim.Adam(model.f_q.fc.parameters(), lr=lr)

# EPOCHS = 100
# res_dict = {"train_loss_list":[],"test_loss_list":[],"train_acc_list":[],"test_acc_list":[]}


# print("START LINEAR EVALUATION \n")
# best_acc = 0
# for epoch in tqdm(range(EPOCHS)):
#     correct = total = total_loss = 0
#     for i, (images, labels) in enumerate(train_loader):
#         labels = labels.to(device)
#         scores = model.f_q(images.to(device))
#         logits = F.normalize(scores, dim=1)
#         predictions = torch.argmax(logits, dim = 1)
#         correct += torch.sum(predictions == labels).item()
#         total += labels.shape[0]
#         loss = criterion(logits, labels)
#         total_loss += loss.item() 
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     res_dict["train_loss_list"].append(total_loss/total)
#     res_dict["train_acc_list"].append(correct/total)
#     print(f"\n train loss: {total_loss/total:.3f} train accuracy:{correct/total:.3f} ")
#     with torch.no_grad():
#         correct = total = total_loss = 0
#         for i, (images, labels) in enumerate(val_loader):
#             labels = labels.to(device)
#             scores = model.f_q(images.to(device))
#             logits = F.normalize(scores, dim=1)
#             predictions = torch.argmax(logits, dim = 1)
#             correct += torch.sum(predictions == labels).item()
#             total += labels.shape[0]
#             loss = criterion(logits, labels)
#             total_loss += loss.item() 
#         res_dict["test_loss_list"].append(total_loss/total)
#         res_dict["test_acc_list"].append(correct/total)

#     if (correct/total)>best_acc:
#         best_acc = correct/total
#         torch.save(model.f_q.fc.state_dict(), "linear_probe.pth")
#     plot(res_dict, eval=True)
#     print(f"test loss: {total_loss/total:.3f} test accuracy:{correct/total:.3f} ")
