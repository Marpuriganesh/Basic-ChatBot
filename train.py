import json
from nltk_utils import tokenize,stem,bag_of_word
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open('data.json', 'r') as f:
    data = json.load(f)

# print(data)

all_words = []
tags = []
xy = []
for intents in data['intents']:
    tag = intents['tag']
    tags.append(tag)
    for pattern in intents['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(tags)
# print(all_words)

X_train = []
y_train = []
for(patter_sentence, tags) in xy:
    bag = bag_of_word(patter_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)  #crossEntropy loss

    X_train = np.array(X_train)
    y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # dataset[i] 
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

   
    def __len__(self):
        return self.n_samples

# HyperParameters

batch_size = 8

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)


