from functools import partial
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 使用split方法将每个单词提取出来作为后续bertToken的输入
class MyDataset(Dataset):
    def __init__(self,sentences,labels,method_name,model_name):
        self.sentences = sentences
        self.labels = labels
        self.method_name = method_name
        self.model_name = model_name
        dataset = list()
        index = 0

        for data in sentences:
            tokens = data.split(' ')
            labels_id = labels[index]
            index += 1
            dataset.append((tokens,labels_id))
        self._dataset = dataset
    
    def __getitem__(self, index):
        return self._dataset[index]
    
    def __len__(self):
        return len(self.sentences)
    

#  对每一个batch的数据进行处理

def my_collate(batch,tokenizer):
    tokens,label_ids = map(list,zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=320,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    return text_ids,torch.tensor(label_ids)

def load_dataset(tokenizer,train_batch_size,test_batch_size,model_name,method_name,workers):
    data = pd.read_csv('datasets.csv',sep=None,header=0,encoding='utf-8',engine='python')
    len1 = int(len(list(data['labels']))*0.1)
    labels = list(data['labels'])[0:len1]
    sentences = list(data['sentences'])[0:len1]

    tr_sen,te_sen,tr_lab,te_lab = train_test_split(sentences,labels,train_size=0.8)

    train_set = MyDataset(tr_sen,tr_lab,method_name,model_name)
    test_set = MyDataset(te_sen,te_lab,method_name,model_name)

    collate_fn = partial(my_collate,tokenizer=tokenizer)

    train_loader = DataLoader(train_set,batch_size=train_batch_size,shuffle=True,num_workers=workers,collate_fn=collate_fn,pin_memory=True)

    test_loader = DataLoader(test_set,batch_size=test_batch_size,shuffle=True,num_workers=workers,collate_fn=collate_fn,pin_memory=True)
    return train_loader,test_loader