import csv
import os
import re

import numpy as np
import pandas as pd
import torch


def get_data():
    pos1,pos2 = os.listdir('/home/wyx/dataset/aclImdb/test/pos'),os.listdir('/home/wyx/dataset/aclImdb/train/pos')
    neg1,neg2 = os.listdir('/home/wyx/dataset/aclImdb/test/neg'),os.listdir('/home/wyx/dataset/aclImdb/train/neg')
    pos_all,neg_all = [],[]
    for p1,n1 in zip(pos1,neg1):
        with open('/home/wyx/dataset/aclImdb/test/pos/'+p1,encoding='utf8') as f:
            pos_all.append(f.read())
        with open('/home/wyx/dataset/aclImdb/test/neg/'+n1,encoding='utf8') as f:
            neg_all.append(f.read())
    for p2,n2 in zip(pos2,neg2):
        with open('/home/wyx/dataset/aclImdb/train/pos/'+p2,encoding='utf8') as f:
            pos_all.append(f.read())
        with open('/home/wyx/dataset/aclImdb/train/neg/'+n2,encoding='utf8') as f:
            neg_all.append(f.read())
    
    datasets = np.array(pos_all+neg_all)
    labels = np.array([1]*25000+[0]*25000)
    return datasets,labels

# 打乱数据
def shuffle_process():
    sentences,lables = get_data()
    shuffle_indexs = np.random.permutation(len(sentences))
    datasets = sentences[shuffle_indexs]
    lables = lables[shuffle_indexs]
    return datasets,lables


# 数据去除标点符号
def save_process():
    datasets,labels = shuffle_process()
    sentences = []
    punc = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]'
    for sen in datasets:
        sen = sen.replace('\n', '')
        sen = sen.replace('<br /><br />', ' ')
        sen = re.sub(punc, '', sen)
        sentences.append(sen)
    # Save
    df = pd.DataFrame({'labels': labels, 'sentences': sentences})
    df.to_csv("datasets.csv", index=False)


if __name__ == '__main__':
    model = save_process()