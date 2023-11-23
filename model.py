import math

import torch
import torch.nn.functional as F
from torch import nn


class Rnn_Model(nn.Module):
    def __init__(self,base_model,num_classes,input_size):
        super().__init__
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size    
        self.Rnn = nn.RNN(input_size=input_size,
                          hidden_size=320,
                          num_layers = 1,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320,80),
                                nn.Linear(80,20),
                                nn.Linear(20,self.num_classes),
                                nn.Softmax(dim=1))
        
        for param in base_model.parameters():
            param.requires_grad = True

    def forward(self,inputs):
        # 上游任务
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        # 下游任务
        outputs,_ = self.Rnn(cls_feats)
        outputs = outputs[:,-1,:]
        outputs = self.fc(outputs)
        return outputs        