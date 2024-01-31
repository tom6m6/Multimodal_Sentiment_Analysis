from torch import nn
import torch
from torchvision import models
from transformers import RobertaModel

class single(nn.Module):
    def __init__(self, dim, D):
        super(single, self).__init__()
        self.resnet = models.__dict__['resnet50'](pretrained=True)
        self.roberta = RobertaModel.from_pretrained('./model_roberta')

        self.fc = nn.Linear(dim, D)
        self.output = nn.Linear(D, 3)

        self.bn = nn.BatchNorm1d(D)
        #self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, fused, emb1, emb2 = None):
        if(fused == 1):
            emb = self.roberta(
                input_ids=emb1,
                attention_mask=emb2
            )
            emb = emb.pooler_output
        else:
            emb = self.resnet(emb1)

        #out = self.drop(self.relu(self.bn(self.fc(emb))))
        out = self.relu(self.bn(self.fc(emb)))
        return self.output(out)