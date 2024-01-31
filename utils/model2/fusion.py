from torch import nn
import torch
from torchvision import models
from transformers import RobertaModel

class fusion(nn.Module):
    def __init__(self, vdim, tdim, D):
        super(fusion, self).__init__()
        self.resnet = models.__dict__['resnet50'](pretrained=True)
        self.roberta = RobertaModel.from_pretrained('./model_roberta')

        self.vfc = nn.Linear(vdim, D)
        self.tfc = nn.Linear(tdim, D)
        self.output = nn.Linear(2 * D, 3)

        self.bn = nn.BatchNorm1d(D)
        #self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, image, input_ids, attention_mask):
        v_emb = self.resnet(image)
        t_emb = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        t_emb = t_emb.pooler_output

        #v_emb = self.drop(self.relu(self.bn(self.vfc(v_emb))))
        #t_emb = self.drop(self.relu(self.bn(self.tfc(t_emb))))

        v_emb = self.relu(self.bn(self.vfc(v_emb)))
        t_emb = self.relu(self.bn(self.tfc(t_emb)))

        fusion = torch.cat((v_emb, t_emb), axis=1)

        return self.output(fusion)