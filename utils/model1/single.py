from torch import nn
import torch

class single(nn.Module):
    def __init__(self, dim, D):
        super(single, self).__init__()

        self.fc = nn.Linear(dim, D)
        self.output = nn.Linear(D, 3)

        self.bn = nn.BatchNorm1d(D)
        #self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, emb):
        #out = self.drop(self.relu(self.bn(self.fc(emb))))
        out = self.relu(self.bn(self.fc(emb)))
        return self.output(out)