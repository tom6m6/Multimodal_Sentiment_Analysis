from torch import nn
import torch

class fusion(nn.Module):
    def __init__(self, vdim, tdim, D):
        super(fusion, self).__init__()

        self.vfc = nn.Linear(vdim, D)
        self.tfc = nn.Linear(tdim, D)
        self.output = nn.Linear(2 * D, 3)

        self.bn1 = nn.BatchNorm1d(D)
        self.bn2 = nn.BatchNorm1d(D)
        #self.drop1 = nn.Dropout(0.3)
        #self.drop2 = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, v_emb, t_emb):
        #v_emb = self.drop1(self.relu(self.bn1(self.vfc(v_emb))))
        #t_emb = self.drop2(self.relu(self.bn2(self.tfc(t_emb))))
        v_emb = self.relu(self.bn1(self.vfc(v_emb)))
        t_emb = self.relu(self.bn2(self.tfc(t_emb)))
        fusion = torch.cat((v_emb, t_emb), axis=1)
        return self.output(fusion)