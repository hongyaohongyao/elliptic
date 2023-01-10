import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_channels=256,
                 num_layers=3,
                 dropout=0.0,
                 batchnorm=False):
        super(MLP, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(torch.nn.Linear(in_feats, hidden_channels))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.linears.append(
                torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.fc = torch.nn.Linear(hidden_channels, out_feats)

        self.dropout = dropout

    def forward(self, x, *args, **kwargs):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x)


def mlp(in_feats, out_feats, **kwargs):
    return MLP(in_feats, out_feats, **kwargs)


def mlp_bn(in_feats, out_feats, **kwargs):
    return MLP(in_feats, out_feats, batchnorm=True, **kwargs)


def mlp_bn_drop05(in_feats, out_feats, **kwargs):
    return MLP(in_feats, out_feats, batchnorm=True, dropout=0.05, **kwargs)


def mlp_bn_drop10(in_feats, out_feats, **kwargs):
    return MLP(in_feats, out_feats, batchnorm=True, dropout=0.1, **kwargs)

def mlp_bn_drop30(in_feats, out_feats, **kwargs):
    return MLP(in_feats, out_feats, batchnorm=True, dropout=0.3, **kwargs)