import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn


class GCN(torch.nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_channels=384,
                 num_layers=3,
                 dropout=0.0,
                 batchnorm=False):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_feats, hidden_channels, cached=True))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.fc = GCNConv(hidden_channels, out_feats, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index, *args, **kwargs):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x, edge_index)


def gcn(in_feats, out_feats, **kwargs):
    return GCN(in_feats, out_feats, **kwargs)


def gcn_bn(in_feats, out_feats, **kwargs):
    return GCN(in_feats, out_feats, batchnorm=True, **kwargs)


def gcn_bn_drop05(in_feats, out_feats, **kwargs):
    return GCN(in_feats, out_feats, batchnorm=True, dropout=0.05, **kwargs)


def gcn_bn_drop10(in_feats, out_feats, **kwargs):
    return GCN(in_feats, out_feats, batchnorm=True, dropout=0.1, **kwargs)


def gcn_bn_drop30(in_feats, out_feats, **kwargs):
    return GCN(in_feats, out_feats, batchnorm=True, dropout=0.3, **kwargs)


if __name__ == "__main__":
    model = gcn_bn_drop30(93, 2)  # 186242
    print(sum(p.numel() for p in model.parameters()))