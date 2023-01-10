import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

heads = 1


class Breadth(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Breadth, self).__init__()
        self.gatconv = GATConv(in_dim, out_dim, heads=heads)

    def forward(self, x, edge_index):
        x = torch.tanh(self.gatconv(x, edge_index))
        return x


class Depth(torch.nn.Module):

    def __init__(self, in_dim, hidden):
        super(Depth, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)


class GeniePathLayer(torch.nn.Module):

    def __init__(self, in_dim, dim=256, lstm_hidden=256):
        super(GeniePathLayer, self).__init__()
        self.breadth_func = Breadth(in_dim, dim)
        self.depth_func = Depth(dim, lstm_hidden)

    def forward(self, x, edge_index, h, c):
        x = self.breadth_func(x, edge_index)
        x = x[None, :]
        x, (h, c) = self.depth_func(x, h, c)
        x = x[0]
        return x, (h, c)


# class GeniePath(torch.nn.Module):

#     def __init__(self,
#                  in_feats,
#                  out_feats,
#                  dim=256,
#                  lstm_hidden=256,
#                  layer_num=4):
#         super(GeniePath, self).__init__()
#         self.lstm_hidden = lstm_hidden
#         self.lin1 = torch.nn.Linear(in_feats, dim)
#         self.gplayers = torch.nn.ModuleList([
#             GeniePathLayer(dim, dim=dim, lstm_hidden=lstm_hidden)
#             for i in range(layer_num)
#         ])
#         self.lin2 = torch.nn.Linear(dim, out_feats)

#     def forward(self, x, edge_index,*args,**kwargs):
#         x = self.lin1(x)
#         h = torch.zeros(1, x.shape[0], self.lstm_hidden).to(x)
#         c = torch.zeros(1, x.shape[0], self.lstm_hidden).to(x)
#         for i, l in enumerate(self.gplayers):
#             x, (h, c) = self.gplayers[i](x, edge_index, h, c)
#         x = self.lin2(x)
#         return x


class GeniePathLazy(torch.nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 dim=48,
                 lstm_hidden=48,
                 layer_num=3):
        super(GeniePathLazy, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.lin1 = torch.nn.Linear(in_feats, dim)
        self.breaths = torch.nn.ModuleList(
            [Breadth(dim, dim) for i in range(layer_num)])
        self.depths = torch.nn.ModuleList(
            [Depth(dim * 2, lstm_hidden) for i in range(layer_num)])
        self.lin2 = torch.nn.Linear(dim, out_feats)

    def forward(self, x, edge_index, *args, **kwargs):
        x = self.lin1(x)
        h = torch.zeros(1, x.shape[0], self.lstm_hidden).to(x)
        c = torch.zeros(1, x.shape[0], self.lstm_hidden).to(x)
        h_tmps = []
        for i, l in enumerate(self.breaths):
            h_tmps.append(self.breaths[i](x, edge_index))
        x = x[None, :]
        for i, l in enumerate(self.depths):
            in_cat = torch.cat((h_tmps[i][None, :], x), -1)
            x, (h, c) = self.depths[i](in_cat, h, c)
        x = self.lin2(x[0])
        return x


def geniepath(in_feats, out_feats, **kwargs):
    return GeniePathLazy(in_feats, out_feats, **kwargs)


def geniepath_big(in_feats, out_feats, **kwargs):
    return GeniePathLazy(in_feats,
                         out_feats,
                         dim=128,
                         lstm_hidden=128,
                         layer_num=4,
                         **kwargs)
