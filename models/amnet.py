import torch
import torch.nn.functional as F
import torch.nn as nn
from .bernconv import BernConv


class AMNet(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 hid_channels=156,
                 K=2,
                 filter_num=4,
                 dropout=0.0):
        super(AMNet, self).__init__()
        self.act_fn = nn.ReLU()
        self.attn_fn = nn.Tanh()
        self.linear_transform_in = nn.Sequential(
            nn.Linear(in_feats, hid_channels),
            self.act_fn,
            nn.Linear(hid_channels, hid_channels),
        )
        self.K = K
        self.filters = nn.ModuleList([
            BernConv(hid_channels, K, normalization=True, bias=True)
            for _ in range(filter_num)
        ])
        self.filter_num = filter_num
        self.W_f = nn.Sequential(
            nn.Linear(hid_channels, hid_channels),
            self.attn_fn,
        )
        self.W_x = nn.Sequential(
            nn.Linear(hid_channels, hid_channels),
            self.attn_fn,
        )
        self.linear_cls_out = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(hid_channels, out_feats))

        self.attn = list(self.W_x.parameters())
        self.attn.extend(list(self.W_f.parameters()))
        self.lin = list(self.linear_transform_in.parameters())
        self.lin.extend(list(self.linear_cls_out.parameters()))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, anomaly_train=None, normal_train=None):
        """
        :param label:
        :param x:
        :param edge_index:
        :return:
        """

        x = self.linear_transform_in(x)
        h_list = []
        for i, filter_ in enumerate(self.filters):
            h = filter_(x, edge_index)
            h_list.append(h)

        h_filters = torch.stack(h_list, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(x).unsqueeze(-1)

        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        score = soft_score

        res = h_filters[:, 0, :] * score[:, 0]
        for i in range(1, self.filter_num):
            res += h_filters[:, i, :] * score[:, i]

        y_hat = self.linear_cls_out(res)
        marginal_loss = 0.

        if self.training:

            normal_bias = score[normal_train][:, 1] - score[normal_train][:, 0]
            anomaly_bias = score[anomaly_train][:, 0] - score[anomaly_train][:,
                                                                             1]
            normal_bias = torch.clamp(normal_bias, -0.)
            anomaly_bias = torch.clamp(anomaly_bias, -0.)
            normal_bias = torch.mean(normal_bias)
            anomaly_bias = torch.mean(anomaly_bias)
            bias = anomaly_bias + normal_bias
            marginal_loss = bias

        if self.training:
            return y_hat, marginal_loss
        else:
            return y_hat


def amnet(in_feats, out_feats, **kwargs):
    return AMNet(in_feats, out_feats, **kwargs)


def amnet_drop05(in_feats, out_feats, **kwargs):
    return AMNet(in_feats, out_feats, dropout=0.05, **kwargs)


def amnet_drop10(in_feats, out_feats, **kwargs):
    return AMNet(in_feats, out_feats, dropout=0.1, **kwargs)


def amnet_drop30(in_feats, out_feats, **kwargs):
    return AMNet(in_feats, out_feats, dropout=0.3, **kwargs)
