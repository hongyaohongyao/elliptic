# Elliptic图异常检测

Elliptic数据集一个比特币交易网络，其中节点是交易，边是比特币货币的流动方向。交易网络中共包含46564个节点，73248条边，每个节点包含93中特征。

本项目使用不同图神经网络检测网络中非法交易。

## 安装环境

1. 安装torch([更多版本](https://pytorch.org/get-started/previous-versions/))

   ```shell
   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

2. 安装PyG([更多版本](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html))

   ```shell
   pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
   ```

3. 安装剩余依赖

   ```shell
   pip install -r requirements.txt
   ```

4. 数据集解压到data目录下

请根据实际环境选择pytorch以及PyG版本

## 训练模型

训练全部模型

```
bash train_all.sh
```

训练一个单独的模型

```shell
python train.py [-h] [--gpu GPU] [-a ARCH] [-s SUFFIX] [-e EPOCHS] [--lr LR] [--nc NC] [--wd WEIGHT_DECAY] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             GPU id to use.
  -a ARCH, --arch ARCH  model architecture
  -s SUFFIX, --suffix SUFFIX
                        the suffix of save directory
  -e EPOCHS, --epochs EPOCHS
                        number of total epochs to run
  --lr LR, --learning-rate LR
                        initial learning rate
  --nc NC, --num-classes NC
                        num class
  --wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        weight decay (mini_net_sgd: 5e-4)
  --seed SEED           Seed for initializing training.
```

范例：训练改进后的AMNet，使用0号gpu，训练后的结果保存在run/amnet_ip文件夹下。

```shell
python train.py -a=amnet_ip --gpu=0
```



