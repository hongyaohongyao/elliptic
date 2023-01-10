# baseline
python train.py -a=mlp --gpu=0
python train.py -a=gcn --gpu=0
python train.py -a=sage --gpu=0
python train.py -a=gat --gpu=0
# batch norm
python train.py -a=mlp_bn --gpu=0
python train.py -a=gcn_bn --gpu=0
python train.py -a=sage_bn --gpu=0
python train.py -a=gat_bn --gpu=0
# dropout 0.05
python train.py -a=mlp_bn_drop05 --gpu=0
python train.py -a=gcn_bn_drop05 --gpu=0
python train.py -a=sage_bn_drop05 --gpu=0
python train.py -a=gat_bn_drop05 --gpu=0
# dropout 0.10
python train.py -a=mlp_bn_drop10 --gpu=0
python train.py -a=gcn_bn_drop10 --gpu=0
python train.py -a=sage_bn_drop10 --gpu=0
python train.py -a=gat_bn_drop10 --gpu=0
# dropout 0.30
python train.py -a=mlp_bn_drop30 --gpu=0
python train.py -a=gcn_bn_drop30 --gpu=0
python train.py -a=sage_bn_drop30 --gpu=0
python train.py -a=gat_bn_drop30 --gpu=0
# dropout 0.50
python train.py -a=sage_bn_drop50 --gpu=0
# amnet
python train.py -a=amnet --gpu=0
python train.py -a=amnet_drop05 --gpu=0
python train.py -a=amnet_drop10 --gpu=0
python train.py -a=amnet_drop30 --gpu=0