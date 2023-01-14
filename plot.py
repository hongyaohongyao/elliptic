import matplotlib.pyplot as plt


def plot_acc(ax, labels, train_acc, test_acc):
    ax.plot(labels, train_acc, label='Train')
    ax.plot(labels, test_acc, label='Test')
    ax.set_xlabel('C')
    ax.set_ylabel('Accuracy')


if __name__ == '__main__':
    # 数据从run文件夹下读取
    labels = ['0%', '5%', '10%', '30%']
    mlp_f1 = [0.8795, 0.8775, 0.8820, 0.8791]
    gcn_f1 = [0.8200, 0.8265, 0.8260, 0.8294]
    gat_f1 = [0.8528, 0.8677, 0.8575, 0.8710]
    sage_f1 = [0.8785, 0.8863, 0.8876, 0.9009]
    amnet_f1 = [0.8798, 0.8789, 0.8842, 0.8821]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(labels, mlp_f1, label='MLP+BN')
    ax.plot(labels, gcn_f1, label='GCN+BN')
    ax.plot(labels, gat_f1, label='GAT+BN')
    ax.plot(labels, sage_f1, label='GraphSAGE+BN')
    ax.plot(labels, amnet_f1, label='AMNet')
    ax.legend()
    ax.set_xlabel('p')
    ax.set_ylabel('F1 Scores')
    ax.set_title("F1 Scores with Different p of Dropout")
    fig.savefig('effect_of_dropout.jpg')
