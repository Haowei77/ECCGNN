import numpy as np
import pandas as pd
import torch
import xlrd
import dgl
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib; matplotlib.use('Agg') # matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os

class EarlyStopping:
    def __init__(self, patience=10, ti=0, name='', dir=''):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.ti = ti
        self.scores=[]
        self.name = name
        self.dir = dir
        self.remane = 0
        self.path = ''

    def step(self, acc, model):
        score = acc
        if len(self.scores) < 20:
            self.scores.append(score)
            self.best_score = np.mean(self.scores)
            self.save_checkpoint(model)
        else:
            self.scores.pop(0)
            self.scores.append(score)
            if np.mean(self.scores) <= self.best_score:
                self.counter += 1
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = np.mean(self.scores)
                self.save_checkpoint(model)
                self.counter = 0
        return self.early_stop
    # def step(self, acc, model):
    #     score = acc
    #     if self.best_score is None:
    #         self.best_score = score
    #         self.save_checkpoint(model)
    #     elif score < self.best_score:
    #         self.counter += 1
    #         # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
    #         if self.counter >= self.patience:
    #             self.early_stop = True
    #     else:
    #         self.best_score = score
    #         self.save_checkpoint(model)
    #         self.counter = 0
    #     return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        self.path = '{}/{}_{}{}_es_checkpoint{:02d}.pt'.format(self.dir, self.name, model.bw, self.ti, self.remane)
        # if os.path.exists(path):
        #     os.remove(path) # 为什么会有占用？
        torch.save(model.state_dict(), self.path)
        self.remane = self.remane+1

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))
        while self.remane-2 >= 0:
            path = '{}/{}_{}{}_es_checkpoint{:02d}.pt'.format(self.dir, self.name, model.bw, self.ti, self.remane-2)
            self.remane = self.remane -1
            try:
                os.remove(path)
            except:
                print('error: ', path)
                pass

def ROCplot(path, roc_auc, fpr, tpr):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(path)

def plotlog(trlog,acc=0,savepath=''):
    # trlog: train_loss, train_acc, val_loss, val_acc
    epoch = trlog.shape[0]
    train_loss = trlog[:, 0]
    train_acc = trlog[:, 1]
    test_loss = trlog[:, 2]
    test_acc = trlog[:, 3]
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # y1_major_locator = plt.MultipleLocator(0.1)  # 设置刻度间隔
    # ax1.yaxis.set_major_locator(y1_major_locator)
    # y2_major_locator = plt.MultipleLocator(0.05)  # 设置刻度间隔
    # ax2.yaxis.set_major_locator(y2_major_locator)
    # ax1.axis([-epoch//40, epoch, 0, 2])
    # ax2.axis([-epoch//40, epoch, 0.5, 1])
    ax1.plot(range(epoch), train_loss, 'coral', label='train_loss')
    ax1.plot(range(epoch), test_loss, 'red', label='test_loss')
    ax2.plot(range(epoch), train_acc, 'yellowgreen', label='train_acc')
    ax2.plot(range(epoch), test_acc, 'green', label='test_acc')
    ax1.set_xlabel('epoch\tacc = %.4f'%acc)
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy')
    ax1.legend()
    ax2.legend()
    ax1.set_title("accuracy = %.4f"%acc)
    # plt.show()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

def preprocessing(path, dataset, tr=0.3, num=1, device=torch.device('cpu')):
    testr = abs(tr)
    g = dgl.DGLGraph()
    content_filename = "{}{}_contents.csv".format(path,dataset)
    cites_filename = "{}{}_edges.csv".format(path,dataset)
    print("read features: {}".format(dataset))
    content = pd.read_csv(content_filename, index_col=0, dtype=int)

    feature_col=content.columns[1:-2].to_list()
    # 尝试选节点特征
    # env = ['Driver', 'Dfault', 'stratum', 'soiltype', 'YFFQ']
    # for e in env:
    #     for col in content.columns:
    #         if e in col: feature_col.remove(col)

    xs = content[feature_col]  # -1
    ids = content['ID']

    newNegSampLoc = content[content['label'] == 5].index
    _, addsamp = train_test_split(newNegSampLoc, test_size=246/416, random_state=3)
    content.loc[addsamp, 'label'] = 4
    ys = np.ones(content.shape[0]) * 4
    if dataset in ['FJ','FL','FLR','FJR']:
        ys[content['label'] == 1] = 1
        ys[content['label'] == 4] = 0
    else:
        ys = content['label']
    g.add_nodes(len(xs))

    print("set dataset: {}".format(dataset))
    xs = xs.to_numpy()
    labels = ys

    seed = num*10
    hp = content[content['label'] == 1].index  #滑坡
    fhp = content[content['label'] == 4].index #非隐患
    hp_num = hp.shape[0]
    fhp_num = fhp.shape[0]
    if hp_num < fhp_num:  # 以正样本为准，取尽可能相等的负样本
        _, fhp = train_test_split(fhp, test_size=hp_num / fhp_num, random_state=seed)
        fhp_num = fhp.shape[0]
    print('positive:{} negative:{}'.format(hp_num, fhp_num))

    idx_train_list = []
    idx_test_list = []
    for j in range(num):
        seed = j*10
        id_hp_train, id_hp_test = train_test_split(hp, test_size=testr, random_state=seed)  #i=8
        id_f_train, id_f_test = train_test_split(fhp, test_size=testr, random_state=seed)  #
        idx_train = np.r_[id_f_train, id_hp_train]  # ,id_xp_train,id_bt_train]
        idx_test = np.r_[id_f_test, id_hp_test]  # ,id_xp_test,id_bt_test]
        np.random.seed(seed)
        np.random.shuffle(idx_train)
        np.random.seed(seed)
        np.random.shuffle(idx_test)

        idx_train_list.append(idx_train)
        idx_test_list.append(idx_test)
    print("read edges: {}".format(dataset))
    edges_weights = pd.read_csv(cites_filename, header=None)
    print("creat graph: {}".format(dataset))
    g.add_edges(edges_weights.iloc[:,0].astype(int), edges_weights.iloc[:,1].astype(int))
    if tr > 0:
        return g, ids, xs, labels, edges_weights, idx_train_list, idx_test_list, idx_test_list
    else:
        return g, ids, xs, labels, edges_weights, idx_test_list, idx_train_list, idx_train_list

def preprocessing_env(path, dataset, tr=0.3, num=1, device=torch.device('cpu')):
    # TODO: 其他区域数据链接到基础图上，基础图如何删减与丰富？
    testr = abs(tr)
    g = dgl.DGLGraph()
    content_filename = "{}{}_contents.csv".format(path,dataset)
    cites_filename = "{}{}_edges.csv".format(path,dataset)
    print("read features: {}".format(dataset))
    content = pd.read_csv(content_filename, index_col=0, dtype=int)

    feature_col = content.columns[1:-2].to_list() #用于计算的特征列
    # 尝试选节点特征
    # env = ['Driver', 'Dfault', 'stratum', 'soiltype', 'YFFQ']
    # for e in env:
    #     for col in content.columns:
    #         if e in col: feature_col.remove(col)

    xs = content[feature_col]  # -1
    ids = content['ID']
    env = content['ENV']
    ys = np.ones(content.shape[0]) * 4
    if dataset in ['FJ','FL','FLR','FJR']:
        ys[content['label'] == 1] = 1
        ys[content['label'] == 4] = 0
        ys[content['label'] == 5] = 0
    else:
        ys = content['label']
    g.add_nodes(len(xs))

    print("set dataset: {}".format(dataset))
    xs = xs.to_numpy()
    labels = ys

    hp = content[content['label'] == 1]  #滑坡
    fhp = content[content['label'] >= 4] #非隐患
    idx_train_list = []
    idx_test_list = []
    seed = 6*10
    for j in range(num):
        # seed = j*10
        id_f_train = []
        id_f_test = []
        id_hp_train = []
        id_hp_test = []
        for i in range(env.max(0)):
            mid_hp = hp[hp['ENV'] == i].index
            mid_fhp = fhp[fhp['ENV'] == i].index
            hp_num = mid_hp.shape[0]
            fhp_num = mid_fhp.shape[0]
            if hp_num == 0:
                continue
            elif hp_num == 1:
                hp_train = mid_hp
            else:
                hp_train, hp_test = train_test_split(mid_hp, test_size=testr, random_state=seed)  #i=8
                id_hp_test = id_hp_test + hp_test.to_list()
            if hp_num < fhp_num:  # 以正样本为准，取尽可能相等的负样本
                _, mid_fhp = train_test_split(mid_fhp, test_size=mid_hp.shape[0]/mid_fhp.shape[0], random_state=seed)
                fhp_num = mid_fhp.shape[0]
            if fhp_num < 2:
                f_train = mid_fhp
            else:
                f_train, f_test = train_test_split(mid_fhp, test_size=testr, random_state=seed)  #
                id_f_test = id_f_test + f_test.to_list()
            id_hp_train = id_hp_train + hp_train.to_list()
            id_f_train = id_f_train + f_train.to_list()
        idx_train = np.r_[id_f_train, id_hp_train]  # ,id_xp_train,id_bt_train]
        np.random.seed(seed)
        np.random.shuffle(idx_train)
        idx_test = np.r_[id_f_test, id_hp_test]  # ,id_xp_test,id_bt_test]
        np.random.seed(seed)
        np.random.shuffle(idx_test)
        idx_train_list.append(idx_train)
        idx_test_list.append(idx_test)
    print('positive:{} negative:{}'.format(
        len(id_hp_train)+len(id_hp_test), len(id_f_train)+len(id_f_test)))
    print("read edges: {}".format(dataset))
    edges_weights = pd.read_csv(cites_filename, header=None)
    # edges_weights = edges_weights[edges_weights.iloc[:, 2:].sum(axis=1) <= 1]
    print("creat graph: {}".format(dataset))
    g.add_edges(edges_weights.iloc[:,0].astype(int), edges_weights.iloc[:,1].astype(int))

    if tr > 0:
        return g, ids, xs, labels, edges_weights, idx_train_list, idx_test_list, idx_test_list
    else:
        return g, ids, xs, labels, edges_weights, idx_test_list, idx_train_list, idx_train_list

def preprocessing_topu(path, dataset, tr=0.3, num=1, device=torch.device('cpu')):
    testr = abs(tr)
    g = dgl.DGLGraph()
    content_filename = "{}{}_contents.csv".format(path,dataset)
    cites_filename = "{}{}_edges_raw.txt".format(path,dataset)
    print("read features: {}".format(dataset))
    content = pd.read_csv(content_filename, index_col=0, dtype=int)

    feature_col=content.columns[1:-2].to_list()
    # 尝试选节点特征
    # env = ['Driver', 'Dfault', 'stratum', 'soiltype', 'YFFQ']
    # for e in env:
    #     for col in content.columns:
    #         if e in col: feature_col.remove(col)

    xs = content[feature_col]  # -1
    ids = content['ID']

    newNegSampLoc = content[content['label'] == 5].index
    _, addsamp = train_test_split(newNegSampLoc, test_size=246/416, random_state=3)
    content.loc[addsamp, 'label'] = 4
    ys = np.ones(content.shape[0]) * 4
    if dataset in ['FJ','FL','FLR','FJR']:
        ys[content['label'] == 1] = 1
        ys[content['label'] == 4] = 0
    else:
        ys = content['label']
    g.add_nodes(len(xs))

    print("set dataset: {}".format(dataset))
    xs = xs.to_numpy()
    labels = ys

    seed = num*10
    hp = content[content['label'] == 1].index  #滑坡
    fhp = content[content['label'] == 4].index #非隐患
    hp_num = hp.shape[0]
    fhp_num = fhp.shape[0]
    if hp_num < fhp_num:  # 以正样本为准，取尽可能相等的负样本
        _, fhp = train_test_split(fhp, test_size=hp_num / fhp_num, random_state=seed)
        fhp_num = fhp.shape[0]
    print('positive:{} negative:{}'.format(hp_num, fhp_num))

    idx_train_list = []
    idx_test_list = []
    for j in range(num):
        seed = j*10
        id_hp_train, id_hp_test = train_test_split(hp, test_size=testr, random_state=seed)  #i=8
        id_f_train, id_f_test = train_test_split(fhp, test_size=testr, random_state=seed)  #
        idx_train = np.r_[id_f_train, id_hp_train]  # ,id_xp_train,id_bt_train]
        idx_test = np.r_[id_f_test, id_hp_test]  # ,id_xp_test,id_bt_test]
        np.random.seed(seed)
        np.random.shuffle(idx_train)
        np.random.seed(seed)
        np.random.shuffle(idx_test)

        idx_train_list.append(idx_train)
        idx_test_list.append(idx_test)
    print("read edges: {}".format(dataset))
    edges_weights = pd.read_csv(cites_filename, header=None)
    ids2 = pd.Series(ids.index, index=ids.values)
    id_dict = ids2.to_dict()
    edges_weights[0] = [id_dict.get(k) for k in edges_weights.iloc[:, 0]]
    edges_weights[1] = [id_dict.get(k) for k in edges_weights.iloc[:, 1]]
    edges_weights.dropna(axis=0,inplace=True)
    edges_weights = edges_weights.reset_index(drop=True)
    print("creat graph: {}".format(dataset))
    g.add_edges(edges_weights.iloc[:,0].astype(int), edges_weights.iloc[:,1].astype(int))
    if tr > 0:
        return g, ids, xs, labels, edges_weights, idx_train_list, idx_test_list, idx_test_list
    else:
        return g, ids, xs, labels, edges_weights, idx_test_list, idx_train_list, idx_train_list

def split_samples(lables, tr=0.3, num=1):
    testr = abs(tr)
    id_f = np.where(lables == 0)[0]     #非隐患
    id_hp = np.where(lables == 1)[0]    #滑坡
    id_xp = np.where(lables == 2)[0]    #斜坡
    id_bt = np.where(lables == 3)[0]    #崩塌
    id_n = np.where(lables == 4)[0]     #无标签
    idx_train_list = []
    idx_test_list = []
    for i in range(num):
        id_f_train,id_f_test = train_test_split(id_f, test_size=testr, random_state=i*10)
        id_hp_train, id_hp_test = train_test_split(id_hp, test_size=testr, random_state=i*10)
        # id_xp_train, id_xp_test = train_test_split(id_xp, test_size=testr)
        # id_bt_train, id_bt_test = train_test_split(id_bt, test_size=testr)

        idx_train = np.r_[id_f_train,id_hp_train]  # ,id_xp_train,id_bt_train]
        np.random.shuffle(idx_train)
        idx_test = np.r_[id_f_test,id_hp_test]  # ,id_xp_test,id_bt_test]
        np.random.shuffle(idx_test)

        idx_train_list.append(idx_train)
        idx_test_list.append(idx_test)
    if tr > 0:
        return idx_train_list, idx_test_list, idx_test_list
    else:
        return idx_test_list, idx_train_list, idx_train_list


def read_excle(b_w = 0, lenth = 12, path = "../../landslide/属性编码对照表.xlsx"):
    data = xlrd.open_workbook(path)
    sheets = data.sheet_names()
    v = []
    for i in range(lenth):
        table = data.sheet_by_name(sheets[i])
        if b_w==0:
            v2 = table.col_values(3)
        else:
            v2 = table.col_values(2)
        v = v+v2
    return v

if __name__ == '__main__':
    read_excle()