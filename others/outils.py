import numpy as np
import pandas as pd
import torch
import dgl
from sklearn.model_selection import train_test_split
import time
import os
import geopandas

def preprocessing_plus(path, dataset, tr=0.3, num=1, device=torch.device('cpu')):
    testr = abs(tr)
    content_filename = "{}{}_contents.csv".format(path,dataset)
    print("read features: {}".format(dataset))
    content = pd.read_csv(content_filename, index_col=0, dtype=int)
    xs = content.iloc[:,1:-2] # -1
    ids = content.iloc[:,0]
    env = content['ENV']
    ys = np.ones(content.shape[0]) * 4
    if dataset in ['FJ','FL']:
        ys[content['lable'] == 1] = 1
        ys[content['lable'] == 4] = 0
        ys[content['lable'] == 5] = 0
    else:
        ys = content['lable']

    print("set dataset: {}".format(dataset))
    xs = xs.to_numpy()
    labels = ys

    idx_train_list = []
    idx_test_list = []
    for j in range(num):
        id_f_train = []
        id_f_test = []
        id_hp_train = []
        id_hp_test = []
        hp = content[content['lable'] == 1]  #滑坡
        fhp = content[content['lable'] >= 4] #非隐患
        for i in range(env.max(0)):
            mid_hp = hp[hp['ENV'] == i].index
            mid_fhp = fhp[fhp['ENV'] == i].index
            hp_num = mid_hp.shape[0]
            fhp_num = mid_fhp.shape[0]
            if hp_num == 0:
                continue
            if hp_num < fhp_num:  # 以正样本为准，取尽可能相等的负样本
                _, mid_fhp = train_test_split(mid_fhp, test_size=mid_hp.shape[0] / mid_fhp.shape[0], random_state=j * 10)
                fhp_num = mid_fhp.shape[0]
            if hp_num < 2:
                hp_train = mid_hp
            else:
                hp_train, hp_test = train_test_split(mid_hp, test_size=testr, random_state=j*10)  #i=8
                id_hp_test = id_hp_test + hp_test.to_list()
            if fhp_num < 2:
                f_train = mid_fhp
            else:
                f_train, f_test = train_test_split(mid_fhp, test_size=testr, random_state=j*10)  #
                id_f_test = id_f_test + f_test.to_list()
            id_hp_train = id_hp_train + hp_train.to_list()
            id_f_train = id_f_train + f_train.to_list()

        idx_train = np.r_[id_f_train, id_hp_train]  # ,id_xp_train,id_bt_train]
        np.random.seed(j*10)
        np.random.shuffle(idx_train)
        idx_test = np.r_[id_f_test, id_hp_test]  # ,id_xp_test,id_bt_test]
        np.random.seed(j*10)
        np.random.shuffle(idx_test)
        idx_train_list.append(idx_train)
        idx_test_list.append(idx_test)

    if tr > 0:
        # print('train:{} test:{}'.format(len(idx_train_list[0]), len(idx_test_list[0])))
        return ids, xs, labels, idx_train_list, idx_test_list, idx_test_list
    else:
        # print('train:{} test:{}'.format(len(idx_test_list[0]), len(idx_train_list[0])))
        return ids, xs, labels, idx_test_list, idx_train_list, idx_train_list


def preprocessing(path, dataset, tr=0.3, num=1, device=torch.device('cpu')):
    testr = abs(tr)
    content_filename = "{}{}_contents.csv".format(path, dataset)
    cites_filename = "{}{}_edges.csv".format(path, dataset)
    print("read features: {}".format(dataset))
    content = pd.read_csv(content_filename, index_col=0, dtype=int)
    xs = content.iloc[:, 1:-2]  # -1
    ids = content.iloc[:, 0]

    newNegSampLoc = content[content['label'] == 5].index
    # ? ? ?  _, addsamp = train_test_split(newNegSampLoc, test_size=246/416, random_state=3)
    if newNegSampLoc.size>0:
        cc = lambda x: (content['label'] == x).sum()
        _, addsamp = train_test_split(newNegSampLoc, test_size=(cc(1)-cc(4))/cc(5), random_state=3)
        content.loc[addsamp, 'label'] = 4
    ys = np.ones(content.shape[0]) * 4
    if dataset in ['FJ','FL']:
        ys[content['label'] == 1] = 1
        ys[content['label'] == 4] = 0
    else:
        ys = content['label']

    print("set dataset: {}".format(dataset))
    xs = xs.to_numpy()
    labels = ys

    hp = content[content['label'] == 1].index  # 滑坡
    fhp = content[content['label'] == 4].index  # 非隐患
    hp_num = hp.shape[0]
    fhp_num = fhp.shape[0]
    if hp_num < fhp_num:  # 以正样本为准，取尽可能相等的负样本
        _, fhp = train_test_split(fhp, test_size=hp_num / fhp_num, random_state=10 * 10)
        fhp_num = fhp.shape[0]
    print('positive:{} negative:{}'.format(hp_num, fhp_num))

    idx_train_list = []
    idx_test_list = []
    for j in range(num):
        id_hp_train, id_hp_test = train_test_split(hp, test_size=testr, random_state=j * 10)  # i=8
        id_f_train, id_f_test = train_test_split(fhp, test_size=testr, random_state=j * 10)  #
        idx_train = np.r_[id_f_train, id_hp_train]  # ,id_xp_train,id_bt_train]
        idx_test = np.r_[id_f_test, id_hp_test]  # ,id_xp_test,id_bt_test]
        np.random.seed(j * 10)
        np.random.shuffle(idx_train)
        np.random.seed(j * 10)
        np.random.shuffle(idx_test)

        idx_train_list.append(idx_train)
        idx_test_list.append(idx_test)
    print('train:{} test:{}'.format(len(idx_train_list[0]), len(idx_test_list[0])))
    if tr > 0:
        # print('train:{} test:{}'.format(len(idx_train_list[0]), len(idx_test_list[0])))
        return ids, xs, labels, idx_train_list, idx_test_list, idx_test_list
    else:
        # print('train:{} test:{}'.format(len(idx_test_list[0]), len(idx_train_list[0])))
        return ids, xs, labels, idx_test_list, idx_train_list, idx_train_list

def split_samples(labels, tr=0.3, num=1):
    testr = abs(tr)
    id_f = np.where(labels == 0)[0]     #非隐患
    id_hp = np.where(labels == 1)[0]    #滑坡
    id_xp = np.where(labels == 2)[0]    #斜坡
    id_bt = np.where(labels == 3)[0]    #崩塌
    id_n = np.where(labels == 4)[0]     #无标签
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

def addRst2TIN(dataset):
    mappath = "D:\\OneDrive - my.swjtu.edu.cn\\项目\\Code\\GNN\\landslide\\0 Final\\ANN_map5.csv"
    data = pd.read_csv(mappath, usecols=[0,2], names=['Tri_Index', 'B'])
    # 0 利用geopandas可视化矢量
    path = r'D:\OneDrive - my.swjtu.edu.cn\项目\Code\滑坡数据\LSM_result'
    file = os.path.join(path, "FJ_tin.shp")
    tin = geopandas.read_file(file)
    print('read shp')
    # ['Tri_Index', 'Shape_Leng', 'Shape_Area', 'NewLabel', 'geometry']

    data_geod = geopandas.GeoDataFrame(data)
    print('merge')
    gdb = tin.merge(data_geod, on='Tri_Index')
    print('save...')
    saevfile = os.path.join(path, "%s_Other_RST.shp"%dataset)
    t0 = time.time()
    gdb.to_file(saevfile)
    print('shp done %.1fs'%(time.time() - t0))

if __name__ == '__main__':
    addRst2TIN('FL')
