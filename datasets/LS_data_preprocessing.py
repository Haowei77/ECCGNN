import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import xlrd
import math
import random
from itertools import combinations
from joblib import Parallel,delayed

import os
# import pickle as pkl
import sys


def load_data_S2(data='FJ', path="./", bw=True):
    content_filename = "{}{}_contents_raw.txt".format(path,data)
    print("read content: {}".format(data))
    content = pd.read_csv(content_filename) #header 默认1，第一行作为列名

    colname_list = ['ID', 'slope', 'aspect', 'height', 'Driver', 'Dfault', 'erosion',
                    'soiltype', 'stratum', 'NDVI', 'DLGQ', 'YFFQ', 'SG', 'SJ', 'label', 'length', 'area']
    if(data=='FL'):
        col = ['Tri_Index','Slope_Pct','Aspect','dem30','水系distance','断层distance',
               'erosion_1km','soil','DC_type','NDVI','EL','易发程度','SG','SJ',
               'label', 'Shape_Length', 'Shape_Area'] # 1.滑坡,2.斜坡,3.崩塌,4.非滑坡
        content = content.loc[:,col]
        content.columns = colname_list
    if(data=='FJ'):
        col = ['Tri_Index','Slope_Pct','Aspect','DEM30','水系distance','断层distance',
               'erosion1km','soiltype','地层DC','NDVI','地理国情EL','易发DJ','SG','SJ',
               'label', 'Shape_Length', 'Shape_Area']
        content = content.loc[:, col]
        content.columns = colname_list
        
    # print("updata data: {}".format(data))
    # # 合并新标签
    # new_filename = "{}{}_newlabel.txt".format(path, data)
    # col_label = ['Tri_Index', 'NonLS']
    # newcontent = pd.read_csv(new_filename, usecols=col_label)
    # newcontent = newcontent.sort_values(by=['NonLS'], ascending=False) #按标签降序排序
    # newcontent = newcontent.drop_duplicates(['Tri_Index'], keep='first')   #删除重复ID,保留第一次出现的，即有标签的
    # content = pd.merge(content, newcontent, how='left', left_on='ID', right_on='Tri_Index')
    # content['label'] = content['NonLS']

    print("modify data: {}".format(data))
    content = content.dropna(subset=['stratum','DLGQ','YFFQ'])
    content.loc[((content['label'] == 1) & (content['slope']<11)), 'slope'] = 10
    slope = content['slope'].values
    for i, v in enumerate(slope):
        slope[i] = math.atan(v/100)/math.pi*180 #百分制转为角度
    content['slope'] = slope
    content['soiltype'] = content['soiltype']//1000

    print("creat graph: {}".format(data))
    idl = np.where(content['label']>0)[0]    # 样本数据
    idn = np.where(content['label'] == 0)[0].flatten()  # 样本数据
    np.random.shuffle(idn)
    # idchoose = np.r_[idl, idn[:5000-idl.shape[0]]]
    col_choose = ['ID', 'slope', 'aspect', 'height', 'Driver', 'Dfault', 'erosion',
                  'soiltype', 'stratum', 'NDVI', 'DLGQ', 'YFFQ', 'length', 'area',
                  'label']
    if idl.shape[0] == 0:
        print('no labled data')
        return
    features = content[col_choose]
    if data == 'QJ':
        features['DLGQ'] = features['DLGQ']//10
    features.index = range(len(features))
    features.to_csv(u"{}{}_normal.csv".format(path, data))
    features, cw_dict = data_excle(features, path = "./feature_parallel_table.xls")#映射对照,权重映射
    features = features.astype(np.int32)
    # onehot
    # f_onehot = encode_onehot(features, cw_dict)
    f_onehot, edge_weight = edge_creat(features, cw_dict)

    features['ENV'] = f_onehot['ENV']
    features.to_csv(u"{}{}_class.csv".format(path, data))
    print("save graph: {}".format(data))
    outF = pd.concat([features['ID'], f_onehot, features['label']], axis=1)
    # outputFeature = np.c_[content.iloc[:]['ID'], f_onehot, content.iloc[:]['label']]#.iloc[idl]
    outF.to_csv(u"{}{}_contents.csv".format(path, data))
    outE = pd.DataFrame(data=edge_weight)
    outE.to_csv(u"{}{}_edges.csv".format(path, data), index=False, header=False)
    # np.savetxt(u"{}{}_contents.txt".format(path,data), outputFeature, fmt='%s', delimiter='\t')
    # np.savetxt(u"{}{}_edges.txt".format(path,data), edge_weight, fmt='%i %i %s', delimiter='\t')
    print('done:Features_dim {},node {},edge {}'.format(f_onehot.shape[1]+2, content['ID'].shape[0], edge_weight.shape[0]))
    return

def contentsload_Raster(data='FJR', path="../../landslide/"):
    content_filename = "{}{}_contents_raw.csv".format(path,data)
    print("read content: {}".format(data))
    content = pd.read_csv(content_filename) #header 默认0，第一行作为列名
    # 1.滑坡,2.斜坡,3.崩塌,4.非滑坡

    if('FJ' in data):
        col = ['pointid', 'DLGQ', 'YFFQ', 'DCnum', 'Driver', 'Dfault', 'NDVI', 'Soiltype', 'Erosion',
               'Slope', 'Altitude', 'Aspect', 'Label']
        colname_list = ['ID', 'DLGQ', 'YFFQ', 'stratum', 'Driver', 'Dfault', 'NDVI', 'soiltype', 'erosion',
                        'slope', 'height', 'aspect', 'label']
        content = content.loc[:, col]
        content.columns = colname_list

    if('FLR' in data):
        col = ['pointid', 'LULC', 'YFFQ', 'Stratum', 'Driver', 'Dfault', 'NDVI', 'Soiltype', 'Erosion',
               'Slope', 'Altitude', 'Aspect', 'label']
        colname_list = ['ID', 'DLGQ', 'YFFQ', 'stratum', 'Driver', 'Dfault', 'NDVI', 'soiltype', 'erosion',
                        'slope', 'height', 'aspect', 'label']
        content = content.loc[:, col]
        content.columns = colname_list

    print("modify data: {}".format(data))
    # content = content.dropna(subset=['stratum','DLGQ','YFFQ','slope', 'height', 'aspect'])
    content = content.dropna()
    # content.loc[((content['label'] == 1) & (content['slope']<11)), 'slope'] = 10
    print("data shape: {}".format(content.shape))
    content['soiltype'] = content['soiltype']//1000
    content['NDVI'] = content['NDVI'] * 100
    content['DLGQ'] = content['DLGQ'] * 10

    print("creat graph: {}".format(data))
    idl = np.where(content['label']>0)[0]    # 样本数据
    if idl.shape[0] == 0:
        print('no labled data')
        return

    col_choose = ['ID', 'slope', 'aspect', 'height', 'Driver', 'Dfault', 'erosion',
                  'soiltype', 'stratum', 'NDVI', 'DLGQ', 'YFFQ', 'label']
    features = content[col_choose]
    features.index = range(len(features))
    features.to_csv(u"{}{}_normal.csv".format(path, data))
    features, cw_dict = data_excle(features, path = "../../landslide/feature_parallel_table.xls")#映射对照,权重映射
    features = features.astype(np.int32)
    # onehot
    # f_onehot = encode_onehot(features, cw_dict)
    f_onehot, edge_weight = edge_creat(features, cw_dict)

    features['ENV'] = f_onehot['ENV']
    features.to_csv(u"{}{}_class.csv".format(path, data))
    print("save graph: {}".format(data))
    outF = pd.concat([features['ID'], f_onehot, features['label']], axis=1)
    outF.to_csv(u"{}{}_contents.csv".format(path, data))
    outE = pd.DataFrame(data=edge_weight)
    outE.to_csv(u"{}{}_edges.csv".format(path, data), index=False, header=False)
    print('done:Features_dim {},node {},edge {}'.format(f_onehot.shape[1]+2, content['ID'].shape[0], edge_weight.shape[0]))
    return

def encode_onehot(features, cw_list):
    node_num = features.shape[0]
    f_onehot = np.zeros([node_num,1],dtype=int)
    f_edge = np.zeros([node_num, 1], dtype=int)
    f_node = np.zeros([node_num, 1], dtype=int)
    e_class = np.zeros([node_num, 1], dtype=int)
    list_edgeclass = ['Driver', 'Dfault', 'stratum', 'YFFQ']
    onehot_name = []
    w_edge = []
    w_node = []
    for k,cw in cw_list.items():
        classes_dict = {c: (np.identity(len(cw.keys()))[c-1, :]) for c, w in cw.items()} #加权可加上 * w
        labels_onehot = np.array(list(map(classes_dict.get, features[k])), dtype=np.float)
        f_onehot = np.c_[f_onehot, labels_onehot]
        onehot_name = onehot_name + list(k + i for i in np.array(list(cw.keys()), dtype=str))
        # TODO 设计连接边
        if k in list_edgeclass:
            for i in sorted(cw):
                w_edge.append(cw[i])
            f_edge = np.c_[f_edge, labels_onehot]
        else:
            for i in sorted(cw):
                w_node.append(cw[i])
            f_node = np.c_[f_node, labels_onehot]
    f_onehot = pd.DataFrame(f_onehot[:,1:], columns=onehot_name)
    f_edge = f_edge[:,1:]*w_edge

    # 环境分类
    edge_pd = pd.DataFrame(f_edge)
    edge_gp = edge_pd.groupby(edge_pd.columns.to_list())
    i=0
    for key, value in edge_gp.indices.items():
        # 依次取边最相近的组的niegh
        eachEnv = value.tolist()
        e_class[eachEnv] = i
        i = i+1
    f_onehot['ENV'] = e_class
    return f_onehot

def edge_creat(features, cw_list):
    node_num = features.shape[0]
    f_onehot = np.zeros([node_num,1],dtype=int)
    f_edge = np.zeros([node_num, 1], dtype=int)
    f_node = np.zeros([node_num, 1], dtype=int)
    e_class = np.zeros([node_num, 1], dtype=int)
    list_edgeclass = ['Driver', 'Dfault', 'stratum', 'YFFQ']
    onehot_name = []
    w_edge = []
    w_node = []
    for k,cw in cw_list.items():
        classes_dict = {c: (np.identity(len(cw.keys()))[c-1, :]) for c, w in cw.items()} #加权可加上 * w
        labels_onehot = np.array(list(map(classes_dict.get, features[k])), dtype=np.float)
        f_onehot = np.c_[f_onehot, labels_onehot]
        onehot_name = onehot_name + list(k + i for i in np.array(list(cw.keys()), dtype=str))
        # TODO 设计连接边
        if k in list_edgeclass:
            for i in sorted(cw):
                w_edge.append(cw[i])
            f_edge = np.c_[f_edge, labels_onehot]
        else:
            for i in sorted(cw):
                w_node.append(cw[i])
            f_node = np.c_[f_node, labels_onehot]
    f_onehot = pd.DataFrame(f_onehot[:,1:], columns=onehot_name)

    f_edge = f_edge[:,1:]*w_edge
    f_node = f_node[:,1:]*w_node

    # 环境分类
    edge_pd = pd.DataFrame(f_edge)
    edge_gp = edge_pd.groupby(edge_pd.columns.to_list())
    i=0
    for key, value in edge_gp.indices.items():
        # 依次取边最相近的组的niegh
        eachEnv = value.tolist()
        e_class[eachEnv] = i
        i = i+1
    f_onehot['ENV'] = e_class

    edges = np.zeros(shape=[1, 4+f_edge.shape[1]])
    for self in range(node_num):
        pt = self*100/node_num
        # if pt % 10 == 0:
        print('\r', '{:.1f}%'.format(pt), end='')
        dist_edge = pd.Series(cdist(f_edge, f_edge[self].reshape((1,f_edge.shape[1])), metric='cosine').flatten())
        # 属性特征相似度
        dist_node = pd.Series(cdist(f_node, f_node[self].reshape((1,f_node.shape[1])), metric='cosine').flatten())
        dist_edge_m = dist_edge.where(dist_edge<0.3).dropna()
        max_edge_num = 10
        edgegroup = dist_edge_m.groupby(dist_edge_m)
        closeENid = []
        for key,value in edgegroup.indices.items():
            # 依次取边最相近的组的niegh
            closeEid = value.tolist()
            # 对上一步的niegh节点特征距离排序，取前几个
            closeENid.extend(dist_node[closeEid].sort_values().head(max_edge_num-len(closeENid) + 1).index.tolist())
            if len(closeENid) < 2:
                continue
            else:
                break
        closeENid = closeENid[1:]
        if len(closeENid) < max_edge_num:
            max_edge_num = len(closeENid)
        weight = f_edge[closeENid]
        selflist = np.ones(max_edge_num, dtype=int) * self
        edge = np.c_[closeENid, selflist[:, None], 1-dist_node[closeENid], 1-dist_edge[closeENid], weight]
        edges = np.r_[edges, edge]
    return f_onehot, np.array(edges[1:,:])

def data_excle(features, path = "../../landslide/属性编码对照表.xlsx"):
    data = xlrd.open_workbook(path)
    sheets = data.sheet_names()
    cw_dict = {}
    for column in features:
        if column in data.sheet_names():
            table = data.sheet_by_name(column)
            k = np.array(table.col_values(0), dtype=np.int)
            v = np.array(table.col_values(1), dtype=np.int)
            v2 = np.array(table.col_values(3), dtype=np.float)
            class_w = dict(zip(v, v2))
            # 映射
            index = np.digitize(features[column], k, right=True)
            features[column] = v[index]
            cw_dict[column] = class_w
    return features, cw_dict



if __name__ == '__main__':
    load_data_S2(data='FJ')
    # load_data_S2(data='FL')
    # contentsload_Raster(data='FLR')
