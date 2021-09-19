"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import dgl
from dgl.data import register_data_args, load_data
from sage_conv import SAGEConv
from dgl.nn.pytorch.softmax import edge_softmax
from utils import preprocessing, EarlyStopping, plotlog, preprocessing_env,preprocessing_topu, ROCplot
import sys
sys.path.append("..")
from datasets import LS_map_process


class GraphSAGE(nn.Module):
    def __init__(self,
                 g, bw,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 edge_drop,
                 aggregator_type,
                 device):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.g = g
        self.bw=bw
        self.device = device

        self.edge_feats = 1
        self.weight = None
        if 'v' in self.bw:  # 根据边向量升维
            self.edge_feats = self.g.edata['epw'].shape[1]
        if 'pw' in self.bw:     # 边向量加权参
            self.weight = nn.Parameter(torch.Tensor(self.g.edata['epw'].shape[1], 1))
        if self.weight is not None:
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_uniform_(self.weight, gain=gain) # TODO: 初始化原因
        self.attn_drop = nn.Dropout(edge_drop)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

        # input layer
        self.layers.append(
            SAGEConv(in_feats, n_hidden, self.edge_feats, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, self.edge_feats, aggregator_type, feat_drop=dropout,
                                        activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, self.edge_feats, aggregator_type, feat_drop=dropout,
                                    activation=None))  # activation None

    def forward(self, features):
        if 'w' in self.bw:
            if 'sum' in self.bw:    # 环境边向量求和
                e = self.g.edata['epw'].sum(dim=1)
                self.g.edata['a'] = self.attn_drop(edge_softmax(self.g, e))
            if 'cos' in self.bw:    # 边向量相似度
                self.g.edata['a'] = self.attn_drop(edge_softmax(self.g, self.g.edata['cw']))
            if 'ecs' in self.bw:    # 环境相似度
                self.g.edata['a'] = self.attn_drop(edge_softmax(self.g, self.g.edata['ew']))
            if 'cwe' in self.bw:    # 环境权重和*边相似度
                self.g.edata['a'] = self.attn_drop(edge_softmax(self.g, self.g.edata['cwe']))
            if 'v' in self.bw:      # 根据边向量升维
                self.g.edata['a'] = self.attn_drop(edge_softmax(self.g, self.g.edata['epw'].view(-1,self.edge_feats,1)))
            if 'pw' in self.bw:     # 边向量加权参
                if self.weight is not None:
                    if 'epw' in self.g.edata:
                        self.g.edata['e'] = torch.matmul(self.g.edata['epw'], self.weight)
                        e = self.relu(self.g.edata.pop('e'))
                        # compute softmax
                        self.g.edata['a'] = self.attn_drop(edge_softmax(self.g, e))

        h = features
        vvv = torch.ones(size=[1, self.edge_feats, 1], dtype=torch.float32).to(self.device)
        if 'v' in self.bw:
            h = h.view(-1, 1, h.shape[1]) * vvv
        for i in range(self.n_layers):
            h = self.layers[i](self.g, h)
            if 'v' in self.bw:
                h = h.view(-1, 1, h.shape[1]) * vvv
        logits = self.layers[-1](self.g, h)
        return logits


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        _2, indices2 = torch.max(labels, dim=1)
        correct = torch.sum(indices == indices2)
        return correct.item() * 1.0 / len(labels)

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    _2, indices2 = torch.max(labels, dim=1)
    correct = torch.sum(indices == indices2)
    # correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def APRF(logits, labels):
    _, indices = torch.max(logits, dim=1)
    _2, indices2 = torch.max(labels, dim=1)
    correct = torch.sum(indices == indices2)
    # correct = torch.sum(indices == labels)
    TP = ((indices == 1) * (indices2 == 1)).sum().item()
    FP = ((indices == 1) * (indices2 == 0)).sum().item()
    FN = ((indices == 0) * (indices2 == 1)).sum().item()
    TN = ((indices == 0) * (indices2 == 0)).sum().item()
    Accuracy = (TP+TN)/(TP+FN+FP+TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F_measures = 2 * Precision * Recall / (Precision + Recall)
    return Accuracy,Precision,Recall,F_measures


def main(args, pggg, info=''):
    # load and preprocess dataset
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
    graph, ids, xs, ts, edges_weights, idx_train, idx_test, idx_val = pggg
    if info != '':
        dir, data, bw, agg, args.n_layers, ti = info
    # if args.data2:
    #     graph2, ids2, xs2, ts2, idx_train2, idx_test2, idx_val2 = \
    #         preprocessing(args.path, args.data2, 0.95, args.bw, graph, device)
    #     xs = np.r_[xs, xs2]
    #     ts = np.r_[ts, ts2]
    #     idx_train = np.r_[idx_train, idx_train2]
    #     idx_test = idx_test2
    #     idx_val = idx_val2

    labels = np.zeros(shape=[ts.shape[0], 2])
    labels[ts == 0, 0] = 1
    labels[ts == 1, 1] = 1

    # Pack data
    features = torch.Tensor(xs).to(device)
    in_feats = features.shape[1]
    labels = torch.FloatTensor(labels).to(device)
    n_classes = labels.shape[1]  # np.unique(labels[idx_train]).shape[0]

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_test).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    graph = graph.to(device)
    graph.ndata['features'] = features
    if 'w' in args.bw:
        if 'cos' in args.bw:
            graph.edata['cw'] = torch.tensor(edges_weights.values[:, 2], dtype=torch.float32).to(device)
        elif 'ecs' in args.bw:
            graph.edata['ew'] = torch.tensor(edges_weights.values[:, 3], dtype=torch.float32).to(device)
        elif 'cwe' in args.bw:
            esum = torch.tensor(edges_weights.values[:,4:], dtype=torch.float32).sum(dim=1)
            cwe = torch.tensor(edges_weights.values[:, 2], dtype=torch.float32)*esum
            graph.edata['cwe'] = torch.tensor(cwe, dtype=torch.float32).to(device)
        else:
            graph.edata['epw'] = torch.tensor(edges_weights.values[:,4:], dtype=torch.float32).to(device)

    # create GraphSAGE model
    model = GraphSAGE(graph, args.bw,
                      in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.edge_drop,
                      args.aggregator_type,
                      device
                      )
    # if cuda:
    model.to(device)

    if args.early_stop:
        stopper = EarlyStopping(patience=300, ti=ti, name='{}_{}'.format(args.data,args.bw), dir=pt_dir)
    # loss_fcn = torch.nn.CrossEntropyLoss()
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    loss_fcn = loss_fcn.to(device)

    # use optimizer
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    earlystop = False
    # initialize graph
    dur = []
    trlog = np.zeros([args.n_epochs,4])
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        train_loss = loss_fcn(logits[idx_train], labels[idx_train])
        train_acc = accuracy(logits[idx_train], labels[idx_train])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        if (epoch+150)%200 == 0:
            lr = lr/2
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        model.eval()
        with torch.no_grad():
            logits = model(features)
            eval_loss = loss_fcn(logits[idx_val], labels[idx_val])
            eval_acc = accuracy(logits[idx_val], labels[idx_val])

        trlog[epoch] = [train_loss.item(), train_acc, eval_loss, eval_acc]
        if args.early_stop:
            if stopper.step(eval_acc, model):
                earlystop = True
                print('\n%d %s %s: %d: %.4f'%(ti, bw, agg, epoch, stopper.best_score))
                trlog = trlog[:epoch]
                break
        # 训练过程输出
        # if args.print_detail == 1:
        if epoch % 50 == 0:
            print("\n",end='')
        print('\r',"Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValLoss{:.4f} | ValAcc {:.4f} ".format(
            epoch, np.mean(dur), train_loss.item(), train_acc, eval_loss, eval_acc),end='')#, file=outputfile)
    stopper.load_checkpoint(model)
    model.eval()
    with torch.no_grad():
        logits = model(features)
        acc,prec,rec,f_1 = APRF(logits[idx_test], labels[idx_test])
        # tra_acc = accuracy(logits[idx_train], labels[idx_train])
    # acc = evaluate(model, features, labels, idx_test)
    print("{} TrainAcc:{:.4f} Test-Accuracy {:.4f}; Precision {:.4f}; Recall {:.4f}; F_measure {:.4f}".format(args.data,train_acc,acc,prec,rec,f_1), file=outputfile)
    pred_rst = F.softmax(logits,dim=1) # predict_proba
    # Test AUROC
    labels = labels.cpu()
    pred_rst = pred_rst.cpu()
    fpr, tpr, thre = metrics.roc_curve(labels[idx_test, 1], pred_rst[idx_test, 1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    rocpath = '{}//{}_{}_{}l{}_roc{}.jpg'.format(dir, data, bw, agg, args.n_layers, ti)
    ROCplot(rocpath, roc_auc, fpr, tpr)
    # map_df = pd.DataFrame(pred_rst.tolist(), index=ids)
    if info != '':
        # map_df.to_csv('{}//{}_{}_{}l{}_map{}.csv'.format(dir, data, bw, agg, args.n_layers, ti))
        plotlog(trlog, acc, '{}//{}_{}_{}l{}_log{}.jpg'.format(dir, data, bw, agg, args.n_layers, ti))
    return train_acc,acc,prec,rec,f_1,roc_auc,pred_rst[:,1].tolist() #, map_df, trlog


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument('--path',
                        type=str, default='../../landslide/')
    parser.add_argument('--data',
                        type=str, default='FJ')
    parser.add_argument('--data2',
                        type=str, default='')
    parser.add_argument("--dropout",
                        type=float, default=0.2, help="dropout probability")
    parser.add_argument("--edge-drop",
                        type=float, default=0.5, help="edge dropout probability")
    parser.add_argument("--gpu",
                        type=int, default=0, help="-1 cpu, 0 gpu")
    parser.add_argument("--lr",
                        type=float, default=0.05, help="learning rate")
    parser.add_argument("--n-epochs",
                        type=int, default=3000, help="number of training epochs")
    parser.add_argument("--n-hidden",
                        type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--n-layers",
                        type=int, default=2, help="number of hidden gcn layers")
    parser.add_argument("--weight-decay",
                        type=float, default=5e-3, help="Weight for L2 loss")
    parser.add_argument("--aggregator-type",
                        type=str, default="mean", help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--test-ratio",
                        type=float, default=0.3, help="test data ratio")
    parser.add_argument("--bw",
                        type=str, default='w_sum', help="no: no weight; w_v; w_sum; w_cos; w_mul_pw")
    parser.add_argument("--env",
                        type=str, default='e', help="e landslide formative enviroment; n non;t topo")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--print-detail",
                        type=int, default=0, help="1 print detail; 0")
    args = parser.parse_args()

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 100)

    accmean_df = {}
    for data in [args.data]: # 'FJ', 'FL', 'FJR'
        # dict_APRFmean = {}
        dict_TrAmean = {}
        dict_Amean = {}
        dict_Pmean = {}
        dict_Rmean = {}
        dict_Fmean = {}
        dict_AUCmean = {}
        acc_df = pd.DataFrame()
        args.data = data

        # 创建文件夹，存放log
        dateT = time.strftime("%m%d%H")
        dir = '{}{}{}_sage_{}-fd{}-ed{}-nh{}-ly{}_TR{}'.format(
            args.path, args.data, args.env,dateT, args.dropout, args.edge_drop, args.n_hidden, args.n_layers, args.test_ratio)
        while os.path.exists(dir):
            dir = dir + '(1)'
        os.mkdir(dir)
        pt_dir = '{}/model_pt'.format(dir)
        os.mkdir(pt_dir)

        # import sys
        filename = 'log-fd{}-ed{}-nh{}-ly{}'.format(args.dropout, args.edge_drop, args.n_hidden, args.n_layers)
        full_path = '{}//{}.txt'.format(dir, filename)  # 也可以创建一个.doc的word文档
        outputfile = open(full_path, 'w')  # None #
        # sys.stdout = outputfile
        print(args, file=outputfile)

        num = 10
        if args.env=='e':
            graph, ids, xs, ts, edges_weights, idx_train_list, idx_test_list, idx_val_list = \
                preprocessing_env(args.path, args.data, args.test_ratio, num, device=device)
            print("preprocessing_env")
        elif args.env == 'n':
            graph, ids, xs, ts, edges_weights, idx_train_list, idx_test_list, idx_val_list = \
                preprocessing(args.path, args.data, args.test_ratio, num, device=device)
        elif args.env == 't':
            graph, ids, xs, ts, edges_weights, idx_train_list, idx_test_list, idx_val_list = \
                preprocessing_topu(args.path, args.data, args.test_ratio, num, device=device)
            print("preprocessing")
        # TODO: 其他区域数据链接到基础图上，基础图如何删减与丰富？
        # idx_train_list, idx_test_list, idx_val_list = split_samples(ts, args.test_ratio, len(idx_test_list[0]), num)
        print("""----Data statistics------'#Edges %d;  #Features %d;  #test_ratio %d""" %
              (graph.number_of_edges(), xs.shape[0], args.test_ratio), file=outputfile)
        map_df = pd.DataFrame(index=ids)
        # C 特征；CS 特征相似度； E 环境；sum 环境权重；ES 环境相似度； CWE 特征*环境权重；
        for bw in ['cwe']: # ['no', 'w_ecs', 'w_cos', 'w_sum', 'cwe', 'w_mul_pw', 'w_v' ]:
            dict_TrAmean[bw] = {}
            dict_Amean[bw] = {}
            dict_Pmean[bw] = {}
            dict_Rmean[bw] = {}
            dict_Fmean[bw] = {}
            dict_AUCmean[bw] = {}
            args.bw = bw
            for agg in ['mean']:#'mean', 'gcn', 'pool', 'poolmean'
                dur=[]
                args.aggregator_type = agg
                tra_acclist = []
                acclist = []
                preclist = []
                reclist = []
                f1list = []
                rauclist = []
                for i in range(num):
                    print("{} train {} {} {}".format(data, i, agg, bw), file=outputfile)
                    starttime = time.time()
                    pggg = graph, ids, xs, ts, edges_weights, idx_train_list[i], idx_test_list[i], idx_val_list[i]
                    info = dir, data, bw, agg, args.n_layers, i
                    if args.print_detail == 1:
                        print('train:{} test:{}'.format(len(idx_train_list[i]), len(idx_test_list[i])))
                    print('train:{} test:{}'.format(len(idx_train_list[i]), len(idx_test_list[i])), file=outputfile)
                    tra_acc,acc,prec,rec,f_1,roc_auc,rst = main(args, pggg, info)
                    map_df['{}_{}{}_{:d}'.format(bw, agg, i, int(acc*100))] = rst
                    # LS_map_process.addRst2TIN(dir,data)
                    if args.print_detail == 1:
                        print('{}_{}_{}_Test{}, TrainAcc:{:.4f},Accuracy:{:.4f},Precision:{:.4f},Recall:{:.4f},F_measure:{:.4f},ROCAUC:{:.4f}'.format(
                            data, bw, agg, i, tra_acc,acc,prec,rec,f_1,roc_auc))
                    tra_acclist.append(tra_acc)
                    acclist.append(acc)
                    preclist.append(prec)
                    reclist.append(rec)
                    f1list.append(f_1)
                    rauclist.append(roc_auc)
                    dur.append(time.time() - starttime)

                print('{}_{}_{}_time(s):{:.2f} ↑Accuracy:{:.2f}% Precision:{:.2f}% Recall:{:.2f}% F_measure:{:.2f}% ROCAUC:{:.2f}↑'.format(
                    data, args.bw, args.aggregator_type, np.mean(dur), np.mean(acclist)*100, np.mean(preclist)*100, np.mean(reclist)*100, np.mean(f1list)*100, np.mean(rauclist)*100))
                acc_df['%s_%s_traacc' % (bw, agg)] = tra_acclist
                acc_df['%s_%s_acc' % (bw, agg)] = acclist
                acc_df['%s_%s_pre' % (bw, agg)] = preclist
                acc_df['%s_%s_rec' % (bw, agg)] = reclist
                acc_df['%s_%s_f1' % (bw, agg)] = f1list
                acc_df['%s_%s_auc' % (bw, agg)] = rauclist
                dict_TrAmean[bw].update({agg: np.mean(tra_acclist)})
                dict_Amean[bw].update({agg: np.mean(acclist)})
                dict_Pmean[bw].update({agg: np.mean(preclist)})
                dict_Rmean[bw].update({agg: np.mean(reclist)})
                dict_Fmean[bw].update({agg: np.mean(f1list)})
                dict_AUCmean[bw].update({agg: np.mean(rauclist)})
                # dict_acc[data+ ' bw-'+args.bw + ' agg-'+args.aggregator_type] = acclist
        # pd.DataFrame(dict['FJ'].values(), index=dict['FJ'].keys(), columns=['a', 'b', 'c'])
        map_df.to_csv('{}//maps.csv'.format(dir))
        print(acc_df, file=outputfile)
        print(acc_df)
        name = '%s%s%d_fd:%.1f,ed:%.1f,hi:%d,ly:%d,TR:%.1f' % (args.env, data, num,args.dropout,args.edge_drop,args.n_hidden,args.n_layers,args.test_ratio)
        print(name, file=outputfile)
        print(name)
        # accmean_df[name] = pd.DataFrame(dict_APRFmean)
        # print(accmean_df[name], file=outputfile)
        accmean_df['%s_Train_Acc' % name] = pd.DataFrame(dict_TrAmean)
        accmean_df['%s_Accuracy' % name] = pd.DataFrame(dict_Amean)
        accmean_df['%s_Precision' % name] = pd.DataFrame(dict_Pmean)
        accmean_df['%s_Recall' % name] = pd.DataFrame(dict_Rmean)
        accmean_df['%s_F1' % name] = pd.DataFrame(dict_Fmean)
        accmean_df['%s_ROCAUC' % name] = pd.DataFrame(dict_AUCmean)
        print(accmean_df['%s_Train_Acc' % name])
        print(accmean_df['%s_Train_Acc' % name], file=outputfile)
        print(accmean_df['%s_Accuracy' % name])
        print(accmean_df['%s_Accuracy' % name], file=outputfile)
    result_df = pd.concat(accmean_df.values(), axis=0, keys=accmean_df.keys())
    print(result_df, file=outputfile)
    print(result_df)
    result_df.to_csv('{}sage_result.csv'.format(args.path), mode='a')
    # pd.set_option('precision', 8) #强制保留八位小数
    # mean = {k: float(sum(v) / len(v)) for k, v in dict_acc.items()}
    # print(mean, file=outputfile)
    # print(mean)

    if outputfile != None:
        outputfile.close()  # close后才能看到写入的数据