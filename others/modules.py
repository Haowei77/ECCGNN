import argparse
import torch as th
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
from outils import preprocessing, preprocessing_plus
import time
import os
import geopandas

def accuracy(logits, labels):
    _, indices = th.max(logits, dim=1)
    _2, indices2 = th.max(labels, dim=1)
    correct = th.sum(indices == indices2)
    # correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def predict(pggg,model_name='ANN',ti=0):
    # 数据准备
    ids, xs, ts, idx_train, idx_test, idx_val = pggg
    x_train = xs[idx_train]  # 加载i行数据部分
    y_train = ts[idx_train]  # 加载类别标签部分
    x_test = xs[idx_test]
    y_test = ts[idx_test]
    # 建模
    if model_name=='SVM':
        # clf = svm.SVC(C=1, kernel='rbf', gamma=1 / (2 * x_train.var()), decision_function_shape='ovr', probability=True)
        model = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr', probability=True)
    if model_name=='RF':
        # model = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
        # model = RandomForestClassifier(n_estimators=10, max_features=n_features, max_depth=None, min_samples_split=2,bootstrap=True)
        model = RandomForestClassifier(n_estimators=100, max_features="auto", max_depth=None, min_samples_split=2,bootstrap=True)
        # model = ExtraTreesClassifier(n_estimators=10, max_features=n_features, max_depth=None, min_samples_split=2,bootstrap=False)
    if model_name=='ANN':
        model = MLPClassifier(hidden_layer_sizes=(16, 16), activation='relu', solver='adam', learning_rate_init=0.001, alpha=0.0001, max_iter=1000,early_stopping=False)

    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    trainAcc = metrics.accuracy_score(y_train, pred_train)
    print('%s train_Accuracy: %f' % (model_name,trainAcc))
    pred_test = model.predict(x_test)
    # print('%s test_Accuracy: %f' % (model_name,metrics.accuracy_score(y_test, pred_test)))
    TP = ((pred_test == 1) * (y_test == 1)).astype(int).sum()
    FP = ((pred_test == 1) * (y_test == 0)).astype(int).sum()
    FN = ((pred_test == 0) * (y_test == 1)).astype(int).sum()
    TN = ((pred_test == 0) * (y_test == 0)).astype(int).sum()
    Accuracy = (TP+TN)/(TP+FN+FP+TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F_measures = 2 * Precision * Recall / (Precision + Recall)

    print('Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F_measures: %.4f' %
          (Accuracy,Precision,Recall,F_measures))
    predRst = model.predict_proba(xs) # predict_proba
    # map_df = pd.DataFrame(predRst.tolist(), index=ids)

    # Test AUROC
    fpr, tpr, thre = metrics.roc_curve(y_test, predRst[idx_test,1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print('ROC curve (area = %0.2f)' % roc_auc)
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
    plt.savefig('{}\\othersRST-{}\\{}_{}_roc{}.jpg'.format(args.path, data, data, model_name, ti))
    return trainAcc,Accuracy,Precision,Recall,F_measures,roc_auc,predRst

def nn(pggg,ti=0):
    # 数据准备
    ids, xs, ts, idx_train, idx_test, idx_val = pggg
    labels = np.zeros(shape=[ts.shape[0], 2])
    labels[ts == 0, 0] = 1
    labels[ts == 1, 1] = 1

    ###########################
    # xs:单元的属性向量，labels：单元标签，idx_train：训练集，idx_test：测试集
    xs = th.Tensor(xs).to(device)
    labels = th.FloatTensor(labels).to(device)
    idx_train = th.LongTensor(idx_train).to(device)
    idx_test = th.LongTensor(idx_test).to(device)
    env_dim = 1     # env_dim：灾害环境维度
    class_num = 2   # 分类数量
    h_dim = 16      # 隐藏层维度

    # 神经网络模型
    model = th.nn.Sequential(
        th.nn.Linear(xs.shape[1], h_dim),  # w1*x+b1
        th.nn.ReLU(),
        th.nn.Linear(h_dim, class_num),     # w2
    )
    model.to(device)
    # 初始化权重参数
    gain = th.nn.init.calculate_gain('relu')
    th.nn.init.xavier_uniform_(model[0].weight, gain=gain) # w1
    th.nn.init.xavier_uniform_(model[2].weight, gain=gain) # w2
    # 学习率
    learning_rate = 0.05
    loss_fcn = th.nn.BCEWithLogitsLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    iteration = 150
    for epoch in range(iteration):
        logits = model(xs)
        # loss

        train_loss = loss_fcn(logits[idx_train], labels[idx_train])
        train_acc = accuracy(logits[idx_train], labels[idx_train])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with th.no_grad():
            logits = model(xs)
            eval_loss = loss_fcn(logits[idx_test], labels[idx_test])
            eval_acc = accuracy(logits[idx_test], labels[idx_test])
            # for param in model.parameters():
            #     param -= learning_rate * param.grad
        model.zero_grad()
        print("Epoch {:05d} | Loss {:.4f} | TrainAcc {:.4f} | ValLoss{:.4f} | ValAcc {:.4f} ".format(
            epoch, train_loss.item(), train_acc, eval_loss, eval_acc))

    # map = F.softmax(logits)
    # map_df = pd.DataFrame(map.tolist(), index=ids)
    # map_df.to_csv('nn_result.csv',header=None)
    return eval_acc

def plot():
    x_values = list(range(11))
    y_values = [x ** 2 for x in x_values]
    y2_values = [11-x for x in x_values]
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x_values, y_values, c='green')
    ax2.plot(x_values, y2_values, c='red')
    ax1.set_xlabel('epoch\tacc = %.4f'%77.76666)
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy')
    plt.title('Squares', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Numbers', fontsize=14)
    plt.ylabel('Squares', fontsize=14)
    x_major_locator = plt.MultipleLocator(1)
    y_major_locator = plt.MultipleLocator(10)
    y2_major_locator = plt.MultipleLocator(1)
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.yaxis.set_major_locator(y_major_locator)
    ax2.yaxis.set_major_locator(y2_major_locator)
    ax2.axis([-0.5, 11, -0.5, 12])
    ax1.axis([-0.5, 11, -5, 110])
    # 把y轴的主刻度设置为10的倍数
    # plt.xlim(-0.5, 11)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    # plt.ylim(-0.5, 110)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.show()

def addRst2TIN(pddata,name):
    # mappath = "D:\\OneDrive - my.swjtu.edu.cn\\项目\\Code\\GNN\\landslide\\0 Final\\ANN_map5.csv"
    # data = pd.read_csv(mappath, usecols=[0,2], names=['Tri_Index', 'B'])
    # 0 利用geopandas可视化矢量
    path = r'D:\OneDrive - my.swjtu.edu.cn\项目\Code\滑坡数据\LSM_result'
    file = os.path.join(path, "%s_tin.shp"%(name))
    tin = geopandas.read_file(file)
    print('read shp')
    # ['Tri_Index', 'Shape_Leng', 'Shape_Area', 'NewLabel', 'geometry']
    pddata['Tri_Index'] = pddata.index
    data_geod = geopandas.GeoDataFrame(pddata)
    print('merge')
    gdb = tin.merge(data_geod, on='Tri_Index')
    print('save...')
    saevfile = os.path.join(path, "%s_Other_RST.shp"%(name))
    t0 = time.time()
    gdb.to_file(saevfile)
    print('shp done %.1fs %s'%((time.time() - t0),saevfile))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    # register_data_args(parser)
    parser.add_argument('--path',
                        type=str, default='../../landslide/')
    parser.add_argument('--test-ratio',
                        type=float, default=-0.3)
    parser.add_argument("--env",
                        type=int, default=0, help="1 landslide formative enviroment; 0")

    device = th.device('cuda:0')
    args = parser.parse_args()

    # methods = ['NN','SVM','RF']
    accmean_df = {}
    for data in ['FL']:
        num = 10
        dur = []
        if args.env==1:
            ids, xs, ts, idx_train_list, idx_test_list, idx_val_list = \
                preprocessing_plus(args.path, data, args.test_ratio, num, device=device)
            print("preprocessing_env")
        else:
            ids, xs, ts, idx_train_list, idx_test_list, idx_val_list = \
                preprocessing(args.path, data, args.test_ratio, num, device=device)
            print("preprocessing")
        other_maps = pd.DataFrame(index=ids)
        acc_df = pd.DataFrame()
        dict_TrAmean = {}
        dict_Amean = {}
        dict_Pmean = {}
        dict_Rmean = {}
        dict_Fmean = {}
        dict_AUCmean = {}
        print('train: %d, test: %d' % (idx_train_list[0].shape[0], idx_test_list[0].shape[0]))
        for model_name in ['ANN','SVM','RF']:  # ['ANN','SVM','RF']
            trainacclist = []
            acclist = []
            preclist = []
            reclist = []
            f1list = []
            AUClist = []
            # dict_TrAmean[model_name] = {}
            # dict_Amean[model_name] = {}
            # dict_Pmean[model_name] = {}
            # dict_Rmean[model_name] = {}
            # dict_Fmean[model_name] = {}
            # dict_AUCmean[model_name] = {}
            for i in range(num):
                starttime = time.time()
                pggg = ids, xs, ts, idx_train_list[i], idx_test_list[i], idx_val_list[i]
                trainAcc,Accuracy,Precision,Recall,F_measures,roc_auc,predRst = predict(pggg,model_name,i)
                other_maps['{}{}_{:.0f}'.format(model_name, i, Accuracy*100)] = predRst[:,1].tolist()
                dur.append(time.time() - starttime)
                # print('%s %s acc: %.2f' % (data, model, Accuracy * 100))
                trainacclist.append(trainAcc)
                acclist.append(Accuracy)
                preclist.append(Precision)
                reclist.append(Recall)
                f1list.append(F_measures)
                AUClist.append(roc_auc)
            acc_df['%s_tracc' % (model_name)] = trainacclist
            acc_df['%s_acc' % (model_name)] = acclist
            acc_df['%s_pre' % (model_name)] = preclist
            acc_df['%s_rec' % (model_name)] = reclist
            acc_df['%s_f1' % (model_name)] = f1list
            acc_df['%s_auc' % (model_name)] = AUClist
            dict_TrAmean['%s_%.1f' % (model_name,args.test_ratio)]=np.mean(trainacclist)
            dict_Amean['%s_%.1f' % (model_name,args.test_ratio)]=np.mean(acclist)
            dict_Pmean['%s_%.1f' % (model_name,args.test_ratio)]=np.mean(preclist)
            dict_Rmean['%s_%.1f' % (model_name,args.test_ratio)]=np.mean(reclist)
            dict_Fmean['%s_%.1f' % (model_name,args.test_ratio)]=np.mean(f1list)
            dict_AUCmean['%s_%.1f' % (model_name,args.test_ratio)]=np.mean(AUClist)
            print('{}:{} time(s):{:.2f} TAcc:{:.2f} ↑ Accuracy:{:.2f}% Precision:{:.2f}% Recall:{:.2f}% '
                  'F_measure:{:.2f}% ROCAUC:{:.2f}% ↑\n'.format(
                data, model_name, np.mean(dur), np.mean(trainacclist) * 100, np.mean(acclist) * 100, np.mean(preclist) * 100,
                np.mean(reclist) * 100, np.mean(f1list) * 100, np.mean(AUClist) * 100))
        print(acc_df)
        acc_df.to_csv('%s\\othersRST-%s\\%s_othersRst.csv' % (args.path, data, data))
        accmean_df['Train_Acc'] = (dict_TrAmean)
        accmean_df['Accuracy'] = (dict_Amean)
        accmean_df['Precision'] = (dict_Pmean)
        accmean_df['Recall'] = (dict_Rmean)
        accmean_df['F1'] = (dict_Fmean)
        accmean_df['ROCAUC'] = (dict_AUCmean)
        other_maps.to_csv('%s\\othersRST-%s\\%s_othersMap%s.csv'%(args.path,data, data, args.test_ratio))
    # result_df = pd.DataFrame(accmean_df.values(), index=accmean_df.keys())
    result_df = pd.DataFrame(accmean_df)
    print(result_df)
    result_df.to_csv('{}\\othersRST-{}\\others_results.csv'.format(args.path, data), mode='a') # mode='a'，便可以追加写入数据。
        # addRst2TIN(other_maps,data)