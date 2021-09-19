
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.classification import accuracy_score


import pandas as pd

from denoising_AE import *
from utils import *
from dbn.tensorflow import SupervisedDBNClassification

def proposed(tmp_feature):
    np.random.shuffle(tmp_feature)  # shuffle
    label_attr = tmp_feature[:, -1].astype(np.float32)  # 加载类别标签部分
    data_atrr = tmp_feature[:, :-1].astype(np.float32)  # 加载i行数据部分

    X_train = data_atrr[:1200, :]
    X_test = data_atrr[1200:, :]
    Y_train = label_attr[:1200]
    Y_test = label_attr[1200:]
    # 归一化
    X_train = X_train / X_train.max(axis=0)
    X_test = X_test / X_test.max(axis=0)
    # Pretrain(Graph 0)
    weights = []
    reconstruction_error = []
    with tf.variable_scope('deepBM'):
        classifier = SupervisedDBNClassification(hidden_layers_structure=[32, 32],  # rbm隐藏层列表
                                                # hidden_layers_structure1=[300, 100],   # dae隐藏层列表（作废）
                                                 learning_rate_rbm=0.001,
                                                 learning_rate=0.01,
                                                 n_epochs_rbm=20,
                                                 n_iter_backprop=100,
                                                 batch_size=32,
                                                 activation_function='sigmoid',
                                                 dropout_p=0.1)
        # RBM fit
        classifier.fit(X_train, weights, reconstruction_error)
    #  绘制rbm_error图,查看多大的节点数设置才能较好拟合原始数据分布
    # for i in range(len(reconstruction_error)):
    #     temp_x = []
    #     for j in range(len(reconstruction_error[i])):
    #         temp_x.append(j)
    #     plt.xlabel("n_epoch")
    #     plt.ylabel("loss")
    #     plt.plot(temp_x, reconstruction_error[i], marker="s")
    #     plt.savefig("rbm_error" + str(i) + ".png", dpi=120)

    rbm_weights = []
    rbm_bias = []

    # transform data by deepBM for dae pretraining(Graph 1)
    activations = tf.constant(X_train, tf.float32)
    for i in range(len(weights)):
        with tf.variable_scope('weights2merge'):  # for merge weights
            rbm_weights.append(tf.Variable(weights[i]['w']))
            rbm_bias.append(tf.Variable(weights[i]['b']))
        activations = transform_sigmoid(activations,
                        rbm_weights[-1], rbm_bias[-1])  # 这里activations用两次会不会产生问题图
        #X_train_dae = tf.transpose(X_train_dae)
    X_train_dae = activations

    with tf.Session() as sess:
        for i in range(len(rbm_weights)):
            sess.run(tf.variables_initializer([rbm_weights[i], rbm_bias[i]]))
        X_train_dae = X_train_dae.eval()
        # 查看deepBM权值
        # weights0 = rbm_weights[0].eval()
        # weights1 = rbm_weights[1].eval()
        # bias0 = rbm_bias[0].eval()
        # bias1 = rbm_bias[1].eval()
    # 查看当前variables
    # current_weights1 = tf.global_variables()  # 10组variables
    # current_weights5 = tf.global_variables(scope='weights2merge')

    # DAE fit(Graph 2)
    # 超参数设置
    input_units = int(X_train_dae.shape[1])  # dae输入节点
    structure = [16]
    n_samples = int(X_train_dae.shape[0])
    training_epochs = 50
    batch_size = 16
    display_step = 1
    dae_weights = []  # 存储dae预训练权重参数
    dae_bias = []  # 存储dae预训练偏置参数
    activations = X_train_dae

    for hidden_units in structure:
        with tf.variable_scope('DAE'):
            autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=input_units, n_hidden=hidden_units,
                                                       transfer_function=tf.nn.softplus,
                                                       optimizer=tf.train.AdamOptimizer(learning_rate=0.00005), scale=0.01)
        print("[START] DAE training step:")
        current_weights2 = tf.global_variables()  # just see if right exist the weights
        for epoch in range(training_epochs):
            cost = 0.
            total_batch = int(n_samples / batch_size)
            for i in range(total_batch):
                batch_xs = get_random_block_from_data(activations, batch_size)  # 不放回抽样
                cost = autoencoder.partial_fit(batch_xs)  # 此处计算loss并优化权参
            print(">> Epoch %d finished \tDAE training loss %f" % (epoch, cost))

        with tf.variable_scope('weights2merge'):  # for merge weights:
            dae_weights.append(tf.Variable(autoencoder.sess.run(tf.transpose(autoencoder.weights['w1']))))
            dae_bias.append(tf.Variable(autoencoder.sess.run(autoencoder.weights['b1'])))
        input_units = hidden_units
        #current_weights = tf.global_variables()  # just see if right exist the weights

        activations = transform_softplus(activations, dae_weights[-1], dae_bias[-1])
        # tensor转array
        with tf.Session() as sess:
            for i in range(len(dae_weights)):
                sess.run(tf.initialize_variables([dae_weights[i], dae_bias[i]]))
            activations = activations.eval()

    #current_weights6 = tf.global_variables(scope='weights2merge')
    # with tf.Session() as sess:  # 查看DAE权值
    #     sess.run(tf.variables_initializer(current_weights6))
    #     weights0 = dae_weights[0].eval()
    #     weights1 = dae_weights[1].eval()
    #     bias0 = dae_bias[0].eval()
    #     bias1 = dae_bias[1].eval()

    #  堆叠deepBM和stacked DAE(融合rbm参数和dae参数)
    weights_concat = list()
    for i in range(len(rbm_weights)):
        temp_dict = {}
        temp_dict['w'] = rbm_weights[i]
        temp_dict['b'] = rbm_bias[i]
        weights_concat.append(temp_dict)
    for i in range(len(dae_weights)):
        temp_dict = {}
        temp_dict['w'] = dae_weights[i]
        temp_dict['b'] = dae_bias[i]
        weights_concat.append(temp_dict)
    with tf.Session() as sess:
        init = tf.global_variables(scope='weights2merge')
        sess.run(tf.variables_initializer(init))
        for i in range(len(weights_concat)):
            b = weights_concat[i]['b'].eval()
            w = weights_concat[i]['w'].eval()

    # supervised learning
    # 超参数设置
    n_iter_backprop = 2000  # 奉节：2000;涪陵:1000
    batch_size = 32
    keep_prob = 0.9  # drop_out节点激活概率
    #current_weights3 = tf.global_variables()

    SV_weights = build_and_train_SVmodel1(X_train, Y_train, weights_concat, X_train.shape[1],
                                                 n_iter_backprop, batch_size, keep_prob)

    # test the model
    # 保存权值信息
    count = len(SV_weights)
    np.savez('logs/savedmodel',SV_weights[0]['w'],SV_weights[0]['b'],
             SV_weights[1]['w'],SV_weights[1]['b'],
             SV_weights[2]['w'],SV_weights[2]['b'],
             SV_weights[3]['w'],SV_weights[3]['b'])

    # 前向传播预测
    # test
    Y_vec = X_test
    for inlayers in range(len(SV_weights)-1):
        Y_vec = transform_relu(Y_vec, SV_weights[inlayers]['w'], SV_weights[inlayers]['b'])  # 这里transform有问题
    Y_vec = tf.transpose(tf.matmul(SV_weights[-1]['w'], tf.transpose(Y_vec))) + SV_weights[-1]['b']
    with tf.Session() as sess:
        Y_vec = Y_vec.eval(session=sess)
    output = tf.nn.softmax(Y_vec)

    # train
    Y_vec1 = X_train
    for inlayers in range(len(SV_weights)-1):
        Y_vec1 = transform_relu(Y_vec1, SV_weights[inlayers]['w'], SV_weights[inlayers]['b'])  # 这里transform有问题
    Y_vec1 = tf.transpose(tf.matmul(SV_weights[-1]['w'], tf.transpose(Y_vec1))) + SV_weights[-1]['b']
    with tf.Session() as sess:
        Y_vec1 = Y_vec1.eval(session=sess)
    output1 = tf.nn.softmax(Y_vec1)
    # initial = tf.global_variables(scope='weights_for_predicting')

    with tf.Session() as sess:
        Y_array1 = sess.run(output1)
        Y_pred1 = np.argmax(Y_array1, 1)
        print('Done.\nTrain_Accuracy: %f' % accuracy_score(Y_train, Y_pred1))

        Y_array = sess.run(output)
        Y_pred = np.argmax(Y_array, 1)
        print('Done.\nTest_Accuracy: %f' % accuracy_score(Y_test, Y_pred))  # 0.74 - 0.77

        return Y_train, Y_array1, Y_test, Y_array  # train_label, train_prediction, test_label, test_prediction

############################################landslide data####################################
data = pd.read_excel('C:\\Users\\hj\\Desktop\\奉节隐患点属性表_输入数据_added_SVM.xls','sheet 1',index_col=0)
data.to_csv('C:\\Users\\hj\\Desktop\\data.csv',encoding='utf-8')
tmp = np.loadtxt('C:\\Users\\hj\\Desktop\\data.csv', dtype=np.str, delimiter=",",encoding='UTF-8')
tmp_feature = tmp[1:,:]
ROC_fold_num = 2

############################################# the SVM ROC ####################################
mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
mean_fpr = np.linspace(0, 1, 100)
cnt = 0
clf = svm.SVC(C=1, kernel='rbf', gamma=0.5, decision_function_shape='ovr', probability=True)
for i in range(ROC_fold_num):
    cnt += 1
    np.random.shuffle(tmp_feature)  # shuffle
    # 训练集
    x_train = tmp_feature[:1200, :-1].astype(np.float32)  # 加载i行数据部分
    y_train = tmp_feature[:1200, -1].astype(np.float32)  # 加载类别标签部分
    x_train = x_train / x_train.max(axis=0)
    # 测试集
    x_test = tmp_feature[1200:, :-1].astype(np.float32)  # 加载i行数据部分
    y_test = tmp_feature[1200:, -1].astype(np.float32)  # 加载类别标签部分
    x_test = x_test / x_test.max(axis=0)
    # fit data
    clf.fit(x_train, y_train)
    # test
    y_pred = clf.predict_proba(x_test)[:, -1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)  # 求auc面积
  #  plt.plot(fpr, tpr, lw=1, color=(0.2, 0.2, 0.2))  # 画出当前分割数据的ROC曲线(, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, roc_auc))

mean_tpr /= cnt  # 求数组的平均值
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, 'k--', label='SVM ROC (area = {0:.3f})'.format(mean_auc), lw=2, color=(1, 0, 0))
############################################## the MLP ROC ####################################
mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
mean_fpr = np.linspace(0, 1, 100)
cnt = 0
# 建模
model = MLPClassifier(hidden_layer_sizes=(32, 32, 16), activation='relu', solver='adam', alpha=0.0001,
                      batch_size=32, max_iter=1000)

for i in range(ROC_fold_num):
    cnt += 1
    np.random.shuffle(tmp_feature)  # shuffle
    # 训练集
    x_train = tmp_feature[:1200, :-1].astype(np.float32)  # 加载i行数据部分
    y_train = tmp_feature[:1200, -1].astype(np.float32)  # 加载类别标签部分
    x_train = x_train / x_train.max(axis=0)
    # 测试集
    x_test = tmp_feature[1200:, :-1].astype(np.float32)  # 加载i行数据部分
    y_test = tmp_feature[1200:, -1].astype(np.float32)  # 加载类别标签部分
    x_test = x_test / x_test.max(axis=0)
    # fit data
    model.fit(x_train, y_train)
    # test predict
    y_pred = model.predict_proba(x_test)[:, -1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)  # 求auc面积
  #  plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, roc_auc))  # 画出当前分割数据的ROC曲线

mean_tpr /= cnt  # 求数组的平均值
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, 'k--', label='MLP ROC (area = {0:.3f})'.format(mean_auc), lw=2, color=(0, 1, 0))
############################################## the RF ROC ####################################
mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
mean_fpr = np.linspace(0, 1, 100)
cnt = 0
clf2 = RandomForestClassifier(n_estimators=10, max_features = 17, max_depth=None, min_samples_split=2,
                              bootstrap=True)
for i in range(ROC_fold_num):
    cnt += 1
    np.random.shuffle(tmp_feature)  # shuffle
    # 训练集
    x_train = tmp_feature[:1200, :-1].astype(np.float32)  # 加载i行数据部分
    y_train = tmp_feature[:1200, -1].astype(np.float32)  # 加载类别标签部分
    x_train = x_train / x_train.max(axis=0)
    # 测试集
    x_test = tmp_feature[1200:, :-1].astype(np.float32)  # 加载i行数据部分
    y_test = tmp_feature[1200:, -1].astype(np.float32)  # 加载类别标签部分
    x_test = x_test / x_test.max(axis=0)
    # fit data
    clf2.fit(x_train, y_train)
    # test predict
    y_pred = clf2.predict_proba(x_test)[:, -1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)  # 求auc面积
#  plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, roc_auc))  # 画出当前分割数据的ROC曲线

mean_tpr /= cnt  # 求数组的平均值
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, 'k--', label='RF ROC (area = {0:.3f})'.format(mean_auc), lw=2, color=(0, 0, 1))

############################################## the proposed ROC ####################################
mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
mean_fpr = np.linspace(0, 1, 100)
cnt = 0
for i in range(ROC_fold_num):
    cnt += 1
    np.random.shuffle(tmp_feature)  # shuffle
    Y_train, Y_pred1, Y_test, Y_pred = proposed(tmp_feature) #暂且只求test_ROC
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred[:, -1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)  # 求auc面积
    #plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, roc_auc))  # 画出当前分割数据的ROC曲线
mean_tpr /= cnt  # 求数组的平均值
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label='Proposed ROC (area = {0:.3f})'.format(mean_auc), lw=2, color=(0, 0, 0))

################################################################################################
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')  # 画对角线
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate(1-Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')  # 可以使用中文，但需要导入一些库即字体
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig("fj_ROC.pdf")
plt.show()



