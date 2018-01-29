import pandas as pd
import numpy as np
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os as os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics
from sklearn import svm
from sklearn.externals import joblib
import src.iolib as il
import src.rnn_test as rnn
import time

project_root_path = il.project_root_path
feature_index_list = il.feature_index_list
feature_index_str = il.feature_index_str
model_path = il.sklearn_model_path
TIME_STEP = il.TIME_STEP


# def get_plot_data(plotdata, datalabel):
#     scatter_normal_x = []
#     scatter_unnormal_x = []
#     new_y = datalabel
#     for i in range(len(plotdata)):
#         if new_y[i] == 1:
#             scatter_normal_x.append(plotdata[i])
#         else:
#             scatter_unnormal_x.append(plotdata[i])
#     scatter_normal_x = np.array(scatter_normal_x)
#     scatter_unnormal_x = np.array(scatter_unnormal_x)
#     return scatter_normal_x, scatter_unnormal_x


def train_save_model(X_train, X_test, y_train, y_test, symbolstr=""):
    sc = sklearn.preprocessing.StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    joblib.dump(sc, model_path + "sc" + symbolstr + ".model")
    total_start = time.clock()
    print("===========Default LR================")
    start = time.clock()
    lr = LogisticRegression(random_state=10)
    lr.fit(X_train_std, y_train)
    y_predprob = lr.predict_proba(X_test_std)[:, 1]
    y_pred = lr.predict(X_test_std)
    y_predprob_train = lr.predict_proba(X_train_std)[:, 1]
    y_pred_train = lr.predict(X_train_std)
    joblib.dump(lr, model_path + "lr" + symbolstr + ".model")
    elapsed = (time.clock() - start)
    print("LR Model Train Time Used: %.4f sec" % elapsed)
    print("LR Default accuracy (Train): ", accuracy_score(y_train.values, y_pred_train))
    print("LR Default AUC Score (Train): %f" % metrics.roc_auc_score(y_train.values, y_predprob_train))
    print("LR Default accuracy (Test): ", accuracy_score(y_test.values, y_pred))
    print("LR Default AUC Score (Test): %f" % metrics.roc_auc_score(y_test.values, y_predprob))
    # print("===========Default SVM================")
    # start = time.clock()
    # clf = svm.SVC(probability=True)
    # clf.fit(X_train_std, y_train)
    # y_predprob = clf.predict_proba(X_test_std)[:, 1]
    # y_pred = clf.predict(X_test_std)
    # y_predprob_train = clf.predict_proba(X_train_std)[:, 1]
    # y_pred_train = clf.predict(X_train_std)
    # joblib.dump(clf, model_path + "svm" + symbolstr + ".model")
    # elapsed = (time.clock() - start)
    # print("SVM Model Train Time Used: %.4f sec" % elapsed)
    # print("SVM Default accuracy (Train): ", accuracy_score(y_train.values, y_pred_train))
    # print("SVM Default AUC Score (Train): %f" % metrics.roc_auc_score(y_train.values, y_predprob_train))
    # print("SVM Default accuracy (Test): ", accuracy_score(y_test.values, y_pred))
    # print("SVM Default AUC Score (Test): %f" % metrics.roc_auc_score(y_test.values, y_predprob))
    print("===========Default RM================")
    start = time.clock()
    # rf0 = RandomForestClassifier(n_estimators=10, min_samples_leaf=10, max_features='sqrt',
    #                              random_state=10,max_depth=7, min_samples_split=80)
    rf0 = RandomForestClassifier(random_state=10)
    rf0.fit(X_train_std, y_train)
    y_predprob = rf0.predict_proba(X_test_std)[:, 1]
    y_pred = rf0.predict(X_test_std)
    y_predprob_train = rf0.predict_proba(X_train_std)[:, 1]
    y_pred_train = rf0.predict(X_train_std)
    joblib.dump(rf0, model_path + "rm" + symbolstr + ".model")
    elapsed = (time.clock() - start)
    print("RM Model Train Time Used: %.4f sec" % elapsed)
    print("RM Default accuracy (Train): ", accuracy_score(y_train.values, y_pred_train))
    print("RM Default AUC Score (Train): %f" % metrics.roc_auc_score(y_train.values, y_predprob_train))
    print("RM Default accuracy (Test): ", accuracy_score(y_test.values, y_pred))
    print("RM Default AUC Score (Test): %f" % metrics.roc_auc_score(y_test.values, y_predprob))
    print("===========Default GBDT================")
    start = time.clock()
    gbm0 = GradientBoostingClassifier(random_state=10)
    gbm0.fit(X_train_std, y_train)
    y_pred = gbm0.predict(X_test_std)
    y_predprob = gbm0.predict_proba(X_test_std)[:, 1]
    y_predprob_train = gbm0.predict_proba(X_train_std)[:, 1]
    y_pred_train = gbm0.predict(X_train_std)
    joblib.dump(gbm0, model_path + "gbdt" + symbolstr + ".model")
    elapsed = (time.clock() - start)
    print("GBDT Model Train Time Used: %.4f sec" % elapsed)
    print("GBDT Default accuracy (Train): ", accuracy_score(y_train, y_pred_train))
    print("GBDT Default AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob_train))
    print("GBDT Default accuracy (Test): ", accuracy_score(y_test, y_pred))
    print("GBDT Default AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))
    elapsed = (time.clock() - total_start)
    print("Total Model Train Time Used: %.4f sec" % elapsed)
    return sc, lr, 0, rf0, gbm0


def load_model(sybol_str=""):
    try:
        start = time.clock()
        sc = joblib.load(model_path + "sc" + sybol_str + ".model")
        lr = joblib.load(model_path + "lr" + sybol_str + ".model")
        rf = joblib.load(model_path + "rm" + sybol_str + ".model")
        # svm = joblib.load(model_path + "svm" + sybol_str + ".model")
        gbm = joblib.load(model_path + "gbdt" + sybol_str + ".model")
        elapsed = (time.clock() - start)
        print("Load model time used:", elapsed)
        return sc, lr, 0, rf, gbm
    except:
        return 0, 0, 0, 0, 0


def predict_batch(sc_model, predict_model, predict_x_list, sample_size=1, feature_size=len(feature_index_list)):
    predict_x_list = predict_x_list[:TIME_STEP * sample_size]
    predict_x = np.array(predict_x_list).reshape((-1, feature_size * TIME_STEP))
    predict_x_norm = sc_model.transform(predict_x)
    predict_y = predict_model.predict(predict_x_norm)
    return predict_y


if __name__ == "__main__":
    pos_filename_list_out, neg_filename_list_out = il.get_filename_list()
    reuse = TIME_STEP
    sybol_str = "_" + "reuse-" + str(reuse) + "_timestep-" + str(TIME_STEP)
    sensor_data = il.get_data(neg_filename_list_out, pos_filename_list_out, sybol_str, reuse)
    X = sensor_data.ix[:, :len(feature_index_list) * TIME_STEP]
    y = sensor_data.ix[:, -1:]  # 标签已经转换成0，1了
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    print("===========Data Insight================")
    train_num = len(X_train)
    test_num = len(X_test)
    print("train size:", train_num, " train distribute: ")
    print(y_train["label"].value_counts())
    print("test size:", test_num, " test distribute:")
    print(y_test["label"].value_counts())
    print("===========Load or Train Model================")
    sc, lr, clf, rf, gbm = load_model(sybol_str)
    if lr == 0:
        sc, lr, clf, rf, gbm = train_save_model(X_train, X_test, y_train, y_test, sybol_str)
    else:
        print("Model load success.")
    print("============Test Sample Data===================")
    test_sample_size = 1
    test_single_x = X_test.values[:test_sample_size].reshape((-1, len(feature_index_list)))
    real_single_y = y_test.values[:test_sample_size]
    test_single_x_list = test_single_x.tolist()
    # test_single_x_list是输入的数据list，shape是(TIME_STEP,feature_size)
    predict_single_y = predict_batch(sc, rf, test_single_x_list, sample_size=test_sample_size)
    print(predict_single_y)
    print(real_single_y.reshape(test_sample_size))
    # print("============RNN Test Sample Data===================")
    # lstm = rnn.SensorLSTM()
    # predict_single_y = lstm.predict_sensor(sc, test_single_x_list, sample_size=test_sample_size)
    # print(predict_single_y)
    # print(real_single_y.reshape(test_sample_size))
    # print("===========Grid Search for n_estimators================")
    # param_test1 = {'n_estimators': range(10, 71, 10)}
    # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
    #                                                          min_samples_leaf=TIME_STEP, max_depth=8, max_features='sqrt',
    #                                                          random_state=10),
    #                         param_grid=param_test1, scoring='roc_auc', cv=5)
    # gsearch1.fit(X_train_std, y_train.values.reshape(len(y_train), ))
    # print(gsearch1.best_params_, gsearch1.best_score_)
    # print("===========Grid Search for max_depth, min_samples_spli================")
    # param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(50, 201, 20)}
    # gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=10,
    #                                                          min_samples_leaf=20, max_features='sqrt',
    #                                                          random_state=10),
    #                         param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    # gsearch2.fit(X_train_std, y_train.values.reshape(len(y_train), ))
    # print(gsearch2.best_params_, gsearch2.best_score_)
    # print("===========Grid Search for min_samples_split, min_samples_leaf================")
    # param_test3 = {'min_samples_split': range(80, 150, 20), 'min_samples_leaf': range(10, 60, 10)}
    # gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=10, max_depth=13,
    #                                                          max_features='sqrt', oob_score=True, random_state=10),
    #                         param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
    # gsearch3.fit(X_train_std, y_train.values.reshape(len(y_train), ))
    # print(gsearch3.best_params_, gsearch3.best_score_)
    # print("===========RM Grid Search Finished================")
    print("============Plot Data with PCA===================")
    il.plot_data_scatter(X_test.values, y_test.values)
    # pca = PCA(n_components=2)
    # sc = sklearn.preprocessing.StandardScaler()
    # sc.fit(X_train)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)
    # newData = pca.fit_transform(X_train_std)
    # scatter_normal_x, scatter_unnormal_x = il.get_plot_data(newData, y_train.values)
    # # scatter_normal_x, scatter_unnormal_x = get_plot_data(X.values,y.values)
    # scatter_normal_x = np.array(scatter_normal_x)
    # scatter_unnormal_x = np.array(scatter_unnormal_x)
    # plt.scatter(scatter_normal_x[:, 0], scatter_normal_x[:, 1], c='b', marker='o')
    # plt.scatter(scatter_unnormal_x[:, 0], scatter_unnormal_x[:, 1], c='r', marker='*')
    # plt.show()

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # #  将数据点分成三部分画，在颜色上有区分度
    # ax.plot(scatter_normal_x[:, 0], scatter_normal_x[:, 1],scatter_normal_x[:, 2], c='b', marker='o')  # 绘制数据点
    # ax.plot(scatter_unnormal_x[:, 0], scatter_unnormal_x[:, 1],scatter_unnormal_x[:, 2], c='r', marker='*')  # 绘制数据点
    # ax.set_zlabel('Z')  # 坐标轴
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # for ii in range(0, 396, 36):
    #     ax.view_init(elev=10., azim=ii)
    #     plt.savefig(project_root_path + r"\pic\movie%d.png" % ii)
    # plt.show()
