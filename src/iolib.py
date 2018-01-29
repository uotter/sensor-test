# -*_coding:utf8-*-
import pandas as pd
import numpy as np
import math as math
import os as os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn as sklearn
from sklearn.model_selection import train_test_split

project_root_path = os.path.abspath('..')
feature_index_list = [0, 1, 2]
TIME_STEP = 200
feature_index_str = "-".join(str(x) for x in feature_index_list)
preprocess_data_path = project_root_path + r"\data\sklearn_" + feature_index_str
sklearn_model_path = project_root_path + r"\model\sklearn_" + feature_index_str
rnn_model_path = project_root_path + r"\model\keras_" + feature_index_str
human_test_data_dir = project_root_path + r"\data\human_test"


def get_human_test_data(human_dir=human_test_data_dir):
    human_dir_file_list = os.listdir(human_dir)  # 列出文件夹下所有的目录与文件
    human_dir_file_name_list = []
    print("========Load Human Test Files==========")
    print("Human Test Files:")
    for i in range(0, len(human_dir_file_list)):
        path = os.path.join(human_dir, human_dir_file_list[i])
        # path = unicode(path , "GB2312")
        print(path)
        if os.path.isfile(path):
            human_dir_file_name_list.append(path)
        else:
            print("File exception:" + path)
    human_test_raw_list = []
    for filename in human_dir_file_name_list:
        f = open(filename)
        human_test_raw = pd.read_csv(f, sep=",", header=None)
        human_test_raw_list.append(human_test_raw)
        f.close()
    columns_list = ["method"]
    columns_list.extend(list(range(0, 120)))
    columns_list.extend(["type", "predict", "real"])
    human_test_df_total = pd.concat(human_test_raw_list, axis=0)
    human_test_df_total.columns = columns_list
    new_columns_list = list(range(0, 60))
    ori_human_test_df = pd.DataFrame(columns=new_columns_list)
    add_human_test_df = pd.DataFrame(columns=new_columns_list)
    for index, row in human_test_df_total.iterrows():
        type = row["type"]
        predict_y = 0 if row["predict"] == "bot" else 1
        real_y = 0 if row["real"] == "bot" else 1
        if type == "add":
            call_columns_list = list(range(0, 60))
            call_columns_list.extend(["predict", "real"])
            data_se = row.loc[call_columns_list]
            add_human_test_df = add_human_test_df.append(data_se, ignore_index=True)
        elif type == "ori":
            call_columns_list = list(range(60, 120))
            call_columns_list.extend(["predict", "real"])
            data_se = row.loc[call_columns_list]
            ori_call_columns_list = list(range(0, 60))
            ori_call_columns_list.extend(["predict", "real"])
            data_se.index = ori_call_columns_list
            ori_human_test_df = ori_human_test_df.append(data_se, ignore_index=True)
    return ori_human_test_df, add_human_test_df


def get_data(neg_filename_list, pos_filename_list, sybol_str="", reuse_step=1, force_reperprocess=False, savedata=True):
    print("========Load or Preprocess Data==========")
    if os.path.isfile(preprocess_data_path + sybol_str + ".csv") and not force_reperprocess:
        df_total = pd.read_csv(preprocess_data_path + sybol_str + ".csv")
        print("Data Load Success.")
    else:
        df_neg_raw_list = []
        df_pos_raw_list = []
        for neg_filename in neg_filename_list:
            f = open(neg_filename)
            df_neg_raw = pd.read_csv(f, sep=" ", dtype="float64", header=None)
            maxline = int(len(df_neg_raw)/TIME_STEP)
            df_neg_raw = df_neg_raw.ix[:maxline*TIME_STEP,:]
            df_neg_raw_list.append(df_neg_raw)
            f.close()
        df_neg_total = pd.concat(df_neg_raw_list, axis=0)
        df_neg_total = df_neg_total[feature_index_list]
        for pos_filename in pos_filename_list:
            f = open(pos_filename)
            df_pos_raw = pd.read_csv(f, sep=" ", dtype="float64", header=None)
            maxline = int(len(df_pos_raw) / TIME_STEP)
            df_pos_raw = df_pos_raw.ix[:maxline * TIME_STEP, :]
            df_pos_raw_list.append(df_pos_raw)
            f.close()
        df_pos_total = pd.concat(df_pos_raw_list, axis=0)
        df_pos_total = df_pos_total[feature_index_list]
        set_pos_num = math.inf
        set_neg_num = math.inf
        pos_raw_sample_num = (len(df_pos_total) - TIME_STEP) if (len(
            df_pos_total) - TIME_STEP) < set_pos_num else set_pos_num
        neg_raw_sample_num = (len(df_neg_total) - TIME_STEP) if (len(
            df_neg_total) - TIME_STEP) < set_neg_num else set_neg_num
        list_pos_reuse = []
        list_neg_reuse = []
        print("Get total positive number: %d" % (len(df_pos_total) - TIME_STEP))
        for i in range(0, pos_raw_sample_num, reuse_step):
            if i % 10000 == 0:
                print("Get process: %d/%d" % (i, pos_raw_sample_num))
            iterdf = df_pos_total.iloc[i:(i + TIME_STEP), :]
            iterlist = iterdf.values.reshape((-1)).tolist()
            iterlist.append(1)
            list_pos_reuse.append(np.array(iterlist))
        print("Get total negative number: %d" % (len(df_neg_total) - TIME_STEP))
        for i in range(0, neg_raw_sample_num, reuse_step):
            if i % 10000 == 0:
                print("Get process: %d/%d" % (i, neg_raw_sample_num))
            iterdf = df_neg_total.iloc[i:(i + TIME_STEP), :]
            iterlist = iterdf.values.reshape((-1)).tolist()
            iterlist.append(0)
            list_neg_reuse.append(np.array(iterlist))
        df_total_columns_list = list(range(TIME_STEP * len(feature_index_list)))
        df_total_columns_list.append("label")
        df_neg_part = pd.DataFrame(np.array(list_neg_reuse), columns=df_total_columns_list)
        df_pos_part = pd.DataFrame(np.array(list_pos_reuse), columns=df_total_columns_list)
        # df_neg_part = pd.DataFrame(
        #     df_neg_reuse[feature_index_list].values.reshape((int(len(df_neg_reuse) / 20)), len(feature_index_list) * 20))
        # df_pos_part = pd.DataFrame(
        #     df_pos_reuse[feature_index_list].values.reshape((int(len(df_pos_reuse) / 20)), len(feature_index_list) * 20))
        # df_neg_part.insert(len(df_neg_part.columns), "label", 0)
        # df_pos_part.insert(len(df_pos_part.columns), "label", 1)
        df_total = pd.concat([df_neg_part, df_pos_part], axis=0)
        if savedata:
            df_total.to_csv(preprocess_data_path + sybol_str + ".csv")
        print("Data Preprocess and Load Success.")
    return df_total


def get_plot_data(plotdata, datalabel):
    scatter_normal_x = []
    scatter_unnormal_x = []
    new_y = datalabel
    for i in range(len(plotdata)):
        if new_y[i] == 1:
            scatter_normal_x.append(plotdata[i])
        else:
            scatter_unnormal_x.append(plotdata[i])
    scatter_normal_x = np.array(scatter_normal_x)
    scatter_unnormal_x = np.array(scatter_unnormal_x)
    return scatter_normal_x, scatter_unnormal_x


def plot_data_scatter(scatter_x, scatter_y, fit_transform=True):
    pca = PCA(n_components=2)
    if fit_transform:
        sc_model = sklearn.preprocessing.StandardScaler()
        scatter_x = sc_model.fit_transform(scatter_x)
        scatter_x = pca.fit_transform(scatter_x)
    scatter_normal_x, scatter_unnormal_x = get_plot_data(scatter_x, scatter_y)
    # scatter_normal_x, scatter_unnormal_x = get_plot_data(X.values,y.values)
    scatter_normal_x = np.array(scatter_normal_x)
    scatter_unnormal_x = np.array(scatter_unnormal_x)
    plt.scatter(scatter_normal_x[:, 0], scatter_normal_x[:, 1], c='b', marker='o')
    plt.scatter(scatter_unnormal_x[:, 0], scatter_unnormal_x[:, 1], c='r', marker='*')
    plt.title('Data PCA.')
    plt.ylabel('x1')
    plt.xlabel('x2')
    plt.legend(['normal', 'unnormal'], loc='upper right')
    plt.show()


def get_filename_list(pos_file_dir=project_root_path + r"\data\pos", neg_file_dir=project_root_path + r"\data\neg"):
    pos_dir_file_list = os.listdir(pos_file_dir)  # 列出文件夹下所有的目录与文件
    neg_dir_file_list = os.listdir(neg_file_dir)
    pos_file_name_list = []
    neg_file_name_list = []
    print("========Load Files==========")
    print("Positive Files:")
    for i in range(0, len(pos_dir_file_list)):
        path = os.path.join(pos_file_dir, pos_dir_file_list[i])
        # path = unicode(path , "GB2312")
        print(path)
        if os.path.isfile(path):
            pos_file_name_list.append(path)
        else:
            print("File exception:" + path)
    print("Negative Files:")
    for i in range(0, len(neg_dir_file_list)):
        path = os.path.join(neg_file_dir, neg_dir_file_list[i])
        # path = path.decode('gbk')
        print(path)
        if os.path.isfile(path):
            neg_file_name_list.append(path)
        else:
            print("File exception:" + path)
    return pos_file_name_list, neg_file_name_list


if __name__ == "__main__":
    # pos_file_name_list, neg_file_name_list = get_filename_list()
    # neg_file_name_list = [project_root_path + r"\data\neg\计步器1.csv",
    #                       project_root_path + r"\data\neg\计步器2.csv",
    #                       project_root_path + r"\data\neg\计步器3.csv",
    #                       project_root_path + r"\data\neg\计步器4.csv",
    #                       project_root_path + r"\data\neg\计步器5.csv"]
    # pos_file_name_list = [project_root_path + r"\data\pos\handwalk_train.csv",
    #                       project_root_path + r"\data\pos\handwalk_test.csv",
    #                       project_root_path + r"\data\pos\handtouch1_train.csv",
    #                       project_root_path + r"\data\pos\handtouch1_test.csv",
    #                       project_root_path + r"\data\pos\handtouch2_train.csv",
    #                       project_root_path + r"\data\pos\handtouch2_test.csv",
    #                       project_root_path + r"\data\pos\pos_righthand.csv",
    #                       project_root_path + r"\data\pos\pos_slip_down.csv",
    #                       project_root_path + r"\data\pos\pos_face_right_laydown.csv",
    #                       project_root_path + r"\data\pos\pos_face_left_laydown.csv",
    #                       project_root_path + r"\data\pos\pos_face_down_laydown.csv",
    #                       project_root_path + r"\data\pos\pos_45ang_laydown.csv",
    #                       project_root_path + r"\data\pos\pos_handwalk.csv",
    #                       project_root_path + r"\data\pos\pos_soft_touch_faceup1.csv",
    #                       project_root_path + r"\data\pos\pos_soft_touch_faceup2.csv"]
    # sybol_str = ""
    # sensor_data = get_data(neg_file_name_list, pos_file_name_list, sybol_str, TIME_STEP, True, False)
    # X = sensor_data.ix[:, :len(feature_index_list) * TIME_STEP]
    # y = sensor_data.ix[:, -1:]  # 标签已经转换成0，1了
    # x_arr = X.values
    # y_arr = y.values
    # plot_data_scatter(x_arr, y_arr, False)
    ori_human_test_df, add_human_test_df = get_human_test_data(human_dir=human_test_data_dir)
    ori_human_test_list = ori_human_test_df.ix[:, 0:60].values.tolist()
    add_human_test_list = add_human_test_df.ix[:, 0:60].values.tolist()
    ori_predict_y = ori_human_test_df.ix[:, 60:61].values.reshape(28).tolist()
    ori_real_y = ori_human_test_df.ix[:, 61:62].values.reshape(28).tolist()
    print(ori_predict_y)
    print(ori_real_y)
