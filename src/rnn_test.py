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
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
import src.iolib as il

feature_index_list = il.feature_index_list
epoc_times = 50
feature_index_str = il.feature_index_str
TIME_STEP = il.TIME_STEP
INPUT_DIV = len(feature_index_list)
OUTPUT_DIV = 1
BATCH_SIZE = 100
reuse = TIME_STEP
sybol_str = "_" + "reuse-" + str(reuse) + "_timestep-" + str(TIME_STEP)
project_root_path = il.project_root_path
model_path = il.rnn_model_path + sybol_str + ".model"


# def get_rnn_data(neg_filename_list, pos_filename_list):
#     global feature_index_list
#     df_neg_raw_list = []
#     df_pos_raw_list = []
#     for neg_filename in neg_filename_list:
#         df_neg_raw = pd.read_csv(neg_filename, sep=" ", dtype="float64", header=None)
#         df_neg_raw_list.append(df_neg_raw)
#     df_neg_total = pd.concat(df_neg_raw_list, axis=0)
#     for pos_filename in pos_filename_list:
#         df_pos_raw = pd.read_csv(pos_filename, sep=" ", dtype="float64", header=None)
#         df_pos_raw_list.append(df_pos_raw)
#     df_pos_total = pd.concat(df_pos_raw_list, axis=0)
#     df_neg_part = pd.DataFrame(
#         df_neg_total[feature_index_list].values.reshape((int(len(df_neg_total) / 20)), len(feature_index_list) * 20))
#     df_pos_part = pd.DataFrame(
#         df_pos_total[feature_index_list].values.reshape((int(len(df_pos_total) / 20)), len(feature_index_list) * 20))
#     df_neg_part.insert(len(df_neg_part.columns), "label", 0)
#     df_pos_part.insert(len(df_pos_part.columns), "label", 1)
#     df_total = pd.concat([df_neg_part, df_pos_part], axis=0)
#     return df_total


def load_rnn_data():
    global TIME_STEP, INPUT_DIV, OUTPUT_DIV, reuse
    sybol_str = "_" + "reuse-" + str(reuse)
    pos_filename_list_out, neg_filename_list_out = il.get_filename_list()
    sensor_data_df = il.get_data(neg_filename_list_out, pos_filename_list_out, sybol_str, reuse)
    X = sensor_data_df.ix[:, :TIME_STEP * INPUT_DIV]
    y = sensor_data_df.ix[:, -1:]  # 标签已经转换成0，1了
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    sc = sklearn.preprocessing.StandardScaler()
    sc.fit(X_train)
    X_train_norm = sc.transform(X_train)
    X_test_norm = sc.transform(X_test)
    # X_train_norm = X_train.values
    # X_test_norm = X_test.values
    X_train_seq = X_train_norm.reshape((X_train_norm.shape[0], TIME_STEP, INPUT_DIV))
    X_test_seq = X_test_norm.reshape((X_test_norm.shape[0], TIME_STEP, INPUT_DIV))
    y_train_seq = y_train.values.reshape((y_train.shape[0], OUTPUT_DIV))
    y_test_seq = y_test.values.reshape((y_test.shape[0], OUTPUT_DIV))
    y_train_seq = to_categorical(y_train_seq, num_classes=2)
    y_test_seq = to_categorical(y_test_seq, num_classes=2)
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq


class SensorLSTM:
    def __init__(self):
        self.model = None

    def load_data(self):
        return load_rnn_data()

    def train(self, epochs=50):
        print('building model ...')
        self.model = SensorLSTM.build_model()

        print('loading data ...')
        text_train, rate_train, text_test, rate_test = self.load_data()

        print('training model ...')
        history = self.model.fit(text_train, rate_train, batch_size=BATCH_SIZE, epochs=epochs, validation_split=0.33)
        # history = self.model.fit(text_train, rate_train, batch_size=BATCH_SIZE, epochs=epochs,
        #                          validation_data=(text_test, rate_test))
        self.model.save(model_path)
        score = self.model.evaluate(text_test, rate_test)
        print(score)
        return history

    def load_trained_model(self, path):
        model = SensorLSTM.build_model()
        model.load_weights(path)
        return model

    def predict_3d_array(self, predict_x):
        if self.model == None:
            self.model = self.load_trained_model(model_path)
        return self.model.predict_classes(predict_x, 100)

    def predict_sensor(self, sc_model, predict_x_list, sample_size=1, feature_size=len(feature_index_list)):
        predict_x_list = predict_x_list[:TIME_STEP * sample_size]
        predict_x = np.array(predict_x_list).reshape((sample_size, TIME_STEP * feature_size))
        predict_x_norm = sc_model.transform(predict_x)
        predict_x_norm = predict_x_norm.reshape((sample_size, TIME_STEP, feature_size))
        predict_y = self.predict_3d_array(predict_x_norm)
        return predict_y

    @staticmethod
    def build_model():
        model = Sequential()
        model.add(Bidirectional(LSTM(128), input_shape=(TIME_STEP, INPUT_DIV)))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile('RMSprop', 'categorical_crossentropy', metrics=['accuracy'])
        return model


def learning_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    lstm = SensorLSTM()
    history = lstm.train(epoc_times)
    learning_curve(history)
    X_train_seq, y_train_seq, X_test_seq, y_test_seq = load_rnn_data()
    print(lstm.predict_3d_array(X_train_seq[0:2, :, :]))
