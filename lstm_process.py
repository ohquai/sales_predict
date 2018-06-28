import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, classification_report
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, LSTM, Embedding, Flatten, Activation
# from keras.layers import Bidirectional
from keras.optimizers import Adam, Adadelta, Nadam
from keras.models import Model, Sequential
np.random.seed(123)


MAX_NAME_SEQ = 10 #17
MAX_ITEM_DESC_SEQ = 75 #269
MAX_CATEGORY_SEQ = 8 #8


class RNNModel:
    def __init__(self):
        # self.path = "H:/df/zsyh/"
        self.path = "D:/DF/sales_predict/"
        self.k = 5
        self.val_split = 0.3
        # Set hyper parameters for the model.
        self.BATCH_SIZE = 128 * 1
        self.epochs = 5
        self.MAX_LOG = 4399+1
        self.log_number = 40

    def read_data(self):
        train_agg = pd.read_csv(self.path + "train_agg.csv", sep="\t", encoding='utf8')
        train_log = pd.read_csv(self.path + "train_log.csv", sep="\t", encoding='utf8')
        train_flg = pd.read_csv(self.path + "train_flg.csv", sep="\t", encoding='utf8')

        test_agg = pd.read_csv(self.path + "test_agg.csv", sep="\t", encoding='utf8')
        test_log = pd.read_csv(self.path + "test_log.csv", sep="\t", encoding='utf8')

        return train_agg, train_log, train_flg, test_agg, test_log

    def rmsle(self, Y, Y_pred):
        assert Y.shape == Y_pred.shape
        return np.sqrt(np.mean(np.square(Y_pred - Y)))

    def get_auc(self, y_true, y_pred_prob):
        fpr, tpr, theadhold = roc_curve(y_true, y_pred_prob)
        return auc(fpr, tpr)

    def evaluate(self, y_true, y_pred_prob):
        """
        evaluate model performance by calculate ks, auc, confusion matrix and find the best segmentation
        通过计算ks值，auc值以及混淆矩阵，来评估模型效果，并且寻找最佳切割点
        :param y_true: true target
        :param y_pred_prob: predict value
        :return:
        """
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        fpr, tpr, theadhold = roc_curve(y_true, y_pred_prob)
        # fpr, tpr, theadhold = roc_curve(y_true, y_pred_prob)
        size = len(tpr)
        max_value = 0
        index = 0
        for i in range(0, size):
            v = tpr[i] - fpr[i]
            if v > max_value:
                max_value = v
                index = i
        mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
        # mean_tpr[0] = 0.0  # 初始处为0
        roc_auc = auc(fpr, tpr)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
        plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.2f)' % roc_auc)
        plt.show()

        print('-------------- ks, auc --------------')
        print('ks: ' + str(max_value))
        print('auc: ' + str(auc(fpr, tpr)))
        print('threshold: ' + str(theadhold[index]))
        print('-------------------------------------')

        best_threshold = round(theadhold[index], 2)
        thres_pr_dict = {}
        threshold_recommended = 0.30
        best_precision = 0
        best_recall = 0
        best_pr = 0
        print('-------------- result details --------------')
        prob_thres = 0.01
        while prob_thres <= 1:
            # print('prob_thres: ' + str(prob_thres))
            print('prob_thres: ' + str(prob_thres))
            test_predict_new = []
            for prob in y_pred_prob:
                # if prob[1] > prob_thres:
                if prob > prob_thres:
                    test_predict_new.append(1)
                else:
                    test_predict_new.append(0)

            y_predict = np.array(test_predict_new)

            accuracy = accuracy_score(y_true, y_predict)
            precision, recall, f1, support = precision_recall_fscore_support(y_true, y_predict)
            matrix = confusion_matrix(y_true, y_predict)
            good_pass, bad_pass, good_deny, bad_deny = matrix[1][1], matrix[0][1], matrix[1][0], matrix[0][0]
            pass_ratio = float(good_pass + bad_pass) / (good_pass + bad_pass + good_deny + bad_deny)
            print('pass_ratio: ' + str(pass_ratio))
            print('accuracy: ' + str(accuracy))
            print('precision: ' + str(precision))
            print('recall: ' + str(recall))
            print('f1: ' + str(f1))
            print('confusion_matrix:')
            print(matrix)
            print(" ")

            thres_pr_dict[prob_thres] = pass_ratio
            if float('%.2f' % prob_thres) == float('%.2f' % best_threshold):
                best_precision = str(precision)
                best_recall = str(recall)
                best_pr = str(pass_ratio)
            prob_thres += 0.01

    def log_feature_process(self, train_log, test_log):
        train_log['EVT_L1'] = [a.split("-")[0] for a in train_log['EVT_LBL']]
        train_log['EVT_L2'] = [a.split("-")[1] for a in train_log['EVT_LBL']]
        train_log['EVT_L3'] = [a.split("-")[2] for a in train_log['EVT_LBL']]
        train_log['EVT_L1'] = train_log['EVT_L1'].astype("float64")
        train_log['EVT_L2'] = train_log['EVT_L2'].astype("float64")
        train_log['EVT_L3'] = train_log['EVT_L3'].astype("float64")

        test_log['EVT_L1'] = [a.split("-")[0] for a in test_log['EVT_LBL']]
        test_log['EVT_L2'] = [a.split("-")[1] for a in test_log['EVT_LBL']]
        test_log['EVT_L3'] = [a.split("-")[2] for a in test_log['EVT_LBL']]
        test_log['EVT_L1'] = test_log['EVT_L1'].astype("float64")
        test_log['EVT_L2'] = test_log['EVT_L2'].astype("float64")
        test_log['EVT_L3'] = test_log['EVT_L3'].astype("float64")

        return train_log, test_log

    def fillwithlist(self, data):
        print("replace with list")
        for col in ['EVT_L1', 'EVT_L2', 'EVT_L3']:
            list_replace = []
            for element in data[col]:
                if type(element) == list:
                    list_replace.append(element)
                else:
                    # list_replace.append(['4399'])
                    list_replace.append(-1)
            data[col] = list_replace
        print("finish")
        return data

    def agg_plus_log_discret(self, train_agg, train_log, test_agg, test_log):
        pivot_l1 = pd.pivot_table(train_log, index=["USRID"], columns=['EVT_L1'], values='EVT_LBL', aggfunc='count')
        pivot_l2 = pd.pivot_table(train_log, index=["USRID"], columns=['EVT_L2'], values='EVT_LBL', aggfunc='count')
        pivot_l3 = pd.pivot_table(train_log, index=["USRID"], columns=['EVT_L3'], values='EVT_LBL', aggfunc='count')
        pivot_l1['USRID'] = pivot_l1.index.tolist()
        pivot_l2['USRID'] = pivot_l2.index.tolist()
        pivot_l3['USRID'] = pivot_l3.index.tolist()
        train_agg = pd.merge(train_agg, pivot_l1, how='left', on='USRID')
        train_agg = pd.merge(train_agg, pivot_l2, how='left', on='USRID')
        train_agg = pd.merge(train_agg, pivot_l3, how='left', on='USRID')

        pivot_l1_t = pd.pivot_table(test_log, index=["USRID"], columns=['EVT_L1'], values='EVT_LBL', aggfunc='count')
        pivot_l2_t = pd.pivot_table(test_log, index=["USRID"], columns=['EVT_L2'], values='EVT_LBL', aggfunc='count')
        pivot_l3_t = pd.pivot_table(test_log, index=["USRID"], columns=['EVT_L3'], values='EVT_LBL', aggfunc='count')
        pivot_l1_t['USRID'] = pivot_l1_t.index.tolist()
        pivot_l2_t['USRID'] = pivot_l2_t.index.tolist()
        pivot_l3_t['USRID'] = pivot_l3_t.index.tolist()
        test_agg = pd.merge(test_agg, pivot_l1_t, how='left', on='USRID')
        test_agg = pd.merge(test_agg, pivot_l2_t, how='left', on='USRID')
        test_agg = pd.merge(test_agg, pivot_l3_t, how='left', on='USRID')

        for col in train_agg.columns:
            if col not in test_agg.columns:
                test_agg[col] = np.nan
        test_agg = test_agg[train_agg.columns.tolist()]
        return train_agg, test_agg

    def agg_plus_log_continue(self, train_agg, train_log, test_agg, test_log):
        f = lambda x: x.tolist()
        # train_log.groupby(by='USRID')['EVT_L1'].apply(f)
        log_trace1 = pd.DataFrame(data=train_log.groupby(by='USRID')['EVT_L1'].apply(f))
        log_trace2 = pd.DataFrame(data=train_log.groupby(by='USRID')['EVT_L2'].apply(f))
        log_trace3 = pd.DataFrame(data=train_log.groupby(by='USRID')['EVT_L3'].apply(f))
        train_agg = pd.merge(train_agg, log_trace1, how='left', left_index=True, right_index=True)
        train_agg = pd.merge(train_agg, log_trace2, how='left', left_index=True, right_index=True)
        train_agg = pd.merge(train_agg, log_trace3, how='left', left_index=True, right_index=True)
        train_agg = self.fillwithlist(train_agg)
        train_agg.head(1000).to_csv(self.path + "df_list_sample.csv", encoding="utf8", index=False)

        log_trace1 = pd.DataFrame(data=test_log.groupby(by='USRID')['EVT_L1'].apply(f))
        log_trace2 = pd.DataFrame(data=test_log.groupby(by='USRID')['EVT_L2'].apply(f))
        log_trace3 = pd.DataFrame(data=test_log.groupby(by='USRID')['EVT_L3'].apply(f))
        test_agg = pd.merge(test_agg, log_trace1, how='left', left_index=True, right_index=True)
        test_agg = pd.merge(test_agg, log_trace2, how='left', left_index=True, right_index=True)
        test_agg = pd.merge(test_agg, log_trace3, how='left', left_index=True, right_index=True)
        test_agg = self.fillwithlist(test_agg)

        # for col in train_agg.columns:
        #     if col not in test_agg.columns:
        #         test_agg[col] = np.nan
        # test_agg = test_agg[train_agg.columns.tolist()]
        return train_agg, test_agg

    def get_rnn_data(self, dataset):
        col_original = [col for col in dataset.columns if not str(col).startswith("EVT")]
        # col_original = [col for col in dataset.columns if col.startswith("V")]

        X = {
            'user_log1': pad_sequences(dataset['EVT_L1'], maxlen=self.log_number, truncating='pre'),
            'user_log2': pad_sequences(dataset['EVT_L2'], maxlen=self.log_number, truncating='pre'),
            'user_log3': pad_sequences(dataset['EVT_L3'], maxlen=self.log_number, truncating='pre'),
            'col_original': np.array(dataset[col_original]),
            # 'V2': np.array(dataset['V2']),
            # 'V3': np.array(dataset['V3']),
            # 'V4': np.array(dataset['V4']),
        }
        X['user_log1'] = np.array(X['user_log1']).reshape((X['user_log1'].shape[0], X['user_log1'].shape[1], 1))
        X['user_log2'] = np.array(X['user_log2']).reshape((X['user_log2'].shape[0], X['user_log2'].shape[1], 1))
        X['user_log3'] = np.array(X['user_log3']).reshape((X['user_log3'].shape[0], X['user_log3'].shape[1], 1))
        return X

    def data_prepare(self, data_train, data_test):
        n_train = data_train.shape[0]
        split_index = np.random.permutation(n_train)
        data_valid = data_train.loc[split_index[int((1 - self.val_split) * n_train):], :]
        data_train = data_train.loc[split_index[: int((1 - self.val_split) * n_train)], :]
        x_train, y_train = self.get_rnn_data(data_train.drop(['FLAG', 'USRID'], axis=1)), to_categorical(np.asarray(data_train['FLAG'].astype("float64")))
        x_valid, y_valid = self.get_rnn_data(data_valid.drop(['FLAG', 'USRID'], axis=1)), to_categorical(np.asarray(data_valid['FLAG'].astype("float64")))
        x_test = self.get_rnn_data(data_test.drop(['USRID'], axis=1))

        return x_train, y_train, x_valid, y_valid, x_test

    # user_log1 = np.array(user_log1).reshape((user_log1.shape[0], user_log1.shape[1], 1))
    # user_log2 = np.array(user_log2).reshape((user_log2.shape[0], user_log2.shape[1], 1))
    # user_log3 = np.array(user_log3).reshape((user_log3.shape[0], user_log3.shape[1], 1))

    def new_rnn_model(self, X_train, lr=0.001, decay=0.0):
        # Inputs
        print([X_train["user_log1"].shape[1]])
        user_log1 = Input(shape=[X_train["user_log1"].shape[1]], name="user_log1")
        user_log2 = Input(shape=[X_train["user_log2"].shape[1]], name="user_log2")
        user_log3 = Input(shape=[X_train["user_log3"].shape[1]], name="user_log3")

        # Embeddings layers (adjust outputs to help model)
        emb_user_log1 = Embedding(self.MAX_LOG, self.log_number)(user_log1)
        emb_user_log2 = Embedding(self.MAX_LOG, self.log_number)(user_log2)
        emb_user_log3 = Embedding(self.MAX_LOG, self.log_number)(user_log3)

        # rnn layers (GRUs are faster than LSTMs and speed is important here)
        rnn_layer1 = GRU(12)(emb_user_log1)
        rnn_layer2 = GRU(12)(emb_user_log2)
        rnn_layer3 = GRU(12)(emb_user_log3)
        # rnn_layer1 = GRU(12)(user_log1)
        # rnn_layer2 = GRU(12)(user_log2)
        # rnn_layer3 = GRU(12)(user_log3)

        # main layers
        main_l = concatenate([rnn_layer1, rnn_layer2, rnn_layer3])
        main_l = Dropout(0.3)(Dense(256, activation='relu')(main_l))
        main_l = Dense(16, activation='relu')(main_l)
        main_l = Dense(2, activation='sigmoid')(main_l)

        # the output layer.
        output = Dense(2, activation="sigmoid")(main_l)

        model = Model([user_log1, user_log2, user_log3], output)
        print(model.summary())
        # optimizer = Adam(lr=lr, decay=decay)
        optimizer = Nadam(lr=lr)
        # (mean squared error loss function works as well as custom functions)
        # model.compile(loss='mse', optimizer=optimizer)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def new_rnn_model_sequence(self, X_train, lr=0.001, decay=0.0):
        # Inputs
        model_gru_1 = Sequential()
        # model_gru_1.add(GRU(50, input_shape=(40, 1), return_sequences=False))
        model_gru_1.add(GRU(12, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
        # model_gru_2 = Sequential()
        # model_gru_2.add(GRU(50, input_shape=(40, 1), return_sequences=False))
        # model_gru_3 = Sequential()
        # model_gru_3.add(GRU(50, input_shape=(40, 1), return_sequences=False))

        model_gru_1.add(Dense(128))
        model_gru_1.add(Activation('relu'))
        model_gru_1.add(Dense(16))
        model_gru_1.add(Activation('relu'))
        model_gru_1.add(Dense(2, activation='sigmoid'))
        print(model_gru_1.summary())
        optimizer = Nadam(lr=lr)
        model_gru_1.compile(loss='binary_crossentropy', optimizer=optimizer)

        return model_gru_1

    def process(self):
        # 读取数据
        train_agg, train_log, train_flg, test_agg, test_log = self.read_data()

        # 计算每个用户每个操作的数量
        train_log, test_log = self.log_feature_process(train_log, test_log)
        train_agg, test_agg = self.agg_plus_log_discret(train_agg, train_log, test_agg, test_log)
        train_agg = train_agg.fillna(-1)
        test_agg = test_agg.fillna(-1)

        train_agg, test_agg = self.agg_plus_log_continue(train_agg, train_log, test_agg, test_log)

        # train_agg = train_agg[train_agg['EVT_L1'] != ['4399']]
        # test_agg = test_agg[train_agg['EVT_L1'] != ['4399']]
        train_agg = train_agg[train_agg['EVT_L1'] != -1]
        test_agg = test_agg[test_agg['EVT_L1'] != -1]

        data_train = pd.merge(train_agg, train_flg, how='inner', on='USRID')
        data_train.head(1000).to_csv(self.path + "sample_with_flag.csv", encoding="utf8", index=False)
        X_train, y_train, X_valid, y_valid, X_test = self.data_prepare(data_train, test_agg)
        print(X_train['user_log1'].shape)
        print(X_train['user_log1'])

        # Calculate learning rate decay.
        exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
        steps = int(len(X_train['user_log1']) / self.BATCH_SIZE) * self.epochs
        lr_init, lr_fin = 0.005, 0.001
        lr_decay = exp_decay(lr_init, lr_fin, steps)

        # Create model and fit it with training dataset.
        print(X_train['user_log3'].shape[1])
        print(X_train['user_log3'].shape[2])
        print(X_train['user_log3'])
        rnn_model = self.new_rnn_model_sequence(X_train['user_log3'], lr=lr_init, decay=lr_decay)
        rnn_model.fit(X_train['user_log3'], y_train, epochs=self.epochs, batch_size=self.BATCH_SIZE, validation_data=(X_valid['user_log3'], y_valid), verbose=1)

        print("Evaluating the model on validation data...")
        Y_dev_preds_rnn = rnn_model.predict(X_valid['user_log3'], batch_size=self.BATCH_SIZE)
        self.evaluate(y_valid[:, 1], Y_dev_preds_rnn[:, 1])
        auc_score = self.get_auc(y_valid[:, 1], Y_dev_preds_rnn[:, 1])

        # print(" RMSLE error:", self.rmsle(y_valid[:, 1], Y_dev_preds_rnn[:, 1]))
        # print(Y_dev_preds_rnn)

        result = rnn_model.predict(X_test['user_log3'], batch_size=self.BATCH_SIZE, verbose=1)
        # result = np.expm1(result)
        print(result)
        # df_result = pd.DataFrame(data=result[:, 1], columns=['RST'])
        df_result = pd.DataFrame(data=result[:, 1], columns=['RST'])
        df_result['USRID'] = test_agg['USRID']
        df_result = df_result[['USRID', 'RST']]
        df_result.to_csv(self.path + "result.csv", encoding="utf8", sep="\t", index=False)
        df_result.to_csv(self.path + "result_view.csv", encoding="utf8", index=False)

        print("auc score :{0}".format(auc_score))
        return Y_dev_preds_rnn[:, 1], result[:, 1]

    def main_process(self):
        val_result, test_result = self.process()
        return val_result, test_result


if __name__ == "__main__":
    obj = RNNModel()
    val_result, test_result = obj.process()


# -------------- ks, auc --------------
# ks: 0.50564600297
# auc: 0.835747391692
# threshold: 0.0494424