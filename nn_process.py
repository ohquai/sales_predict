# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from scipy import interp
from sklearn.preprocessing import LabelEncoder
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras import initializers
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, classification_report
import xgboost as xgb
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, LSTM, Embedding, Flatten, Activation
# from keras.layers import Bidirectional
from keras.optimizers import Adam, Adadelta, Nadam, SGD
from keras.models import Model, Sequential
from keras import regularizers
np.random.seed(2018)
# [339]	validation_0-auc:0.930062	validation_1-auc:0.860385
# [357]	validation_0-auc:0.929215	validation_1-auc:0.86141


class Model:
    def __init__(self):
        # self.path = "H:/df/zsyh/"
        self.path = "D:/DF/sales_predict/"
        self.k = 5
        self.val_split = 0.3
        self.epochs = 8
        self.BATCH_SIZE = 32*2

    def read_data(self):
        train_agg = pd.read_csv(self.path + "train_agg.csv", sep="\t", encoding='utf8')
        train_log = pd.read_csv(self.path + "train_log.csv", sep="\t", encoding='utf8')
        train_flg = pd.read_csv(self.path + "train_flg.csv", sep="\t", encoding='utf8')

        test_agg = pd.read_csv(self.path + "test_agg.csv", sep="\t", encoding='utf8')
        test_log = pd.read_csv(self.path + "test_log.csv", sep="\t", encoding='utf8')

        return train_agg, train_log, train_flg, test_agg, test_log

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
        plt.xlabel('Specificity (假阳率)')
        plt.ylabel('Sensitivity (真阳率)')
        plt.title('ROC curve')
        plt.grid()
        plt.legend()
        # plt.show()

        # print('-------------- ks, auc --------------')
        # print('ks: ' + str(max_value))
        # print('auc: ' + str(auc(fpr, tpr)))
        # print('threshold: ' + str(theadhold[index]))

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

    def get_feature_importance(self, clf, path):
        # 存储特征重要性
        weight = []
        for i in list(clf.feature_importances_):
            weight.append(float(i))
        result = pd.DataFrame(np.array([clf._Booster.feature_names, weight]).T, columns=['name', 'weight'])
        result = result.sort_values(by=['weight'], ascending=False)
        result.to_csv(path, index=False)

    def find_best_parameters(self, mdl, param_tune, SKF, X, y):
        clf = GridSearchCV(mdl, param_tune, cv=SKF, scoring='roc_auc', verbose=2, refit=True)
        print("start grid search CV")
        clf.fit(X, y)

        # observer best parameters and result on validation set
        # best_parameters, score, _ = max(clf.cv_results_, key=lambda x: x[1])
        # print('Raw AUC score:', score)
        # for param_name in sorted(best_parameters.keys()):
        #     print("%s: %r" % (param_name, best_parameters[param_name]))
        best_parameters = clf.best_params_
        score = clf.best_score_
        print('Raw AUC score:', score)
        for param_name in sorted(best_parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        return best_parameters

    def xgb_fit(self, X_train, y_train, X_valid, y_valid, gridsearch_params):
        print('train model.')
        clf = xgb.XGBClassifier()

        # 合并train和validate数据集，进行gridsearch选择最优参数，之后再用validate去训练模型。
        _is_gridsearch = False
        params = {}
        for key in gridsearch_params.keys():
            if len(gridsearch_params[key]) > 1:
                _is_gridsearch = True
                print("we will use gridsearch")
            else:
                params[key] = gridsearch_params[key][0]

        if _is_gridsearch:
            X = np.append(X_train, X_valid, 0)
            y = np.append(y_train, y_valid, 0).tolist()
            # skf = StratifiedKFold(y=y, n_splits=3, shuffle=True, random_state=520)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=520)
            params = self.find_best_parameters(clf, gridsearch_params, skf, X, y)

        print(params)
        clf.set_params(**params)
        clf.fit(X_train, y_train, eval_metric=['auc'], eval_set=[(X_train, y_train), (X_valid, y_valid)],
                early_stopping_rounds=30)
        return clf, params

    def log_feature_process(self, train_log, test_log):
        train_log['EVT_L1'] = [a.split("-")[0] for a in train_log['EVT_LBL']]
        train_log['EVT_L2'] = [a.split("-")[1] for a in train_log['EVT_LBL']]
        train_log['EVT_L3'] = [a.split("-")[2] for a in train_log['EVT_LBL']]

        test_log['EVT_L1'] = [a.split("-")[0] for a in test_log['EVT_LBL']]
        test_log['EVT_L2'] = [a.split("-")[1] for a in test_log['EVT_LBL']]
        test_log['EVT_L3'] = [a.split("-")[2] for a in test_log['EVT_LBL']]

        return train_log, test_log

    def agg_plus_log_discret(self, train_agg, train_log, test_agg, test_log):
        col_origin = train_agg.columns.tolist()
        col_origin.extend(test_agg.columns.tolist())

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

        colname_update = {}
        for col in train_agg:
            if col not in col_origin:
                colname_update.update({col: 'f_'+col})
        for col in test_agg:
            if col not in col_origin:
                colname_update.update({col: 'f_'+col})
        train_agg = train_agg.rename(columns=colname_update)
        test_agg = test_agg.rename(columns=colname_update)

        for col in train_agg.columns:
            if col not in test_agg.columns:
                test_agg[col] = np.nan
        test_agg = test_agg[train_agg.columns.tolist()]
        return train_agg, test_agg

    def data_prepare(self, data_train, data_test):
        n_train = data_train.shape[0]
        split_index = np.random.permutation(n_train)
        data_valid = data_train.loc[split_index[int((1 - self.val_split) * n_train):], :]
        data_train = data_train.loc[split_index[: int((1 - self.val_split) * n_train)], :]
        x_train, y_train = data_train.drop(['FLAG', 'USRID'], axis=1).astype('float64'), to_categorical(np.asarray(data_train['FLAG'].astype("float64")))
        x_valid, y_valid = data_valid.drop(['FLAG', 'USRID'], axis=1).astype('float64'), to_categorical(np.asarray(data_valid['FLAG'].astype("float64")))
        x_test = data_test.drop(['USRID'], axis=1).astype('float64')

        # X_train = np.array(x_train)
        # X_valid = np.array(x_valid)
        # X_test = np.array(x_test)

        return x_train, y_train, x_valid, y_valid, x_test

    def new_nn_model(self, X_train, lr=0.001, decay=0.0):
        # Inputs
        model = Sequential()
        # model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        # model.add(Dense(128, input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.05)))
        model.add(Dense(128, input_shape=(X_train.shape[1],)))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(PReLU())
        # model.add(Dropout(0.2))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(16))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(PReLU())
        model.add(Dense(2, activation='sigmoid'))
        print(model.summary())
        optimizer = Nadam(lr=lr)
        # optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
        # model.compile(loss='mse', optimizer=optimizer)
        model.compile(loss='binary_crossentropy', optimizer=optimizer)

        # model.add(BatchNormalization())
        # model.add(PReLU())
        # model.add(Dense(10, kernel_initializer=initializers.Orthogonal()))
        return model

    def fit_model(self, X_train, y_train, X_valid, y_valid):
        params = {
            'max_depth': 5,
            'learning_rate': 0.02,
            'n_estimators': 300,
            'objective': 'binary:logistic'
        }
        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        mdl = xgb.XGBClassifier(seed=2018)
        mdl.set_params(**params)
        mdl.fit(X_train, y_train, early_stopping_rounds=30, eval_metric=['logloss'], eval_set=eval_set, verbose=True)
        return mdl

    def process(self):
        train_agg, train_log, train_flg, test_agg, test_log = self.read_data()
        train_log, test_log = self.log_feature_process(train_log, test_log)
        train_agg, test_agg = self.agg_plus_log_discret(train_agg, train_log, test_agg, test_log)
        # train_agg.head(1000).to_csv(self.path + "log_view.csv", encoding="utf8", index=False)
        data_train = pd.merge(train_agg, train_flg, how='inner', on='USRID')

        # data_train = data_train.fillna(-1)
        # test_agg = test_agg.fillna(-1)
        data_train = data_train.fillna(0)
        test_agg = test_agg.fillna(0)

        # col_remove = []
        # for col in data_train.columns:
        #     corr, pvalue = pearsonr(data_train[col], data_train['FLAG'])
        #     if 0.05 > corr > -0.05 and col.startswith('f_'):
        #         print("{0} deleted".format(col))
        #         col_remove.append(col)
        # data_train.drop(col_remove, axis=1, inplace=True)
        # test_agg.drop(col_remove, axis=1, inplace=True)

        X_train, y_train, X_valid, y_valid, X_test = self.data_prepare(data_train, test_agg)

        print(X_train)
        print(y_train)
        print(X_valid)
        print(y_valid)
        print(X_test)
        model = self.new_nn_model(X_train, lr=0.001, decay=0.0)
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=1)
        # model, best_parameters_dict = self.xgb_fit(X_train, y_train, X_valid, y_valid, self.gridsearch_params)
        # self.get_feature_importance(model, self.path + "feature_importance.csv")

        Y_dev_preds_rnn = model.predict_proba(X_valid)
        self.evaluate(y_valid[:, 1], Y_dev_preds_rnn[:, 1])
        auc_score = self.get_auc(y_valid[:, 1], Y_dev_preds_rnn[:, 1])

        result = model.predict_proba(X_test)

        df_result = pd.DataFrame(data=result[:, 1], columns=['RST'])
        df_result['USRID'] = test_agg['USRID']
        df_result = df_result[['USRID', 'RST']]
        df_result.to_csv(self.path + "result.csv", encoding="utf8", sep="\t", index=False)
        print("auc: {0}".format(auc_score))

        return auc_score


if __name__ == "__main__":
    obj = Model()
    obj.process()
    # auc_score_dict = {}
    # for sd in np.arange(990, 1020, 3):
    #     np.random.seed(sd)
    #     obj = Model()
    #     auc_score = obj.process()
    #     auc_score_dict.update({sd: auc_score})
    # print(auc_score_dict)
