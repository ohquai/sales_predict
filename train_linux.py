# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from scipy import interp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, classification_report
import xgboost as xgb
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, LSTM, Embedding, Flatten, Activation
# from keras.layers import Bidirectional
from keras.optimizers import Adam, Adadelta, Nadam
from keras.models import Model
np.random.seed(2018)
# [339]	validation_0-auc:0.930062	validation_1-auc:0.860385
# [357]	validation_0-auc:0.929215	validation_1-auc:0.86141

class Sales_Model:
    def __init__(self):
        # self.path = "H:/df/zsyh/"
        self.path = '/code/riskdata/df/'
        self.k = 5
        self.val_split = 0.3

        # params for xgboost
        self.gridsearch_params = {
            'learning_rate': [0.02],
            'n_estimators': [700],
            'objective': ['binary:logistic'],
            'min_child_weight': [2],  # done
            'max_depth': [5],  # done
            'gamma': [0],  # done
            'max_delta_step': [2],  # done
            'subsample': [0.5],
            'colsample_bytree': [0.7],  # done
            'colsample_bylevel': [0.7],  # done
            'scale_pos_weight': [1],
        }

        # params for rnn
        self.BATCH_SIZE = 128 * 1
        self.epochs = 5
        self.MAX_LOG = 4399+1
        self.log_number = 40

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
        print(fpr)
        print(tpr)
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
        # plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.2f)' % roc_auc)
        # plt.show()

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

    def get_auc(self, y_true, y_pred_prob):
        fpr, tpr, theadhold = roc_curve(y_true, y_pred_prob)
        return auc(fpr, tpr)

    def get_propre_split(self, rnn_result, xgb_result, y_val):
        max_auc = 0
        best_rnn_prop = 0
        for rnn_prop in np.arange(0, 1, 0.001):
            rnn_part = map(lambda x: x * rnn_prop, rnn_result)
            xgb_part = map(lambda x: x * (1-rnn_prop), xgb_result)
            # new_result = [a+b for a,b in (rnn_part, xgb_part)]
            new_result = list(map(lambda x, y: x + y, list(rnn_part), list(xgb_part)))
            auc = self.get_auc(y_val, new_result)
            if auc > max_auc:
                max_auc = auc
                best_rnn_prop = rnn_prop
            print("{0} : {1}".format(rnn_prop, auc))
        return best_rnn_prop, max_auc

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

    def fillwithlist(self, data):
        print("replace with list")
        for col in ['EVT_L1', 'EVT_L2', 'EVT_L3']:
            list_replace = []
            for element in data[col]:
                if type(element) == list:
                    list_replace.append(element)
                else:
                    list_replace.append(['4399'])
                    # list_replace.append(-1)
            data[col] = list_replace
        print("finish")
        return data

    def new_rnn_model(self, X_train, lr=0.001, decay=0.0):
        # Inputs
        print([X_train["user_log1"].shape[1]])
        user_log1 = Input(shape=[X_train["user_log1"].shape[1]], name="user_log1")
        user_log2 = Input(shape=[X_train["user_log2"].shape[1]], name="user_log2")
        user_log3 = Input(shape=[X_train["user_log3"].shape[1]], name="user_log3")
        f_col_original = Input(shape=[X_train["col_original"].shape[1]], name="col_original")

        # Embeddings layers (adjust outputs to help model)
        emb_user_log1 = Embedding(self.MAX_LOG, self.log_number)(user_log1)
        emb_user_log2 = Embedding(self.MAX_LOG, self.log_number)(user_log2)
        emb_user_log3 = Embedding(self.MAX_LOG, self.log_number)(user_log3)

        # rnn layers (GRUs are faster than LSTMs and speed is important here)
        rnn_layer1 = GRU(12)(emb_user_log1)
        rnn_layer2 = GRU(12)(emb_user_log2)
        rnn_layer3 = GRU(12)(emb_user_log3)

        # main layers
        main_l = concatenate([rnn_layer1, rnn_layer2, rnn_layer3])
        main_l = Dropout(0.3)(Dense(256, activation='relu')(main_l))
        main_l = Dense(16, activation='relu')(main_l)
        main_l = Dense(2, activation='sigmoid')(main_l)

        main_l = concatenate([main_l, f_col_original])
        main_l = Dropout(0.3)(Dense(256, activation='relu')(main_l))
        main_l = Dense(64, activation='relu')(main_l)
        main_l = Dense(16, activation='relu')(main_l)

        # the output layer.
        output = Dense(2, activation="sigmoid")(main_l)

        model = Model([user_log1, user_log2, user_log3, f_col_original], output)
        print(model.summary())
        optimizer = Adam(lr=lr, decay=decay)
        # optimizer = Nadam(lr=lr)
        # (mean squared error loss function works as well as custom functions)
        # model.compile(loss='mse', optimizer=optimizer)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

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
        return X

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
        # train_agg.head(1000).to_csv(self.path + "df_list_sample.csv", encoding="utf8", index=False)

        log_trace1 = pd.DataFrame(data=test_log.groupby(by='USRID')['EVT_L1'].apply(f))
        log_trace2 = pd.DataFrame(data=test_log.groupby(by='USRID')['EVT_L2'].apply(f))
        log_trace3 = pd.DataFrame(data=test_log.groupby(by='USRID')['EVT_L3'].apply(f))
        test_agg = pd.merge(test_agg, log_trace1, how='left', left_index=True, right_index=True)
        test_agg = pd.merge(test_agg, log_trace2, how='left', left_index=True, right_index=True)
        test_agg = pd.merge(test_agg, log_trace3, how='left', left_index=True, right_index=True)
        test_agg = self.fillwithlist(test_agg)

        return train_agg, test_agg

    def data_prepare(self, data_train, data_test, split_index, algo):
        n_train = data_train.shape[0]
        data_valid = data_train.loc[split_index[int((1 - self.val_split) * n_train):], :]
        data_train = data_train.loc[split_index[: int((1 - self.val_split) * n_train)], :]

        x_train, y_train = data_train.drop(['FLAG', 'USRID'], axis=1), data_train['FLAG'].astype("float64")
        x_valid, y_valid = data_valid.drop(['FLAG', 'USRID'], axis=1), data_valid['FLAG'].astype("float64")
        x_test = data_test.drop(['USRID'], axis=1)

        if algo == 'rnn':
            return self.get_rnn_data(x_train), to_categorical(np.asarray(y_train)), self.get_rnn_data(x_valid), to_categorical(np.asarray(y_valid)), self.get_rnn_data(x_test)
        elif algo == 'xgb':
            return np.array(x_train.drop(['EVT_L1', 'EVT_L2', 'EVT_L3'], axis=1)), y_train \
                , np.array(x_valid.drop(['EVT_L1', 'EVT_L2', 'EVT_L3'], axis=1)), y_valid \
                , np.array(x_test.drop(['EVT_L1', 'EVT_L2', 'EVT_L3'], axis=1))

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

    def get_model_xgb(self, X_train, y_train, X_valid, y_valid):
        model, best_parameters_dict = self.xgb_fit(X_train, y_train, X_valid, y_valid, self.gridsearch_params)
        self.get_feature_importance(model, self.path + "feature_importance.csv")
        return model

    def get_model_rnn(self, X_train, y_train, X_valid, y_valid):
        # Calculate learning rate decay.
        exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
        steps = int(len(X_train['user_log1']) / self.BATCH_SIZE) * self.epochs
        lr_init, lr_fin = 0.005, 0.001
        lr_decay = exp_decay(lr_init, lr_fin, steps)

        # Create model and fit it with training dataset.
        rnn_model = self.new_rnn_model(X_train, lr=lr_init, decay=lr_decay)
        rnn_model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=1)
        return rnn_model

    def process(self):
        # 读取数据
        train_agg, train_log, train_flg, test_agg, test_log = self.read_data()

        # 增加特征维度
        train_log, test_log = self.log_feature_process(train_log, test_log)
        train_agg, test_agg = self.agg_plus_log_discret(train_agg, train_log, test_agg, test_log)
        train_agg = train_agg.fillna(-1)
        test_agg = test_agg.fillna(-1)

        # 计算每个用户行为轨迹，用list表示，用来做rnn
        train_agg, test_agg = self.agg_plus_log_continue(train_agg, train_log, test_agg, test_log)

        # 训练和标签关联
        data_train = pd.merge(train_agg, train_flg, how='inner', on='USRID')

        # 数据集的切分
        n_train = data_train.shape[0]
        split_index = np.random.permutation(n_train)
        X_train_xgb, y_train_xgb, X_valid_xgb, y_valid_xgb, X_test_xgb = self.data_prepare(data_train, test_agg, split_index, algo='xgb')
        # X_train_rnn, y_train_rnn, X_valid_rnn, y_valid_rnn, X_test_rnn = self.data_prepare(data_train, test_agg, split_index, algo='rnn')

        # 模型训练
        model_xgb = self.get_model_xgb(X_train_xgb, y_train_xgb, X_valid_xgb, y_valid_xgb)
        # model_rnn = self.get_model_rnn(X_train_rnn, y_train_rnn, X_valid_rnn, y_valid_rnn)

        # validation验证
        print("Evaluating the model on validation data...")
        valid_predict_xgb = model_xgb.predict_proba(X_valid_xgb)
        self.evaluate(y_valid_xgb, valid_predict_xgb[:, 1])
        auc = self.get_auc(y_valid_xgb, valid_predict_xgb[:, 1])
        # valid_predict_rnn = model_rnn.predict(X_valid_rnn, batch_size=self.BATCH_SIZE)
        # self.evaluate(y_valid_rnn[:, 1], valid_predict_rnn[:, 1])
        # print(" RMSLE error:", self.rmsle(y_valid_rnn[:, 1], valid_predict_rnn[:, 1]))

        # 结果预测
        result_xgb = model_xgb.predict_proba(X_test_xgb)
        # result_rnn = model_rnn.predict(X_test_rnn, batch_size=self.BATCH_SIZE, verbose=1)

        # best_rnn_prop, max_auc = self.get_propre_split(valid_predict_rnn[:, 1], valid_predict_xgb[:, 1], y_valid_xgb)
        # print("best auc is {0}, best_rnn_prop is {1}".format(max_auc, best_rnn_prop))

        # result = result_rnn[:, 1]
        result = result_xgb[:, 1]
        # result = list(map(lambda x, y: (best_rnn_prop*x + (1-best_rnn_prop)*y), list(result_rnn[:, 1]), list(result_xgb[:, 1])))

        # 存储结果
        # df_result = pd.DataFrame(data=result, columns=['RST'])
        # df_result['USRID'] = test_agg['USRID']
        # df_result = df_result[['USRID', 'RST']]
        # df_result.to_csv(self.path + "result.csv", encoding="utf8", sep="\t", index=False)

        return auc


if __name__ == "__main__":
    obj = Sales_Model()
    auc_score = obj.process()
    print(auc_score)
    # result = {}
    # for param in [[1], [2], [3], [4], [5]]:
    #     obj = Sales_Model()
    #     obj.gridsearch_params.update({'min_child_weight': param})
    #    auc_score = obj.process()
    #     result[param[0]] = auc_score
    # for key in result.keys():
    #     print("{}: {}".format(key, result[key]))
