# -*- coding: utf-8 -*-
"""
FeatureEngineeringCommon中包含了常用的特征处理、特征工程需要的方法
FeatureEngineeringSpecific中包含了这批特征中一些特殊处理，其中许多是指定列名的，适用性不广
Created on 20171130
@author: ChengTeng

更新20171205：
1、不使用del_indicated_feature，使用pandas自带的drop配合inplace=True实现内存节约。
2、优化了feature_unify， one_hot_specific，且feature_unify的特征统一为Int格式。
3、del_feature中，加入了指定的als特征项。
"""

from __future__ import division
import sys

sys.path.append("..")
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, StandardScaler, label_binarize
from sklearn.linear_model import LassoCV, Lasso
from statsmodels.stats.outliers_influence import variance_inflation_factor
# import feature_conf.jsqb_feature_names_lr as lrf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from scipy.stats import pearsonr
import gc
import os


class FeatureEngineeringCommon(object):
    """
    common feature engineering methods
    """

    def __init__(self):
        pass

    @staticmethod
    def load_columns_json(json_path):
        f = open(json_path, 'r')
        a = f.read()
        columns = eval(a)
        f.close()

        return columns

    @staticmethod
    def update_columns_json(dic, json_path):
        if os.path.exists(json_path):
            os.remove(json_path)
        f = open(json_path, 'w')
        f.write(str(dic))
        f.close()

    # @staticmethod
    # def load_columns_txt(txt_path):
    #     """
    #     :param txt_path: 读取列名json文件的路径
    #     :return:
    #     """
    #     with open(txt_path, 'r') as f:
    #         column_str = f.read()
    #     columns = column_str.replace("u'", "").replace("'", "").replace("[", "").replace("]", "").replace("\"",
    #                                                                                                       "").replace(
    #         " ", "").split(",")
    #     f.close()
    #     return columns

    @staticmethod
    def load_columns_txt(txt_path):
        """
        :param txt_path: 读取列名json文件的路径
        :return:
        """
        column_str = ""
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not len(line):
                    continue
                column_str += line
        columns = column_str.replace("'", "").replace("[", "").replace("]", "").replace(" ", "")[:-1].split(",")
        f.close()
        return columns

    @staticmethod
    def update_columns_txt(columns, txt_path):
        """
        :param columns:
        :param txt_path: 存储列名json文件的路径
        :return:
        """
        if os.path.exists(txt_path):
            os.remove(txt_path)
        with open(txt_path, 'w') as f:
            f.write("[\n")
            for col in columns:
                f.write("'"+col+"',\n")
            f.write("]")
        f.close()

    @staticmethod
    def balance_bi_label(df, max_pro):
        """
        control the maximum degree of unbalance
        :param df: input dataframe
        :param max_pro: max proportion of the unbalanced data
        :return:
        """
        df1 = df[df.label == 1]
        df0 = df[df.label == 0]
        len1 = len(df1)
        len0 = len(df0)
        if len1 >= len0:
            df1 = shuffle(df1)
            df1 = df1.iloc[0:len0 * max_pro]
            df = df1.append(df0)
        else:
            df0 = shuffle(df0)
            df0 = df0.iloc[0:len1 * max_pro]
            df = df0.append(df1)

        return df

    @staticmethod
    def del_indicated_feature(df, cols):
        """
        delete useless columns, replace dataframe.drop to save memory
        :param df: input dataframe
        :param cols: list of columns need to be dropped
        :return:
        """
        col_remain = df.columns.tolist()
        for col in cols:
            col_remain.remove(col)
        df = df[col_remain]
        return df

    @staticmethod
    def get_feature_trend(df, part=0.5):
        """
        get the vary of means between latest data and earlier data to have a better recognition of data
        :param df: input dataframe
        :param part: the proportion of earlier data in whole data
        :return:
        """
        length = df.shape[0]

        early_data = df[:int(length * part)]
        last_data = df[int(length * part):]
        early_mean = early_data.describe().loc['mean']
        last_mean = last_data.describe().loc['mean']
        mean_trend = last_mean / early_mean

        df_desc = pd.DataFrame(mean_trend.reshape(1, len(early_mean)), columns=df.columns, index=['coef'])

        return df_desc

    @staticmethod
    def del_insignificant_column(df):
        """
        delete all-none columns or all-same columns
        :param df: input dataframe
        :return:
        """
        print("length of columns: {0}".format(len(df.columns)))
        del_col = []
        for c in df.columns:
            if len(pd.value_counts(df[c])) == 0:
                print("all blanc column %s" % c)
                del_col.append(c)
                continue
            if df.loc[pd.Series.isnull(df[c]), c].shape[0] == df[c].shape[0] or len(pd.value_counts(df[c])) == 1:
                print("delete all-none or all-same column %s" % c)
                del_col.append(c)
        # df.drop(del_col, axis=1, inplace=True)
        df = df[np.setdiff1d(df.columns.tolist(), del_col).tolist()]
        print("length of columns: {0}".format(len(df.columns)))
        return df

    @staticmethod
    def lasso_feature_selection(df, exclude_col, label, n_features=100, alpha=0.001, max_iter=10000):
        """
        select columns with Lasso which use l1 regularization
        :param df: input dataframe
        :param exclude_col: columns excluded from selecting
        :param label: label in dataframe
        :param n_features: number of features remaining
        :param alpha: Lasso parameter
        :param max_iter: Lasso parameter
        :return:
        """
        # df_x = df.drop(exclude_col, axis=1)
        df_x = df[np.setdiff1d(df.columns.tolist(), exclude_col).tolist()]
        df_y = df[label]

        sc = StandardScaler()
        train_x = sc.fit_transform(df_x)
        clf = Lasso(max_iter=max_iter, alpha=alpha)
        clf.fit(train_x, df_y)
        print("feature columns are {0}".format(df_x.columns.tolist()))
        print("feature coefficients are {0}".format(clf.coef_.tolist()))

        dict_coef = dict(zip(df_x.columns.tolist(), clf.coef_.tolist()))
        dict_sorted = sorted(dict_coef.items(), key=lambda x: x[1], reverse=True)
        col_important = [dict_sorted[:n_features][i][0] for i in range(n_features)]
        col_important.extend(exclude_col)

        col_unimportant = np.setdiff1d(df.columns, col_important).tolist()
        print("unimportant features deleted {0}".format(col_unimportant))

        df = df[col_important]
        return df

    @staticmethod
    def feature_combine(df, col_headers, col_levels, mode='min', drop=True, dummy_nan=-1):
        """
        combine columns with same prefix and different suffix
        :param df: input dataframe
        :param col_headers: list of prefix
        :param col_levels: list of suffix
        :param mode: mode to combine the value, within ["min", "max", "sum"]
        :param drop: if drop the original columns
        :param dummy_nan: value that need to be considered as NaN
        :return:
        """

        def f_min(x):
            return np.nanmin(x)

        def f_max(x):
            return np.nanmax(x)

        def f_sum(x):
            return np.nansum(x)

        for sl_col_header in col_headers:
            cols = [sl_col_header + l for l in col_levels if sl_col_header + l in df.columns]
            print("columns to be combine: {0}".format(cols))
            for col in cols:
                df.loc[df[col] == dummy_nan, col] = np.NaN
            if len(cols) > 0:
                if mode == 'min':
                    df.loc[:, sl_col_header + 'ttl'] = df[cols].apply(f_min, axis=1)
                elif mode == 'max':
                    df.loc[:, sl_col_header + 'ttl'] = df[cols].apply(f_max, axis=1)
                elif mode == 'sum':
                    df.loc[:, sl_col_header + 'ttl'] = df[cols].apply(f_sum, axis=1)
                else:
                    raise Exception("invalid mode {0}, with valid mode in: 'min','max' and 'sum'".format(mode))
                df.loc[:, sl_col_header + 'ttl'] = df[sl_col_header + 'ttl'].fillna(dummy_nan)
                if drop:
                    df = df[np.setdiff1d(df.columns.tolist(), cols).tolist()]

        return df

    @staticmethod
    def one_hot_encoder(df, cols, label, if_drop=True):
        """
        one-hot encode for categorical features
        :param df: input dataframe
        :param cols: list of columns to be dealt
        :param label: label for label binarizer
        :param if_drop: if drop the original columns
        :return:
        """
        for col in cols:
            new_col_names = [col + '_' + str(l) for l in label]
            one_hot_sl_ = label_binarize(df[col].tolist(), classes=label)
            if 'one_hot_sl_nadarry' not in locals().keys():
                one_hot_sl_nadarry = one_hot_sl_
                one_hot_sl_col_name = new_col_names
            else:
                one_hot_sl_array = one_hot_sl_
                one_hot_sl_nadarry = np.hstack((one_hot_sl_nadarry, one_hot_sl_array))
                one_hot_sl_col_name.extend(new_col_names)
        if 'one_hot_sl_nadarry' in locals().keys():
            df_sl_ = pd.DataFrame(one_hot_sl_nadarry, columns=one_hot_sl_col_name, index=df.index)
            col_names = df.columns.tolist()
            new_cols = col_names + one_hot_sl_col_name
            df = pd.concat([df, df_sl_], axis=1, ignore_index=True)  # 此处index的问题高危
            df.columns = new_cols
        else:
            raise Exception("no column in {0} is valid".format(cols))

        if 'one_hot_sl_nadarry' in locals().keys() and if_drop:
            df = df[np.setdiff1d(df.columns.tolist(), cols).tolist()]
        del df_sl_, one_hot_sl_nadarry, one_hot_sl_col_name
        gc.collect()

        return df

    @staticmethod
    def calculate_vif_(df, thresh=10.0, mode='all', name_start=None, name_end=None, name_contain=None):
        """
        calculate VIF for all columns or indicated by prefix or suffix
        :param df: input dataframe
        :param thresh: threshold of VIF, 5/10 recommended
        :param mode: all columns or indicated columns
        :param name_start: list of prefix
        :param name_end: list of suffix
        :param name_contain: list of inner word
        :return:
        """
        if 'constant' not in df.columns:
            df['constant'] = 1
        mat = df.as_matrix(columns=None)
        col_name = df.columns.tolist()
        print("original columns are {0}".format(col_name))

        high_vif_cols = []
        for i in range(mat.shape[1]):
            if mode == 'all':
                one_vif = variance_inflation_factor(mat, i)
                if one_vif > thresh:
                    print("{0} has high vif value {1}".format(col_name[i], one_vif))
                    high_vif_cols.append(col_name[i])
            elif mode == 'startwith':
                for name in name_start:
                    if col_name[i].startswith(name):
                        one_vif = variance_inflation_factor(mat, i)
                        if one_vif > thresh:
                            print("{0} has high vif value {1}".format(col_name[i], one_vif))
                            high_vif_cols.append(col_name[i])
            elif mode == 'endwith':
                for name in name_end:
                    if col_name[i].endswith(name):
                        one_vif = variance_inflation_factor(mat, i)
                        if one_vif > thresh:
                            print("{0} has high vif value {1}".format(col_name[i], one_vif))
                            high_vif_cols.append(col_name[i])
            elif mode == 'in':
                for name in name_contain:
                    if name in col_name[i]:
                        one_vif = variance_inflation_factor(mat, i)
                        if one_vif > thresh:
                            print("{0} has high vif value {1}".format(col_name[i], one_vif))
                            high_vif_cols.append(col_name[i])
            else:
                raise Exception("invalid mode {0}, with mode in: 'all','startwith','endwith' and 'in'".format(mode))

        df = df[np.setdiff1d(df.columns.tolist(), high_vif_cols).tolist()]
        df = df[np.setdiff1d(df.columns.tolist(), ['constant']).tolist()]
        print('Remaining columns are {0}'.format(df.columns.tolist()))

        return df

    @staticmethod
    def del_col_pearsonr(df, label, correlation=True, corr_thres=0.001, pv=False, p_value=0.05):
        """
        drop columns with low correlation or high p-value with target
        :param df: input dataframe
        :param label: target column
        :param correlation: if use correlation filter
        :param corr_thres: the threshold of minimum correlation
        :param pv: if use p-value filter
        :param p_value: the threshold of maximum p-value
        :return:
        """
        # 删除相关度低和p-value高的特征
        col_del = []
        for col in df.columns:
            corr, pvalue = pearsonr(df[col], df[label])
            if correlation:
                if np.isnan(corr) == True or abs(corr) <= corr_thres:
                    print("low correlation column {0}, it's value is {1}".format(col, pearsonr(df[col], df[label])[0]))
                    col_del.append(col)
            if pv:
                if pvalue > p_value:
                    print("high p-value column {0}, it's value is {1}".format(col, pearsonr(df[col], df[label])[1]))
                    col_del.append(col)

        if len(col_del) > 0:
            # df = self.del_indicated_feature(df, col_del)
            df = df[np.setdiff1d(df.columns.tolist(), col_del).tolist()]

        return df

    @staticmethod
    def miss_value_padding(df):
        """
        缺失值填充：
        1、离散变量特征的缺失值比较少(<20%)时，将以众数填充；
        2、连续变量特征的缺失值比较少(<20%)时，将以中位数填充；
        3、缺失值较多时，将以-1替代
        :param df:
        :return:
        """
        exclude_cols = ['order_id', 'label', 'target']
        df_cols = df.columns.tolist()
        cols = np.setdiff1d(df_cols, exclude_cols)

        sample_count = df.shape[0]
        miss_replace = {}

        for col in cols:
            df.loc[df[col] == -1, col] = np.NaN
            # 特殊的几个字段，以-1替代空值，意义最合适
            if col.startswith("als_") or col.startswith("sl_") or col.startswith("report"):
                miss_replace[col] = -1
                df.loc[df[col].isnull(), col] = -1
                continue

            value_counts = pd.value_counts(df[col])  # 这一列每一种值以及对应的个数
            value_emun_count = value_counts.shape[0]  # 有多少种值
            count_no_missing = np.sum(value_counts)  # 某列有值的数量
            # 全空的列，以-1代替空值
            if count_no_missing == 0:
                miss_replace[col] = -1
                df.loc[df[col].isnull(), col] = -1
                continue

            if count_no_missing / sample_count > 0.8:  # >80%的数据均有值，认为缺失值问题不严重，需要对缺失值进行替代
                if value_emun_count > 6:  # 多于10种值，认为是连续变量
                    mean_value = np.median(df.loc[~df[col].isnull(), col])
                    miss_replace[col] = mean_value
                    df.loc[df[col].isnull(), col] = mean_value
                else:  # 少于10种值，认为是离散变量
                    mode_value = value_counts.index[0]  # 获取众数
                    miss_replace[col] = mode_value
                    df.loc[df[col].isnull(), col] = mode_value
            else:  # >20%的数据有缺失值，认为缺失值问题严重，将缺失值单独做考虑
                miss_replace[col] = -1
                df.loc[df[col].isnull(), col] = -1

        return df, miss_replace

    @staticmethod
    def feature_unify(df, col, replace_dict):
        cols = df.columns
        if col in cols:
            col_new = []
            for value in list(df[col]):
                if value in replace_dict.keys():
                    col_new.append(replace_dict[value])
                else:
                    col_new.append(1)
            df.loc[:, col] = col_new
        return df

    @staticmethod
    def output_result(result_path, content):
        print(content)
        with open(result_path, 'a') as f:
            f.write(content)
            f.write('\n')
        f.close()

    # def creat_f_train(self, data, flag, gbdtpath, estimators=40, depth=5):
    def creat_f_train(self, data, flag, model_path, estimators=40, depth=5):
        index = data.index
        gbdt = GradientBoostingClassifier(n_estimators=estimators, max_depth=depth)
        gbdt.fit(np.array(data), flag)
        joblib.dump(gbdt, model_path)
        f_value = gbdt.apply(data)[:, :, 0]

        name = []
        for i in range(0, estimators):
            name.append('gf_' + str(i))

        result = pd.DataFrame(f_value, columns=name, index=index)
        return result

    def creat_f(self, gbdt, data, estimators=40):
        index = data.index
        f_value = gbdt.apply(data)[:, :, 0]

        name = []
        for i in range(0, estimators):
            name.append('gf_' + str(i))
        # print name,f_value,estimators
        result = pd.DataFrame(f_value, columns=name, index=index)
        return result