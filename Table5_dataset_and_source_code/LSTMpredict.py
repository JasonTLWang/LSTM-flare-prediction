# =========================================================================
#   (c) Copyright 2019
#   All rights reserved
#   Programs written by Hao Liu
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from keras.models import *
from keras.layers import *
from sklearn.metrics import confusion_matrix
import sys


def load_data(datafile, flare_label, series_len, start_feature, n_features, mask_value):
    df = pd.read_csv(datafile)
    df_values = df.values
    X = []
    y = []
    tmp = []
    for k in range(start_feature, start_feature + n_features):
        tmp.append(mask_value)
    for idx in range(0, len(df_values)):
        each_series_data = []
        row = df_values[idx]
        if flare_label == 'M5' and row[1][0] == 'M' and float(row[1][1:]) >= 5.0:
            label = 'X'
        else:
            label = row[1][0]
        if flare_label == 'M' and label == 'X':
            label = 'M'
        if flare_label == 'C' and (label == 'X' or label == 'M'):
            label = 'C'
        if flare_label == 'B' and (label == 'X' or label == 'M' or label == 'C'):
            label = 'B'
        if flare_label == 'M5' and (label == 'M' or label == 'C' or label == 'B'):
            label = 'N'
        if flare_label == 'M' and (label == 'B' or label == 'C'):
            label = 'N'
        if flare_label == 'C' and label == 'B':
            label = 'N'
        has_zero_record = False
        # if at least one of the 25 physical feature values is missing, then discard it.
        if flare_label == 'C':
            if float(row[5]) == 0.0:
                has_zero_record = True
            if float(row[7]) == 0.0:
                has_zero_record = True
            for k in range(9, 13):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(14, 16):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(18, 21):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            if float(row[22]) == 0.0:
                has_zero_record = True
            for k in range(24, 33):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(38, 42):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
        elif flare_label == 'M':
            for k in range(5, 10):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(13, 16):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            if float(row[19]) == 0.0:
                has_zero_record = True
            if float(row[21]) == 0.0:
                has_zero_record = True
            for k in range(23, 30):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(31, 33):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(34, 37):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(39, 41):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            if float(row[42]) == 0.0:
                has_zero_record = True
        elif flare_label == 'M5':
            for k in range(5, 12):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(19, 21):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(22, 31):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(32, 37):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(40, 42):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break

        if has_zero_record is False:
            cur_noaa_num = int(row[3])
            each_series_data.append(row[start_feature:start_feature + n_features].tolist())
            itr_idx = idx - 1
            while itr_idx >= 0 and len(each_series_data) < series_len:
                prev_row = df_values[itr_idx]
                prev_noaa_num = int(prev_row[3])
                if prev_noaa_num != cur_noaa_num:
                    break
                has_zero_record_tmp = False
                if flare_label == 'C':
                    if float(row[5]) == 0.0:
                        has_zero_record_tmp = True
                    if float(row[7]) == 0.0:
                        has_zero_record_tmp = True
                    for k in range(9, 13):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(14, 16):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(18, 21):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    if float(row[22]) == 0.0:
                        has_zero_record_tmp = True
                    for k in range(24, 33):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(38, 42):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                elif flare_label == 'M':
                    for k in range(5, 10):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(13, 16):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    if float(row[19]) == 0.0:
                        has_zero_record_tmp = True
                    if float(row[21]) == 0.0:
                        has_zero_record_tmp = True
                    for k in range(23, 30):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(31, 33):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(34, 37):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(39, 41):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    if float(row[42]) == 0.0:
                        has_zero_record_tmp = True
                elif flare_label == 'M5':
                    for k in range(5, 12):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(19, 21):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(22, 31):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(32, 37):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(40, 42):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break

                if len(each_series_data) < series_len and has_zero_record_tmp is True:
                    each_series_data.insert(0, tmp)

                if len(each_series_data) < series_len and has_zero_record_tmp is False:
                    each_series_data.insert(0, prev_row[start_feature:start_feature + n_features].tolist())
                itr_idx -= 1

            while len(each_series_data) > 0 and len(each_series_data) < series_len:
                each_series_data.insert(0, tmp)

            if len(each_series_data) > 0:
                X.append(np.array(each_series_data).reshape(series_len, n_features).tolist())
                y.append(label)
    X_arr = np.array(X)
    y_arr = np.array(y)
    print(X_arr.shape)
    return X_arr, y_arr

def partition_10_folds(X, y, num_of_fold):
    num = len(X)
    index = [i for i in range(num)]
    np.random.seed(123)
    np.random.shuffle(index)
    X_output = []
    y_output = []
    num_in_each_fold = round(num / num_of_fold)
    for i in range(num_of_fold):
        if i == (num_of_fold - 1):
            idx = index[num_in_each_fold * (num_of_fold - 1):]
        else:
            idx = index[num_in_each_fold * i : num_in_each_fold * (i + 1)]
        X_output.append(X[idx])
        y_output.append(y[idx])
    return X_output, y_output

def data_transform(data):
    encoder = LabelEncoder()
    encoder.fit(data)
    encoded_Y = encoder.transform(data)
    converteddata = np_utils.to_categorical(encoded_Y)
    return converteddata

def attention_3d_block(hidden_states, series_len):
    hidden_size = int(hidden_states.shape[2])
    hidden_states_t = Permute((2, 1), name='attention_input_t')(hidden_states)
    hidden_states_t = Reshape((hidden_size, series_len), name='attention_input_reshape')(hidden_states_t)
    score_first_part = Dense(series_len, use_bias=False, name='attention_score_vec')(hidden_states_t)
    score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
    h_t = Lambda(lambda x: x[:, :, -1], output_shape=(hidden_size, 1), name='last_hidden_state')(hidden_states_t)
    score = dot([score_first_part_t, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states_t, attention_weights], [2, 1], name='context_vector')
    context_vector = Reshape((hidden_size,))(context_vector)
    h_t = Reshape((hidden_size,))(h_t)
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(hidden_size, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector

def lstm(nclass, n_features, series_len):
    inputs = Input(shape=(series_len, n_features,))
    lstm_out = LSTM(10, return_sequences=True, dropout=0.5)(inputs)
    attention_mul = attention_3d_block(lstm_out, series_len)
    layer1_out = Dense(200, activation='relu')(attention_mul)
    layer2_out = Dense(500, activation='relu')(layer1_out)
    output = Dense(nclass, activation='softmax', activity_regularizer=regularizers.l2(0.0001))(layer2_out)
    model = Model(input=[inputs], output=output)
    return model

    # result_file = '../data/result.csv'
    # with open(result_file, 'w', encoding='UTF-8') as result_csv:
    #     w = csv.writer(result_csv)

if __name__ == '__main__':
    flare_label = sys.argv[1]
    filepath = './' + flare_label + '/'
    num_of_fold = 10
    n_features = 0
    if flare_label == 'M5':
        n_features = 20
    elif flare_label == 'M':
        n_features = 22
    elif flare_label == 'C':
        n_features = 14
    start_feature = 5
    mask_value = 0
    series_len = 10
    epochs = 7
    batch_size = 256
    nclass = 2
    thlistsize = 201
    thlist = np.linspace(0, 1, thlistsize)

    X_train_data, y_train_data = load_data(datafile=filepath + 'normalized_training.csv',
                                           flare_label=flare_label, series_len=series_len,
                                           start_feature=start_feature, n_features=n_features,
                                           mask_value=mask_value)
    X_train_fold, y_train_fold = partition_10_folds(X_train_data, y_train_data, num_of_fold)

    X_valid_data, y_valid_data = load_data(datafile=filepath + 'normalized_validation.csv',
                                           flare_label=flare_label, series_len=series_len,
                                           start_feature=start_feature, n_features=n_features,
                                           mask_value=mask_value)
    X_valid_fold, y_valid_fold = partition_10_folds(X_valid_data, y_valid_data, num_of_fold)

    X_test_data, y_test_data = load_data(datafile=filepath + 'normalized_testing.csv',
                                         flare_label=flare_label, series_len=series_len,
                                         start_feature=start_feature, n_features=n_features,
                                         mask_value=mask_value)
    X_test_fold, y_test_fold = partition_10_folds(X_test_data, y_test_data, num_of_fold)

    max_recall0 = np.zeros(thlistsize)
    max_precision0 = np.zeros(thlistsize)
    max_recall1 = np.zeros(thlistsize)
    max_precision1 = np.zeros(thlistsize)
    max_acc = np.zeros(thlistsize)
    max_tss = np.zeros(thlistsize)
    max_bacc = np.zeros(thlistsize)
    max_hss = np.zeros(thlistsize)
    recall0list = []
    recall1list = []
    precision0list = []
    precision1list = []
    acclist = []
    hsslist = []
    tsslist = []
    bacclist = []
    for ithlistsize in range(thlistsize):
        recall0list.append([])
        recall1list.append([])
        precision0list.append([])
        precision1list.append([])
        acclist.append([])
        hsslist.append([])
        tsslist.append([])
        bacclist.append([])
    fraction_of_positives_list = []
    mean_predicted_value_list = []
    fpr_list = []
    tpr_list = []

    for train_itr in range(num_of_fold):
        X_train = []
        y_train = []
        for j in range(num_of_fold):
            if j != train_itr:
                for k in range(len(X_train_fold[j])):
                    X_train.append(X_train_fold[j][k])
                    y_train.append(y_train_fold[j][k])

        for test_itr in range(num_of_fold):
            print('------------- ' + str(train_itr * num_of_fold + test_itr) + ' iteration----------------')
            X_valid = []
            y_valid = []
            X_test = []
            y_test = []
            for j in range(num_of_fold):
                if j != test_itr:
                    for k in range(len(X_valid_fold[j])):
                        X_valid.append(X_valid_fold[j][k])
                        y_valid.append(y_valid_fold[j][k])
                    for k in range(len(X_test_fold[j])):
                        X_test.append(X_test_fold[j][k])
                        y_test.append(y_test_fold[j][k])

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_valid = np.array(X_valid)
            y_valid = np.array(y_valid)
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            y_train_tr = data_transform(y_train)
            y_valid_tr = data_transform(y_valid)
            y_test_tr = data_transform(y_test)

            class_weights = class_weight.compute_class_weight('balanced',
                                                              np.unique(y_train),
                                                              y_train)
            class_weight_ = {0: class_weights[0], 1: class_weights[1]}

            model = lstm(nclass, n_features, series_len)
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            history = model.fit(X_train, y_train_tr,
                                epochs=epochs, batch_size=batch_size,
                                verbose=False, shuffle=True, class_weight=class_weight_)

            classes = model.predict(X_test, batch_size=batch_size, verbose=0, steps=None)

            for th in range(len(thlist)):
                idx = th
                thresh = thlist[th]
                cmat = None
                if flare_label == 'X':
                    cmat = confusion_matrix(
                        np.array([np.argmax(y_test_tr[i], axis=0) for i in range(y_test_tr.shape[0])]),
                        np.array([classes[i][0] < (1. - thresh) for i in range(classes.shape[0])]).reshape(-1, 1)
                    )
                else:
                    cmat = confusion_matrix(
                        np.array([np.argmax(y_test_tr[i], axis=0) for i in range(y_test_tr.shape[0])]),
                        np.array([classes[i][0] < thresh for i in range(classes.shape[0])]).reshape(-1, 1)
                    )
                N = np.sum(cmat)

                recall = np.zeros(nclass)
                precision = np.zeros(nclass)
                accuracy = np.zeros(nclass)
                bacc = np.zeros(nclass)
                tss = np.zeros(nclass)
                hss = np.zeros(nclass)
                tp = np.zeros(nclass)
                fn = np.zeros(nclass)
                fp = np.zeros(nclass)
                tn = np.zeros(nclass)
                for p in range(nclass):
                    tp[p] = cmat[p][p]
                    for q in range(nclass):
                        if q != p:
                            fn[p] += cmat[p][q]
                            fp[p] += cmat[q][p]
                    tn[p] = N - tp[p] - fn[p] - fp[p]

                    recall[p] = round(float(tp[p]) / float(tp[p] + fn[p] + 1e-6), 3)
                    precision[p] = round(float(tp[p]) / float(tp[p] + fp[p] + 1e-6), 3)
                    accuracy[p] = round(float(tp[p] + tn[p]) / float(N), 3)
                    bacc[p] = round(
                        0.5 * (float(tp[p]) / float(tp[p] + fn[p]) + float(tn[p]) / float(tn[p] + fp[p])), 3)
                    hss[p] = round(2 * float(tp[p] * tn[p] - fp[p] * fn[p])
                                   / float((tp[p] + fn[p]) * (fn[p] + tn[p])
                                           + (tp[p] + fp[p]) * (fp[p] + tn[p])), 3)
                    tss[p] = round((float(tp[p]) / float(tp[p] + fn[p] + 1e-6) - float(fp[p]) / float(
                        fp[p] + tn[p] + 1e-6)), 3)

                if tss[0] > max_tss[idx]:
                    max_tss[idx] = tss[0]
                    max_bacc[idx] = bacc[0]
                    max_hss[idx] = hss[0]
                    max_recall0[idx] = recall[0]
                    max_precision0[idx] = precision[0]
                    max_recall1[idx] = recall[1]
                    max_precision1[idx] = precision[1]
                    max_acc[idx] = accuracy[0]

                recall0list[idx].append(recall[0])
                recall1list[idx].append(recall[1])
                precision0list[idx].append(precision[0])
                precision1list[idx].append(precision[1])
                acclist[idx].append(accuracy[0])
                bacclist[idx].append(bacc[0])
                tsslist[idx].append(tss[0])
                hsslist[idx].append(hss[0])

    avg_recall0_list = []
    std_recall0_list = []
    avg_precision0_list = []
    std_precision0_list = []
    avg_acc_list = []
    std_acc_list = []
    avg_bacc_list = []
    std_bacc_list = []
    avg_hss_list = []
    std_hss_list = []
    avg_tss_list = []
    std_tss_list = []
    for th in range(len(thlist)):
        idx = th
        thresh = thlist[th]
        avg_recall0 = np.mean(np.array(recall0list[idx]))
        std_recall0 = np.std(np.array(recall0list[idx]))

        avg_recall1 = np.mean(np.array(recall1list[idx]))
        std_recall1 = np.std(np.array(recall1list[idx]))

        avg_precision0 = np.mean(np.array(precision0list[idx]))
        std_precision0 = np.std(np.array(precision0list[idx]))

        avg_precision1 = np.mean(np.array(precision1list[idx]))
        std_precision1 = np.std(np.array(precision1list[idx]))

        avg_acc = np.mean(np.array(acclist[idx]))
        std_acc = np.std(np.array(acclist[idx]))

        avg_bacc = np.mean(np.array(bacclist[idx]))
        std_bacc = np.std(np.array(bacclist[idx]))
        max_bacc = np.max(np.array(bacclist[idx]))
        min_bacc = np.min(np.array(bacclist[idx]))

        avg_hss = np.mean(np.array(hsslist[idx]))
        std_hss = np.std(np.array(hsslist[idx]))
        max_hss = np.max(np.array(hsslist[idx]))
        min_hss = np.min(np.array(hsslist[idx]))

        avg_tss = np.mean(np.array(tsslist[idx]))
        std_tss = np.std(np.array(tsslist[idx]))
        max_tss = np.max(np.array(tsslist[idx]))
        min_tss = np.min(np.array(tsslist[idx]))

        avg_recall0_list.append(round(avg_recall0, 3))
        std_recall0_list.append(round(std_recall0, 3))
        avg_precision0_list.append(round(avg_precision0, 3))
        std_precision0_list.append(round(std_precision0, 3))
        avg_acc_list.append(round(avg_acc, 3))
        std_acc_list.append(round(std_acc, 3))
        avg_bacc_list.append(round(avg_bacc, 3))
        std_bacc_list.append(round(std_bacc, 3))
        avg_hss_list.append(round(avg_hss, 3))
        std_hss_list.append(round(std_hss, 3))
        avg_tss_list.append(round(avg_tss, 3))
        std_tss_list.append(round(std_tss, 3))

    max_avg_tss = 0.0
    max_avg_hss = 0.0
    max_avg_recall0 = 0.0
    max_avg_tss_idx = -1
    for idx in range(thlistsize):
        if avg_tss_list[idx] > max_avg_tss:
            max_avg_tss = avg_tss_list[idx]
            max_avg_hss = avg_hss_list[idx]
            max_avg_recall0 = avg_recall0_list[idx]
            max_avg_tss_idx = idx
        elif avg_tss_list[idx] == max_avg_tss:
            if avg_hss_list[idx] > max_avg_hss:
                max_avg_tss = avg_tss_list[idx]
                max_avg_hss = avg_hss_list[idx]
                max_avg_recall0 = avg_recall0_list[idx]
                max_avg_tss_idx = idx
            elif avg_hss_list[idx] == max_avg_hss:
                if avg_recall0_list[idx] > max_avg_recall0:
                    max_avg_tss = avg_tss_list[idx]
                    max_avg_hss = avg_hss_list[idx]
                    max_avg_recall0 = avg_recall0_list[idx]
                    max_avg_tss_idx = idx

    f = open("./output.txt", "w")
    f.write('avg recall: ' + str(avg_recall0_list[max_avg_tss_idx]) + ' (' + str(std_recall0_list[max_avg_tss_idx]) + ')\n')
    f.write('avg precision: ' + str(avg_precision0_list[max_avg_tss_idx]) + ' (' + str(std_precision0_list[max_avg_tss_idx]) + ')\n')
    f.write('avg acc: ' + str(avg_acc_list[max_avg_tss_idx]) + ' (' + str(std_acc_list[max_avg_tss_idx]) + ')\n')
    f.write('avg bacc: ' + str(avg_bacc_list[max_avg_tss_idx]) + ' (' + str(std_bacc_list[max_avg_tss_idx]) + ')\n')
    f.write('avg hss: ' + str(avg_hss_list[max_avg_tss_idx]) + ' (' + str(std_hss_list[max_avg_tss_idx]) + ')\n')
    f.write('avg tss: ' + str(avg_tss_list[max_avg_tss_idx]) + ' (' + str(std_tss_list[max_avg_tss_idx]) + ')\n')
    f.close()
