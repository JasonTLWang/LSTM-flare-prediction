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
import csv
import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
try :
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('turn off loggins is not supported')

# data_mean = [9.26048596e+02, 2.58664488e-01, 1.06633638e+02, 5.11085855e-01,
#              6.24011676e+02, 2.04194132e+23, 2.33909547e+01, 1.90406673e+13,
#              3.32675181e+00, 1.32545323e+22, 5.96746073e+03, 1.85633869e-02,
#              2.19502276e-06, 4.75747286e+12]
# data_std = [1.08065295e+03, 7.01180865e-01, 2.02918470e+02, 1.41554237e+00,
#             6.55378540e+02, 3.35568384e+23, 1.57301570e+01, 2.08734375e+13,
#             7.34233192e+00, 1.44636883e+22, 4.10534455e+03, 1.20567656e-01,
#             1.74265529e-05, 7.53463926e+12]
# data_max = [1.42231700e+04, 9.61858226e+00, 4.05506900e+03, 1.80000000e+01,
#             7.21147559e+03, 5.36934000e+24, 7.66870000e+01, 4.81340300e+14,
#             8.70000000e+01, 2.07016000e+23, 2.80475700e+04, 1.05571716e+01,
#             9.30000000e-04, 1.08546200e+14]
# data_min = [2.720000e-01, 0.000000e+00, 1.000000e-03, 0.000000e+00, 6.860500e-02,
#             3.117358e+19, 1.800000e-02, 9.951892e+09, 0.000000e+00, 3.300907e+18,
#             5.640048e+02, 0.000000e+00, 0.000000e+00, 7.529357e+07]


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
        label = row[1][0]
        if flare_label == 'M' and label == 'X':
            label = 'M'
        if flare_label == 'M' and (label == 'B' or label == 'C'):
            label = 'N'
        has_zero_record = False
        # if at least one of the 25 physical feature values is missing, then discard it.
        if flare_label == 'M':
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
            for k in range(23, 26):
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
                if flare_label == 'M':
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
                    for k in range(23, 26):
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


if __name__ == '__main__':
    flare_label = sys.argv[1]
    train_again = int(sys.argv[2])
    filepath = './'
    n_features = 0
    if flare_label == 'M':
        n_features = 22
    start_feature = 5
    mask_value = 0
    series_len = 10
    epochs = 7
    batch_size = 256
    nclass = 2
    result_file = './output.csv'

    if train_again == 1:
        # Train
        X_train_data, y_train_data = load_data(datafile=filepath + 'normalized_training.csv',
                                               flare_label=flare_label, series_len=series_len,
                                               start_feature=start_feature, n_features=n_features,
                                               mask_value=mask_value)

        X_train = np.array(X_train_data)
        y_train = np.array(y_train_data)
        y_train_tr = data_transform(y_train)

        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y_train), y_train)
        class_weight_ = {0: class_weights[0], 1: class_weights[1]}
        # print(class_weight_)

        model = lstm(nclass, n_features, series_len)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(X_train, y_train_tr,
                            epochs=epochs, batch_size=batch_size,
                            verbose=False, shuffle=True, class_weight=class_weight_)
        model.save('./model.h5')
    else:
        model = load_model('./model.h5')

    # Test
    X_test_data, y_test_data = load_data(datafile=filepath + 'normalized_testing.csv',
                                         flare_label=flare_label, series_len=series_len,
                                         start_feature=start_feature, n_features=n_features,
                                         mask_value=mask_value)
    X_test = np.array(X_test_data)
    y_test = np.array(y_test_data)
    y_test_tr = data_transform(y_test)

    classes = model.predict(X_test, batch_size=batch_size, verbose=0, steps=None)

    with open(result_file, 'w', encoding='UTF-8') as result_csv:
        w = csv.writer(result_csv)
        with open(filepath + 'normalized_testing.csv', encoding='UTF-8') as data_csv:
            reader = csv.reader(data_csv)
            i = -1
            for line in reader:
                if i == -1:
                    line.insert(0, 'Predicted Label')
                else:
                    if classes[i][0] >= 0.6:
                        line.insert(0, 'Positive')
                    else:
                        line.insert(0, 'Negative')
                i += 1
                w.writerow(line)


