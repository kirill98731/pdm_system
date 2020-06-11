import pandas as pd
from sklearn import preprocessing
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load


def gen_history(data, cycles_number, need_cols):
    df = data[need_cols]
    df_all = df
    for i in cycles_number:
        df_all = df_all.join(df.shift(i), rsuffix='_'+str(i)+'_back')
    return df_all.dropna()


# function to reshape features into (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]


# function to generate labels
def gen_rul(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


def preprocessing_data(data, window, new=True):
    train_without_const = pd.read_csv(data)
    vars_to_drop = ['Unit_Number', 'Time_in_cycles']
    train_only_sensor = train_without_const.drop(vars_to_drop, axis=1)
    X = train_only_sensor.drop(['RUL'], axis=1)
    # y = train_only_sensor['RUL']
    if new:
        scaler = preprocessing.StandardScaler()
        scaler.fit(X)
        dump(scaler, 'Pdm_system/static/scaler.joblib')
    else:
        scaler = load('Pdm_system/static/scaler.joblib')
        scaler.fit(X)
        dump(scaler, 'Pdm_system/static/scaler.joblib')
    X_scale = scaler.transform(X)
    RUL_to_norm = np.array(train_without_const['RUL']).reshape(-1, 1)
    if new:
        scaler_min_max = preprocessing.MinMaxScaler()
        scaler_min_max.fit(RUL_to_norm)
        dump(scaler_min_max, 'Pdm_system/static/scaler_min_max.joblib')
    else:
        scaler_min_max = load('Pdm_system/static/scaler_min_max.joblib')
        scaler_min_max.fit(RUL_to_norm)
        dump(scaler_min_max, 'Pdm_system/static/scaler_min_max.joblib')
    RUL_scale = scaler_min_max.transform(RUL_to_norm)
    train_df = pd.DataFrame(X_scale)
    train_df.columns = X.columns
    train_df['Unit_Number'] = train_without_const['Unit_Number']
    train_df['Time_in_cycles'] = train_without_const['Time_in_cycles']
    train_df['RUL'] = RUL_scale
    train_df['RUL_1'] = train_without_const['RUL']
    last = 31 - window
    lasr_sensor = [10, 20, last]
    transformed_data_aa = pd.DataFrame()
    for unit_num in train_df['Unit_Number'].unique():
        temp = gen_history(train_df[train_df['Unit_Number'] == unit_num], lasr_sensor, X.columns)
        transformed_data_aa = pd.concat([transformed_data_aa, temp])

    o_unit = pd.DataFrame(np.copy(transformed_data_aa), columns=transformed_data_aa.columns,
                          index=transformed_data_aa.index)
    o_unit['Unit_Number'] = train_df.loc[transformed_data_aa.index]['Unit_Number']
    o_unit['RUL'] = train_df.loc[transformed_data_aa.index]['RUL']

    # generator for the sequences
    seq_gen_aa = (list(gen_sequence(o_unit[o_unit['Unit_Number'] == unit_num], window, transformed_data_aa.columns))
                  for unit_num in o_unit['Unit_Number'].unique())
    # generate sequences and convert to numpy array
    seq_array_aa = np.concatenate(list(seq_gen_aa)).astype(np.float32)
    # generator for the sequences
    rul_gen_aa = (list(gen_rul(o_unit[o_unit['Unit_Number'] == unit_num], window, ['RUL']))
                  for unit_num in o_unit['Unit_Number'].unique())
    rul_array_aa = np.concatenate(list(rul_gen_aa)).astype(np.float32)
    return seq_array_aa, rul_array_aa


def create_new_model(data):
    window = 3
    seq_array_aa, rul_array_aa = preprocessing_data(data, window)
    scaler_min_max = load('Pdm_system/static/scaler_min_max.joblib')
    model = Sequential()
    model.add(LSTM(
        input_shape=(window, seq_array_aa.shape[2]),
        units=100,
        return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(
        units=30,
        return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(seq_array_aa, rul_array_aa, epochs=15, batch_size=200, validation_split=0.05, verbose=1)
    y_pred = model.predict(seq_array_aa, verbose=1, batch_size=500)
    y_pred = scaler_min_max.inverse_transform(y_pred)
    y_true = scaler_min_max.inverse_transform(np.array(rul_array_aa).reshape(-1, 1))
    # print("Mean Squared Error: ", mean_squared_error(y_true, y_pred))
    # print("Mean Absolute Error: ", mean_absolute_error(y_true, y_pred))
    # print("r-squared: ", r2_score(y_true, y_pred))
    model.save('Pdm_system/static/LSTM_3.h5')
    return mean_squared_error(y_true, y_pred), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)


def update_model(data):
    window = 3
    seq_array_aa, rul_array_aa = preprocessing_data(data, window, False)
    model = keras.models.load_model('Pdm_system/static/LSTM_3.h5')
    scaler_min_max = load('Pdm_system/static/scaler_min_max.joblib')
    model.fit(seq_array_aa, rul_array_aa, epochs=15, batch_size=200, validation_split=0.05, verbose=1)
    y_pred = model.predict(seq_array_aa, verbose=1, batch_size=500)
    y_pred = scaler_min_max.inverse_transform(y_pred)
    y_true = scaler_min_max.inverse_transform(np.array(rul_array_aa).reshape(-1, 1))
    # print("Mean Squared Error: ", mean_squared_error(y_true, y_pred))
    # print("Mean Absolute Error: ", mean_absolute_error(y_true, y_pred))
    # print("r-squared: ", r2_score(y_true, y_pred))
    model.save('Pdm_system/static/LSTM_3.h5')
    return mean_squared_error(y_true, y_pred), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)


def add_iot(data):
    print(data)


def get_prediction(data):
    window = 3
    sequence_cols = ['Sensor_2', 'Sensor_3', 'Sensor_4', 'Sensor_7', 'Sensor_8', 'Sensor_9',
     'Sensor_11', 'Sensor_12', 'Sensor_13', 'Sensor_14', 'Sensor_15',
     'Sensor_17', 'Sensor_20', 'Sensor_21']
    test = pd.read_csv(data)
    scaler = load('Pdm_system/static/scaler.joblib')
    scaler_min_max = load('Pdm_system/static/scaler_min_max.joblib')
    test_df = pd.DataFrame(scaler.transform(test[sequence_cols]), columns=sequence_cols)
    test_df['Unit_Number'] = test['Unit_Number']
    test_df['Time_in_cycles'] = test['Time_in_cycles']
    test_df.head()
    last = 31 - window
    lasr_sensor = [10, 20, last]
    transformed_data_aa = pd.DataFrame()
    for unit_num in test_df['Unit_Number'].unique():
        temp = gen_history(test_df[test_df['Unit_Number'] == unit_num], lasr_sensor, sequence_cols)
        transformed_data_aa = pd.concat([transformed_data_aa, temp])
    o_unit_test = pd.DataFrame(np.copy(transformed_data_aa), columns=transformed_data_aa.columns,
                               index=transformed_data_aa.index)
    o_unit_test['Unit_Number'] = test_df.loc[transformed_data_aa.index]['Unit_Number']

    seq_array_test_last_aa = [
        o_unit_test[o_unit_test['Unit_Number'] == unit_num][transformed_data_aa.columns].values[-window:]
        for unit_num in o_unit_test['Unit_Number'].unique()]

    seq_array_test_last_aa = np.asarray(seq_array_test_last_aa).astype(np.float32)

    model = keras.models.load_model('Pdm_system/static/LSTM_3.h5')
    y_pred = model.predict(seq_array_test_last_aa, verbose=1, batch_size=200)
    y_pred = scaler_min_max.inverse_transform(y_pred)
    return y_pred
