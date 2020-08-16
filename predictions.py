import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from IPython.display import Image


def convert_data_types(converted):
    """
      Converts data types to the relevant type

      Args:
        converted (dataframe): numpy array with data values.

      Returns:
        converted (dataframe): standardized and normalized numerical variables
    """
    converted['Date'] = pd.to_datetime(converted['Date'])
    converted['InvoiceID'] = converted['InvoiceID'].astype('category')
    converted['ProductCode'] = converted['ProductCode'].astype('category')
    converted['Description'] = converted['Description'].astype('category')
    return converted


def drop_col(df, col_list):
    df.drop(col_list, axis=1, inplace=True)
    return df


def generate_ft_transform(ft):  # Takes into consideration discounts and returns
    """
      Groups and averages dataframe by day.
      Standardizes numerical feature values of data

      Args:
        ft (dataframe): numpy array with data values.

      Returns:
        ft_npy (numpy array): standardized and normalized numerical variables
    """
    ft = drop_col(ft, col_list=['InvoiceID', 'ProductCode'])
    ft['IsReturns'] = ft['Volume'].apply(lambda x: 1 if x < 0 else 0)
    ft['IsDiscount'] = ft['UnitPrice'].apply(lambda x: 1 if x == 0 else 0)
    ft['IsWeekDay'] = ft.apply(lambda x: x["Date"].weekday(), axis=1)
    ft['IsWeekDay'] = (ft['IsWeekDay'] < 5).astype(int)
    ft = ft.groupby(pd.Grouper(key='Date', freq='D')).agg(
        VolumeMean=pd.NamedAgg(column='Volume', aggfunc=np.mean),
        UnitPriceMean=pd.NamedAgg(column='UnitPrice', aggfunc=np.mean),
        TotReturns=pd.NamedAgg(column='IsReturns', aggfunc=np.sum),
        TotDiscount=pd.NamedAgg(column='IsDiscount', aggfunc=np.sum),
        TotWeekDays=pd.NamedAgg(column='IsWeekDay', aggfunc=np.sum)
    )
    ft = ft.fillna(0)
    ft = ft.values

    global mean_data
    mean_data = ft[:, 0].mean(axis=0)  # Used for converting predictions in the end
    global std_data
    std_data = ft[:, 0].std(axis=0)

    global data_mean
    data_mean = ft.mean(axis=0)  # Not selecting the categorical features
    global data_std
    data_std = ft.std(axis=0)

    ft_npy = (ft - data_mean) / data_std

    return ft_npy


# Window for Training Set. Data Pipeline optimized to run on TPU
def windowed_dataset_train(series, window_size, batch_size, shuffle_buffer, step_forecast):
    """
    Standardizes and normalizes our numerical data in
    our numpy array to remove the presence of outliers

    Args:
      series (numpy array): numpy array with data values.
      window_size (integer): value for sliding window size.
      batch_size (integer): value to batch time windows.
      shuffle_buffer (integer): value to shuffle data from buffer.
      step_forecast (integer): value to forecast how many days ahead.

    Returns:
      ds (tensor): tensorflow tensor with sliding batched window and labels forecast
    """
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + step_forecast, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + step_forecast)).cache()
    ds = ds.map(lambda w: (w[:-step_forecast], [w[i:i + step_forecast, 0] for i in range(1, window_size + 1)]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return ds


# Window for Validation Set
def windowed_dataset_valid(series, window_size, batch_size, shuffle_buffer, step_forecast):
    """
    Standardizes and normalizes our numerical data in
    our numpy array to remove the presence of outliers

    Args:
      series (numpy array): numpy array with data values.
      window_size (integer): value for sliding window size.
      batch_size (integer): value to batch time windows.
      shuffle_buffer (integer): value to shuffle data from buffer.
      step_forecast (integer): value to forecast how many days ahead.

    Returns:
      ds (tensor): tensorflow tensor with sliding batched window and labels forecast
    """
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + step_forecast, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + step_forecast)).cache()
    ds = ds.map(lambda w: (w[:-step_forecast], [w[i:i + step_forecast, 0] for i in range(1, window_size + 1)]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def generate_predict_ft(ft):  # Takes into consideration discounts and returns
    """
      Groups and averages dataframe by day.
      Standardizes numerical feature values of data

      Args:
        ft (dataframe): numpy array with data values.

      Returns:
        ft_npy (numpy array): standardized and normalized numerical variables
    """
    ft = drop_col(ft, col_list=['InvoiceID', 'ProductCode'])
    ft['IsReturns'] = ft['Volume'].apply(lambda x: 1 if x < 0 else 0)
    ft['IsDiscount'] = ft['UnitPrice'].apply(lambda x: 1 if x == 0 else 0)
    ft['IsWeekDay'] = ft.apply(lambda x: x["Date"].weekday(), axis=1)
    ft['IsWeekDay'] = (ft['IsWeekDay'] < 5).astype(int)
    ft = ft.groupby(pd.Grouper(key='Date', freq='D')).agg(
        VolumeMean=pd.NamedAgg(column='Volume', aggfunc=np.mean),
        UnitPriceMean=pd.NamedAgg(column='UnitPrice', aggfunc=np.mean),
        TotReturns=pd.NamedAgg(column='IsReturns', aggfunc=np.sum),
        TotDiscount=pd.NamedAgg(column='IsDiscount', aggfunc=np.sum),
        TotWeekDays=pd.NamedAgg(column='IsWeekDay', aggfunc=np.sum)
    )
    ft = ft.fillna(0)
    ft = ft.values

    ft_npy = (ft - data_mean) / data_std

    return ft_npy


def predict_transform(df_to_predict):
    # Might have to create a read csv to pass from command line
    convert_data_types(df_to_predict)
    fts = df_to_predict.copy()
    fts = drop_col(fts, col_list=['Description'])
    data_set = generate_predict_ft(fts)
    predict_set = windowed_dataset_valid(data_set, 44, 2, 60, 7)
    return predict_set


def main():
    data = pd.read_csv('/content/IsaziChallenge/data/DS - case study 1 - add material - sales_volumes.csv', index_col=0)
    data = convert_data_types(data)

    featuress = data.copy()
    featuress = drop_col(featuress, col_list=['Description'])

    datasett = generate_ft_transform(featuress)

    tf.random.set_seed(0)

    train_sett = windowed_dataset_train(datasett[:124, :], 44, 2, 60, 7)
    valid_sett = windowed_dataset_valid(datasett[124:178, :], 44, 2, 60, 7)

    modell = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(7, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, input_shape=[None, 5]),
    ])
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    modell.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])

    checkpoint_path = "/content/IsaziChallenge/models/checkpoint_path/cp.ckpt"
    modell.load_weights(checkpoint_path)  # Loads model weights from checkpoints

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", required=True,
                        help="Pass file path to csv file to predict")
    args = parser.parse_args()

    data_prediction = data = pd.read_csv(args.filepath, index_col=0)

    prediction_set = predict_transform(data_prediction)  # Create an argument parser to parse prediction dataframe file through command line
    npyz = (modell.predict(prediction_set) * std_data) + mean_data
