import pandas as pd
import numpy as np
import sys
import os
sys.path.append('../src/')
from tensorflow import keras

# title: Deep Neural Networks for Securing IoT Enabled Vehicular Ad-Hoc Networks
# years: 2021
# authors: Tejasvi Alladi & co
def CNN_LSTM_1(input_sequence_shape):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(20, 3, activation='relu', input_shape=input_sequence_shape))
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.Dense((20,2), activation='sigmoid'))

    model.compile(loss='mean_absolute_error', optimizer="adam", metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score()])

    return model

# title: Securing the Internet of Vehicles: A Deep Learning-Based Classification Framework
# years: 2021
# authors: Tejasvi Alladi & co
def CNN_LSTM_2(input_sequence_shape):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(1024, 3, activation='relu', input_shape=input_sequence_shape))
    model.add(keras.layers.Conv1D(512, 3, activation='relu'))
    model.add(keras.layers.MaxPooling1D())
    model.add(keras.layers.LSTM(512, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))

    adam = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score()])

    return model

def three_LSTM(input_sequence_shape):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))

    adam = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score()])

    return model

def four_LSTM(input_sequence_shape):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))

    adam = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score()])

    return model

def five_LSTM(input_sequence_shape):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(256, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))

    adam = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score()])

    return model

# title: Artiﬁcial Intelligence (AI)-Empowered Intrusion Detection Architecture for the Internet of Vehicles
# years: 2021
# authors: Tejasvi Alladi & co

def LSTM_2(input_sequence_shape):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(512, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(512, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(512, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(512, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))

    adam = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score()])

    return model

def CNN_LSTM_3(input_sequence_shape):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(1024, 8, strides=1, activation='relu', input_shape=input_sequence_shape))
    model.add(keras.layers.Conv1D(512, 8, strides=1, activation='relu'))
    model.add(keras.layers.MaxPooling1D())
    model.add(keras.layers.LSTM(512, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))

    adam = keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score()])

    return model

#TO DO CNN model
#TO DO MLP model