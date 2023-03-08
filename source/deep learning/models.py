import pandas as pd
import numpy as np
import sys
import os
sys.path.append('../src/')
from tensorflow import keras

def dense_network(input_sequence_shape):
    simple_model = keras.Sequential(
        [
            layers.Dense(128, input_shape=(20, input_sequence_shape), activation="relu"),
            # layers.BatchNormalization(), # les batchNormalization fond baisser l'accuracy
            layers.Dense(256, activation="relu"),
            # layers.BatchNormalization(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="linear"),  # avec relu on perd un peu d'accuracy
        ]
    )
    return simple_model

def stacked_RNN(hidden_units=32, dense_units=1, input_sequence_shape=input_sequence_shape, activation=['relu', 'relu']):
    '''
    hidden_units : nombre de neurones dans la couche SimpleRNN
    dense_units : nombre de neurones dans la couche Dense
    activation : liste des deux fonctions d'activation
    '''
    
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=(20,input_sequence_shape), return_sequences=True, activation=activation[0]))
    model.add(SimpleRNN(32, activation=activation[0]))
    model.add(keras.layers.BatchNormalization())
    model.add(Dense(64, activation=activation[1]))
    model.add(Dense(units=dense_units, activation='sigmoid'))
    
    return model


def stacked_LSTM_small(input_sequence_shape):
    lstm_model = keras.Sequential([
    layers.Dense(32, input_shape=(20,input_sequence_shape), activation='relu'),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(128),
    #layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    ])
    return lstm_model

# Mix
def mix_rnn_lstm(input_sequence_shape):
    lstm_rnn_model = keras.Sequential([
    layers.Dense(32,input_shape=(20,input_sequence_shape), activation='relu'),
    layers.LSTM(128, return_sequences=True, activation='relu'), # return_sequences à True pour que la sortie soit de dimension 3
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.SimpleRNN(64,activation='relu', return_sequences=True),
    layers.SimpleRNN(32,activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    #layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
    ])
    return lstm_rnn_model

# Single GRU layer of 256 units
def simple_GRU(input_sequence_shape):
    simple_GRU = keras.models.Sequential()
    simple_GRU.add( keras.layers.Dense(32,input_shape=(20,input_sequence_shape), activation='relu'))
    simple_GRU.add( keras.layers.Dropout(0.2) )
    simple_GRU.add( keras.layers.BatchNormalization() )
    simple_GRU.add( keras.layers.GRU(256, return_sequences=False, activation='relu') )
    simple_GRU.add( keras.layers.Dropout(0.2) )
    simple_GRU.add( keras.layers.BatchNormalization() )
    # simple_GRU.add( keras.layers.Dense(64, activation='relu') )
    # simple_GRU.add( keras.layers.Dropout(0.2) )
    simple_GRU.add( keras.layers.Dense(1, activation='sigmoid') )
    return simple_GRU

# 3-stacked GRU model
def stacked_GRU(input_sequence_shape):
    stacked_GRU_model = keras.models.Sequential()
    stacked_GRU_model.add( keras.layers.Dense(32,input_shape=(20,input_sequence_shape), activation='relu'))
    stacked_GRU_model.add( keras.layers.Dropout(0.2) )
    stacked_GRU_model.add( keras.layers.BatchNormalization() )
    stacked_GRU_model.add( keras.layers.GRU(256, return_sequences=True, activation='relu') )
    stacked_GRU_model.add( keras.layers.GRU(256, return_sequences=True, activation='relu') )
    stacked_GRU_model.add( keras.layers.GRU(256, return_sequences=False, activation='relu') )
    stacked_GRU_model.add( keras.layers.Dropout(0.2) )
    stacked_GRU_model.add( keras.layers.BatchNormalization() )
    # stacked_GRU_model.add( keras.layers.Dense(64, activation='relu') )
    # stacked_GRU_model.add( keras.layers.Dropout(0.2) )
    stacked_GRU_model.add( keras.layers.Dense(1, activation='sigmoid') )
    return stacked_GRU_model


# 3-stacked LSTM model
def stacked_LSTM(input_sequence_shape):
    stacked_LSTM_model = keras.models.Sequential()
    stacked_LSTM_model.add( keras.layers.Dense(32,input_shape=(20,input_sequence_shape), activation='relu'))
    stacked_LSTM_model.add( keras.layers.Dropout(0.2) )
    stacked_LSTM_model.add( keras.layers.BatchNormalization() )
    stacked_LSTM_model.add( keras.layers.LSTM(256, return_sequences=True, activation='relu') )
    stacked_LSTM_model.add( keras.layers.LSTM(256, return_sequences=True, activation='relu') )
    stacked_LSTM_model.add( keras.layers.LSTM(256, return_sequences=False, activation='relu') )
    stacked_LSTM_model.add( keras.layers.Dropout(0.2) )
    stacked_LSTM_model.add( keras.layers.BatchNormalization() )
    # stacked_LSTM_model.add( keras.layers.Dense(64, activation='relu') )
    # stacked_LSTM_model.add( keras.layers.Dropout(0.2) )
    stacked_LSTM_model.add( keras.layers.Dense(1, activation='sigmoid') )
    return stacked_LSTM_model

# 3-stacked RNN model
def stacked_RNN(input_sequence_shape):
    stacked_RNN_model = keras.models.Sequential()
    stacked_RNN_model.add( keras.layers.Dense(32,input_shape=(20,input_sequence_shape), activation='relu'))
    stacked_RNN_model.add( keras.layers.Dropout(0.2) )
    stacked_RNN_model.add( keras.layers.BatchNormalization() )
    stacked_RNN_model.add( keras.layers.SimpleRNN(256, return_sequences=True, activation='relu') )
    stacked_RNN_model.add( keras.layers.SimpleRNN(256, return_sequences=True, activation='relu') )
    stacked_RNN_model.add( keras.layers.SimpleRNN(256, return_sequences=False, activation='relu') )
    stacked_RNN_model.add( keras.layers.Dropout(0.2) )
    stacked_RNN_model.add( keras.layers.BatchNormalization() )
    # stacked_RNN_model.add( keras.layers.Dense(64, activation='relu') )
    # stacked_RNN_model.add( keras.layers.Dropout(0.2) )
    stacked_RNN_model.add( keras.layers.Dense(1, activation='sigmoid') )
    return stacked_RNN_model