import pandas as pd
import numpy as np
import sys
import os
sys.path.append('../src/')
import dataset_prep

def dense_network():
    simple_model = keras.Sequential(
        [
            layers.Dense(128, input_shape=(20, 13), activation="relu"),
            # layers.BatchNormalization(), # les batchNormalization fond baisser l'accuracy
            layers.Dense(256, activation="relu"),
            # layers.BatchNormalization(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="linear"),  # avec relu on perd un peu d'accuracy
        ]
    )
