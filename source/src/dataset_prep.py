import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def import_dataset(
    dataset_path,
    columns=[
        "distance",
        "distRealSR1",
        "pos_y_rec_f",
        "pos_y_rec",
        "pos_x_rec_f",
        "pos_x_rec",
        "nb_packets_sent",
        "label",
    ],
    data_type={
        "label":"int8"
    }
):
    # Import du csv
    data = pd.read_csv(
        dataset_path,
        usecols=columns,
        index_col=False,
        dtype=data_type
    )

    print("Nombre de lignes avant nettoyage : ", data.shape[0])

    # On remplace les données infinies par nan si elles existent
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop les lignes avec nan
    data.dropna(inplace=True)
    print("Nombre de lignes après nettoyage : ", data.shape[0])

    return data


def sample_dataset(dataset, sample_nb):
    return dataset.sample(sample_nb)


def data_preparation(dataset, test_size=0.1):
    # Transformation en array numpy
    X = np.array(dataset.drop(["label"], axis=1))
    y = np.array(dataset["label"])

    # TODO :  rajouter la ligne qui change tout ce qui n'est pas un 0 en 1
    for i in range(len(y)):
        if y[i] != 0:
            y[i] = 1

    # Séparation en données d'entrainement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test
