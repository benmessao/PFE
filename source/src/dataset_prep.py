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

def flat_sequence_creation(df):
    senders_sequences = []
    senders_label = []
    senders = np.unique(df["sender"].values)
    for sender in senders:
        # Données d'un seul sender rangée en fonction de l'heure d'envoi
        sender_data_sorted = df.loc[df['sender'] == sender].sort_values("sendTime")

        # On récupère la valeur du label pour ce sender
        """ On remplasse toute les valeur !=0 en 1 """
        if sender_data_sorted['label'].tolist()[0] != 0 :
            label=1
        else :
            label = sender_data_sorted['label'].tolist()[0]
        #On supprime les colonnes label et sender
        sender_data_sorted = sender_data_sorted.drop(["label","sender"], axis=1)
        
        #sequence_array = []

        length = sender_data_sorted.shape[0]
        slide = 10
        start = 0
        end = 20

        # On vérifie qu'il est possible de faire une séquence de taille 20
        while length > 20:
            # Extraction par tranche de 20 avec une inter de 10
            sequence = sender_data_sorted[start:end]

            # Labels correspondant
            #labels =  pd.Series.tolist(sequence["label"])

            # On transforme les 13 en 1, cette formule marche toujours si on met d'autres types d'attaques
            #labels[:] = [x if x == 0 else 1 for x in labels]

            # Attribution des tableaux numpy
            senders_sequences.append(np.array(sequence.values.tolist(), dtype=np.float32))
            senders_label.append(label)

            # Mise à jour des variables
            start += slide
            end += slide
            length -= 10
        
        print('Nombre de séquences : ',len(senders_sequences))
    return senders_sequences, senders_label

def data_preparation(df, sample=False, test_size=0.1):
   
    sorted_dataset = df.sort_values("sender")
    sequence_test, label_test = flat_sequence_creation(sorted_dataset)

    # Transformation en array numpy
    X = np.array(sequence_test)
    y = np.array(label_test, dtype=np.float32)

    # Réduire le temps de training en prenant les 100000 premiers éléments
    if sample:
        X = X[:100000]
        y = y[:100000]
    # Séparation en données d'entrainement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print("X_train : ", X_train.shape)
    print("y_train : ", y_train.shape)
    print("X_test : ", X_test.shape)
    print("y_test : ", y_test.shape)
    
    return X_train, X_test, y_train, y_test
