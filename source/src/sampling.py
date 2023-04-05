import imblearn
import numpy as np
import pandas as pd


# oversampling
def oversampling(df):
    oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority')
    sample_columns, sample_label = oversample.fit_resample(df.drop(["label"], axis=1), df['label'])

    sampled_df = pd.concat([sample_columns, sample_label], axis=1)
    return sampled_df

# undersamplig 
def undersamplig(df):
    undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority')
    sample_columns, sample_label = undersample.fit_resample(df.drop(["label"], axis=1), df['label'])

    sampled_df = pd.concat([sample_columns, sample_label], axis=1)
    return sampled_df