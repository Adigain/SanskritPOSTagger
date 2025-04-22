
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

from .config import MAX_SEQ_LENGTH

def preprocess_sample_input(sample_sentences, tokenizer, max_len=MAX_SEQ_LENGTH):
    for idx in range(len(sample_sentences)):
        if len(sample_sentences[idx]) < max_len:
            sample_sentences[idx] += [''] * (max_len - len(sample_sentences[idx]))
        else:
            sample_sentences[idx] = sample_sentences[idx][:max_len]
    sequences = tokenizer.texts_to_sequences(sample_sentences)
    padded = pad_sequences(sequences, maxlen=max_len, padding="pre", truncating="post")
    return padded

def preprocess_sample_ground_truth(ground_truth, max_len=MAX_SEQ_LENGTH):
    for idx in range(len(ground_truth)):
        if len(ground_truth[idx]) < max_len:
            ground_truth[idx] += [''] * (max_len - len(ground_truth[idx]))
        else:
            ground_truth[idx] = ground_truth[idx][:max_len]
    sequences = ground_truth
    return sequences


def load_base_data():
    df_X = pd.read_csv("POSTagger/data/JNU_dataset_sentences.csv", header=None).replace(np.nan, '', regex=True)
    df_Y = pd.read_csv("POSTagger/data/JNU_dataset_labelupos.csv", header=None).replace(np.nan, '', regex=True)
    df_X.drop([0], axis=0, inplace=True)
    df_X.drop([0], axis=1, inplace=True)
    df_Y.drop([0], axis=0, inplace=True)
    df_Y.drop([0], axis=1, inplace=True)
    X, Y = df_X.values.tolist(), df_Y.values.tolist()
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(X)
    tag_tokenizer = Tokenizer()
    tag_tokenizer.fit_on_texts(Y)
    X_encoded = word_tokenizer.texts_to_sequences(X)
    Y_encoded = tag_tokenizer.texts_to_sequences(Y)
    X_padded = pad_sequences(X_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
    Y_padded = pad_sequences(Y_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
    return X_padded, Y_padded, word_tokenizer, tag_tokenizer


