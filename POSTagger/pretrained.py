import fasttext
import numpy as np
import os
import pickle

def get_fasttext_weights(tokenizer, embedding_dim=300, model_path='POSTagger/data/cc.sa.300.bin', cache_dir='POSTagger/data/'):
    embedding_file = os.path.join(cache_dir, 'embedding_matrix.npy')
    vocab_file = os.path.join(cache_dir, 'vocab.pkl')

    if os.path.exists(embedding_file) and os.path.exists(vocab_file):
        embedding_weights = np.load(embedding_file)
        with open(vocab_file, 'rb') as f:
            word_index = pickle.load(f)
        tokenizer.word_index = word_index
        return embedding_weights

    ft_model = fasttext.load_model(model_path)
    vocab_size = len(tokenizer.word_index) + 1
    embedding_weights = np.zeros((vocab_size, embedding_dim))

    for word, index in tokenizer.word_index.items():
        embedding_weights[index] = ft_model.get_word_vector(word)

    np.save(embedding_file, embedding_weights)
    with open(vocab_file, 'wb') as f:
        pickle.dump(tokenizer.word_index, f)

    return embedding_weights
