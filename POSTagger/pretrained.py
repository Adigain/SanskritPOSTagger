import fasttext
import numpy as np

def get_fasttext_weights(tokenizer, embedding_dim=300, model_path='POSTagger/data/cc.sa.300.bin'):
    ft_model = fasttext.load_model(model_path)
    vocab_size = len(tokenizer.word_index) + 1
    embedding_weights = np.zeros((vocab_size, embedding_dim))

    for word, index in tokenizer.word_index.items():
        try:
            embedding_weights[index] = ft_model.get_word_vector(word)
        except KeyError:
            pass

    return embedding_weights
