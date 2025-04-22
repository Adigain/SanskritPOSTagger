import numpy as np

def predict_tags(model, X_sample, tag_tokenizer):
    predictions = model.predict(X_sample)
    tag_indices = np.argmax(predictions, axis=-1)
    index2tag = {v: k for k, v in tag_tokenizer.word_index.items()}
    index2tag[0] = 'PAD'
    tagged_sentences = [[index2tag.get(tag, 'UNK') for tag in seq] for seq in tag_indices]
    print("Predicted POS tags:")
    for i, sent in enumerate(tagged_sentences):
        print(f"Sentence {i+1}:", sent)
    return tagged_sentences
