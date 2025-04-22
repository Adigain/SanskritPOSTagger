
from POSTagger import (
    load_base_data,
    get_fasttext_weights,
    build_model,
    train_and_prepare_model,
    preprocess_sample_input,
    preprocess_sample_ground_truth,
    predict_tags,
    evaluate_predictions,
    plot_comparison
)

from POSTagger.config import LAYERS, MAX_SEQ_LENGTH
from POSTagger.input import SENTENCES, TAGSET

print("\n[1] Loading Base Tagset training data...")
X, Y, word_tokenizer, tag_tokenizer = load_base_data()

print("\n[2] Loading FastText Embeddings...")
embedding_matrix = get_fasttext_weights(word_tokenizer)

for i in LAYERS:
    print(f"\n[3] Building {i.upper()} model...")
model = build_model(input_dim=embedding_matrix.shape[0],
                    output_dim=embedding_matrix.shape[1],
                    input_len=MAX_SEQ_LENGTH,
                    num_classes=len(tag_tokenizer.word_index) + 1,
                    embedding_weights=embedding_matrix,
                    )

print("\n[4] Training the model...")
model, history = train_and_prepare_model(model, X, Y)

print("\n[5] Plotting training history...")
plot_comparison(history)

sample_sentences = SENTENCES
print("\n[6] Preprocessing sample input and running predictions...")
X_sample = preprocess_sample_input(sample_sentences, word_tokenizer)
predicted_tags = predict_tags(model, X_sample, tag_tokenizer)

#Optional
ground_truth = TAGSET 
ground_truth = preprocess_sample_ground_truth(ground_truth)
print("\n[7] Evaluating predictions against ground truth...")
evaluate_predictions(ground_truth, predicted_tags)

print("\n All steps completed successfully.")
