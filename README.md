# Sanskrit POS Tagger

A modular, exportable Python package for Sanskrit Part-of-Speech (POS) tagging using RNN-based architectures (SimpleRNN, LSTM, GRU, BiLSTM) with FastText embeddings and the JNU tagset. This library provides training, inference, and evaluation utilities. Users can easily modify the dataset, model architecture, and hyperparameters to suit their needs. The package supports both interactive Python usage and a command-line interface.

---

## 🚀 Features

- **Data Loading**: Seamlessly loads JNU-provided CSV datasets of sentences and POS tags.
- **Embeddings**: Integrates pre-trained FastText (`cc.sa.300.bin`) for robust Sanskrit word representations.
- **Model Builder**: Switch between SimpleRNN, LSTM, GRU, and BiLSTM architectures with configurable units, dropout, activations, and recurrent parameters.
- **Attention Layer**: Additive self-attention for enhanced context modeling.
- **Training Pipeline**: Utilities for train/validation splitting, categorical encoding, and model fitting with TensorFlow.
- **Inference**: Batch prediction of sentences with readable POS tag output.
- **Evaluation**: Token-level accuracy computation and training history visualization.
- **CLI Tool**: Entry point for end-to-end execution from the command line.

---

## ⚙️ Installation

1. **Clone the repository**
2. **Install dependencies**
3. **Install the package**
4. **Download Data**
- Place `JNU_dataset_sentences.csv` and `JNU_dataset_labelupos.csv` in `POSTagger/data/`
- Place `cc.sa.300.bin` (FastText model) in `POSTagger/data/`

---

## 🔧 Configuration

All hyperparameters (units, dropout, optimizer, learning rate, etc.) are in `sanskrit_pos_tagger/config.py`. Edit defaults or override at run-time via CLI flags.

## 🔍 Hyperparameters

- **MAX_SEQ_LENGTH**: Maximum length of input sequences (sentences).
- **EMBEDDING_SIZE**: Size of the word embedding vectors.
- **RNN_UNITS**: Number of units (neurons) in the RNN layer.
- **TRAINABLE_EMBEDDINGS**: Whether to train the embedding weights or keep them fixed.
- **LAYERS**: List of RNN layers to use in the model, starting from input towards output.
- **MASK_ZERO**: Whether to mask the zero-padding tokens in sequences.
- **ACTIVATION**: Activation function for the recurrent layers.
- **RECURRENT_ACTIVATION**: Activation function for the recurrent units (gates).
- **DROPOUT**: Dropout rate for regularization to prevent overfitting.
- **RECURRENT_DROPOUT**: Dropout rate applied specifically to the recurrent layer's connections.
- **RETURN_SEQUENCES**: Whether to return the output for each timestep or only the final timestep.
- **OPTIMIZER**: The optimizer used to train the model.
- **LEARNING_RATE**: Learning rate for the optimizer.
- **LOSS**: Loss function used during training.
- **BATCH_SIZE**: The number of samples processed together in one pass.
- **EPOCHS**: Number of training epochs (iterations over the entire dataset).
- **VAL_SPLIT**: Fraction of the data used for validation.

Initial default values are provided in .config file. You can fine-tune them to your liking.

---

## 📊 Results & Evaluation

- **Validation Accuracy**: ~94–95% 
- **Real Token Accuracy**: ~90–95% 
- **Further Improvements**: Try attention, CRF, or transformer-based embeddings for higher accuracy.

## 🙏 Acknowledgements

- JNU Sanskrit POS tagset dataset
- FastText (`cc.sa.300.bin`) for Sanskrit embeddings
- TensorFlow & Keras for model building and training

---




