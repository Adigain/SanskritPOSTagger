from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Bidirectional, TimeDistributed, Dense
from .config import RNN_UNITS, TRAINABLE_EMBEDDINGS, LAYERS, OPTIMIZER, LOSS, MASK_ZERO, ACTIVATION, RECURRENT_ACTIVATION, DROPOUT, RECURRENT_DROPOUT, RETURN_SEQUENCES

def get_rnn_layer(arch, rnn_units, activation, recurrent_activation, dropout, recurrent_dropout, return_sequences):
    if arch == 'bilstm':
        return Bidirectional(LSTM(
            units=rnn_units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=return_sequences))
    elif arch == 'lstm':
        return LSTM(
            units=rnn_units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=return_sequences)
    elif arch == 'gru':
        return GRU(
            units=rnn_units,
            activation=activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=return_sequences)
    elif arch == 'rnn':
        return SimpleRNN(
            units=rnn_units,
            activation=activation,
            dropout=dropout,
            return_sequences=return_sequences)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


def build_model(input_dim,
                output_dim, 
                input_len, 
                num_classes, 
                embedding_weights,
                rnn_units=RNN_UNITS,
                layers=LAYERS,
                trainable=TRAINABLE_EMBEDDINGS, 
                optimizer=OPTIMIZER,
                loss=LOSS,
                mask_zero=MASK_ZERO,
                activation=ACTIVATION,
                recurrent_activation=RECURRENT_ACTIVATION,
                dropout=DROPOUT,
                recurrent_dropout=RECURRENT_DROPOUT,
                return_sequences=RETURN_SEQUENCES
                ):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim,
                        output_dim=output_dim,
                        input_length=input_len,
                        weights=[embedding_weights],
                        mask_zero=mask_zero,
                        trainable=trainable))

    for arch in layers:
        model.add(get_rnn_layer(
            arch=arch,
            rnn_units=rnn_units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=return_sequences))

    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    model.compile(
    optimizer=optimizer,
    loss=loss,
    loss_weights=None,
    metrics=['acc'],
    weighted_metrics=None,
    run_eagerly=False,
    steps_per_execution=1,
    jit_compile="auto",
    auto_scale_loss=True,
)
    model.build(input_shape=(None, input_len))
    model.summary()
    return model

