from tensorflow.keras.layers import AdditiveAttention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, SimpleRNN, LSTM, GRU, Bidirectional,
    TimeDistributed, Dense, Input,
)
from .config import RNN_UNITS, TRAINABLE_EMBEDDINGS, LAYERS, OPTIMIZER, LEARNING_RATE, LOSS, MASK_ZERO, ACTIVATION, RECURRENT_ACTIVATION, DROPOUT, RECURRENT_DROPOUT, RETURN_SEQUENCES

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
                learning_rate=LEARNING_RATE,
                loss=LOSS,
                activation=ACTIVATION,
                recurrent_activation=RECURRENT_ACTIVATION,
                dropout=DROPOUT,
                recurrent_dropout=RECURRENT_DROPOUT,
                return_sequences=RETURN_SEQUENCES
                ):
    
    input_layer = Input(shape=(input_len,))
    x = Embedding(input_dim=input_dim,
                  output_dim=output_dim,
                  weights=[embedding_weights],
                  trainable=trainable)(input_layer)
    
    for arch in layers:
        x = get_rnn_layer(
            arch=arch,
            rnn_units=rnn_units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=return_sequences)(x)
    
    attention_output = AdditiveAttention()([x, x])
    x = Concatenate()([x, attention_output])

    output = TimeDistributed(Dense(num_classes, activation='softmax'))(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=optimizer(learning_rate=learning_rate),
        loss=loss,
        metrics=['acc'],
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile="auto",
        auto_scale_loss=True,
    )
    model.summary()
    return model
