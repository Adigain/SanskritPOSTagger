from tensorflow.keras.utils import to_categorical
from .config import BATCH_SIZE, EPOCHS, VAL_SPLIT

def train_and_prepare_model(model, X, Y, val_split=VAL_SPLIT):
    Y_cat = to_categorical(Y)
    history = model.fit(
    x=X,
    y=Y_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose="auto",
    callbacks=None,
    validation_split=val_split,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=BATCH_SIZE,
    validation_freq=1,
)
    return model, history
