from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def evaluate_predictions(y_true, y_pred):
    flat_true, flat_pred = [], []
    for true_tags, pred_tags in zip(y_true, y_pred):
        min_len = min(len(true_tags), len(pred_tags))
        for t, p in zip(true_tags[:min_len], pred_tags[:min_len]):
            flat_true.append(t.lower())
            flat_pred.append(p.lower())

    print("Flat True Length:", len(flat_true))
    print("Flat Pred Length:", len(flat_pred))

    acc = accuracy_score(flat_true, flat_pred)
    print(f"\nEvaluation Result: Accuracy = {acc * 100:.2f}%")
    return acc

def plot_comparison(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.grid(True)
    plt.show()