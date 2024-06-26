import numpy
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_preds):
    logits, labels = eval_preds


    if type(logits) == tuple:
        ## for BART model
        logits = logits[0]

    predictions = numpy.argmax(logits, axis=-1)

    metrics = {
        "accuracy": accuracy_score(y_true=labels, y_pred=predictions),
        "f1": f1_score(y_true=labels, y_pred=predictions, average="macro"),
    }

    return metrics