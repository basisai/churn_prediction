"""
Script containing commonly used functions.
"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


def lgb_roc_auc_score(y_true, y_pred):
    return "roc_auc", metrics.roc_auc_score(y_true, y_pred), True


def print_results(actual, probs):
    preds = (probs > 0.5).astype(int)
    print('Confusion matrix:')
    print(metrics.confusion_matrix(actual, preds), "\n")
    print(metrics.classification_report(actual, preds))


# ROC(tpr-fpr) curve
def plot_roc_curve(actual, pred, ax=None):
    """Plot ROC."""
    fpr, tpr, _ = metrics.roc_curve(actual, pred)

    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC AUC = {:.4f}'.format(
        metrics.roc_auc_score(actual, pred)))
    return ax


# Precision-recall curve
def plot_pr_curve(actual, pred, ax=None):
    """Plot PR curve."""
    precision, recall, _ = metrics.precision_recall_curve(actual, pred)

    if ax is None:
        fig, ax = plt.subplots()
        
    ax.step(recall, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Avg precision = {:.4f}'.format(
        metrics.average_precision_score(actual, pred)))
    return ax
