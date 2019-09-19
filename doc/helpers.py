import seaborn as sns
import matplotlib.pyplot as plt
from inspect import signature
from sklearn import metrics


def print_results(actual, pred):
    y_preds = (pred > 0.5).astype(int)
    print('Confusion matrix:')
    print(metrics.confusion_matrix(actual, y_preds))
    print('Accuracy: {:.2f}%'.format(metrics.accuracy_score(actual, y_preds) * 100))
    print('Precision: {:.2f}%'.format(metrics.precision_score(actual, y_preds) * 100))
    print('Recall: {:.2f}%'.format(metrics.recall_score(actual, y_preds) * 100))
    print('F1 score: {:.2f}%'.format(metrics.f1_score(actual, y_preds) * 100))


# ROC(tpr-fpr) curve
def plot_roc_curve(actual, pred):
    """Plot ROC."""
    fpr, tpr, _ = metrics.roc_curve(actual, pred)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC AUC = {:.4f}'.format(
        metrics.roc_auc_score(actual, pred)))
    return fig


# Precision-recall curve
def plot_pr_curve(actual, pred):
    """Plot PR curve."""
    precision, recall, _ = metrics.precision_recall_curve(actual, pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    fig, ax = plt.subplots()
    ax.step(recall, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Avg precision = {:.4f}'.format(
        metrics.average_precision_score(actual, pred)))
    return fig

