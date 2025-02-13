import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_curve
from sklearn.metrics import RocCurveDisplay



def plot_precision_recall(ground_truth_labels_one_hot, predict_scores, class_names):
    n_classes = len(class_names)
    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth_labels_one_hot.ravel(), predict_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ground_truth_labels_one_hot[:, i], predict_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    
    # Average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            ground_truth_labels_one_hot[:, class_id],
            predict_scores[:, class_id],
            name=f"ROC curve for {class_names[class_id]}",
            # color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
        )
    
    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
    )