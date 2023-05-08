import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

epochs = ["20", "100", "150", "200"]
r = "results/metrics.txt"
with open(r, "w") as f:
        f.write("label,epoch,precision,recall,roc_auc,pr_auc\n")

for i in range(0, 10):
    precision_sum = 0
    recall_sum = 0
    for e in epochs:
        path = str(i) + "_" + e + "_score.csv"

        df = pd.read_csv("results/" + path)
        trainig_label = i
        labels = np.where(df["label"].values == trainig_label, 0, 1)
        anomaly_score = df["anomaly_score"].values
        img_distance = df["img_distance"].values
        z_distance = df["z_distance"].values

        fpr, tpr, _ = roc_curve(labels, img_distance)
        precision, recall, _ = precision_recall_curve(labels, img_distance)
        roc_auc = auc(fpr, tpr)
        pr_auc =  auc(recall, precision)
        precision = np.average(precision)
        recall = np.average(recall)

        with open(r, "a") as f:
            f.write(str(i)+","+e+","+str(precision)+","+str(recall)+","+str(roc_auc)+","+str(pr_auc)+"\n")

        precision_sum += precision
        recall_sum += recall
        """
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC-AUC")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        p = "results/" + str(i) + "_" + e + "_" +"rocauc.png"
        plt.savefig(p)

        plt.plot(recall, precision, label=f"PR = {pr_auc:3f}")
        plt.title("PR-AUC")
        plt.xlabel("Recall")
        plt.ylabel("Pecision")
        plt.legend()
        q = "results/" + str(i) + "_" + e + "_" +"prauc.png"
        plt.savefig(q)
        """

    precision_sum = precision_sum/4
    recall_sum = recall_sum/4
    with open(r, "a") as f:
            f.write("overall_"+str(i)+","+e+","+str(precision_sum)+","+str(recall_sum)+"\n")