from __future__ import division
from __future__ import print_function

import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.gmm import GMM
import tools
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def metrics(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    pr_auc =  auc(recall, precision)
    return np.round(np.average(precision),4), np.round(np.average(recall),4), np.round(np.average(roc_auc),4), np.round(np.average(pr_auc),4)


def main(opt):
    contamination = 0.1  # percentage of outliers
   
    # Generate sample data
    if opt.data == "mnist":
        X_train, X_test, y_train, y_test = tools.load_mnist("dataset",
                                        training_label=opt.training_label)
    elif opt.data == "svhn":
        X_train, X_test, y_train, y_test = tools.load_svhn("dataset",
                                        training_label=opt.training_label)
    elif opt.data == "usps":
        X_train, X_test, y_train, y_test = tools.load_usps("dataset", training_label=opt.training_label)

    nsamples, nx, ny = X_train.shape
    d2_train_dataset = X_train.reshape((nsamples,nx*ny))

    nsamples_test, nx_test, ny_test = X_test.shape
    d2_test_dataset = X_test.reshape((nsamples_test,nx_test*ny_test))

    # train VAE detector (Beta-VAE)
    # clf = VAE(epochs=opt.n_epochs, contamination=contamination, gamma=0.8, capacity=0.2)
    clf = GMM(n_components=10)
    clf.fit(d2_train_dataset)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(d2_test_dataset)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(d2_test_dataset)  # outlier scores
    
    print("\nOn Test Data:")
    precision_test, recall_test, roc_auc_test, pr_auc_test = metrics(y_test, y_test_pred)
    
    # test data write  
    with open("results/" + opt.data + '_test.txt', 'a') as f:
        a = "\n" + str(opt.training_label) + ",test," + str(opt.n_epochs) + "," + str(precision_test) + "," + str(recall_test) + "," + str(roc_auc_test) + "," + str(pr_auc_test)
        f.write(a)

       
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of the batches")
    parser.add_argument("--data", type=str, default="None",
                        help="type of dataset")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels")
    parser.add_argument("--training_label", type=int, default=0,
                        help="label for normal images")
    
    opt = parser.parse_args()

    main(opt)
