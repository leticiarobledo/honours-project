# code based on https://github.com/danieltan07

import numpy as np
import argparse 
import torch

from train import TrainerDAGMM
from test import eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of epochs")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_milestones', type=list, default=[50],
                        help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size")
    parser.add_argument('--latent_dim', type=int, default=1,
                        help='Dimension of the latent variable z')
    parser.add_argument('--n_gmm', type=int, default=4,
                        help='Number of Gaussian components ')
    parser.add_argument('--lambda_energy', type=float, default=0.1,
                        help='Parameter labda1 for the relative importance of sampling energy.')
    parser.add_argument('--lambda_cov', type=int, default=0.005,
                        help='Parameter lambda2 for penalizing small values on'
                             'the diagonal of the covariance matrix')
    #parsing arguments.
    args = parser.parse_args() 

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get data
    from tools import *
    from torch.utils.data import DataLoader

    for data in ["mnist"]: #, "svhn", "usps"]:
        for training_label in range(0,10):
            if data == "mnist":
                X_train, X_test, y_train, y_test = load_mnist("dataset",training_label=training_label,download=False, benchmarking=True)
            elif data == "svhn":
                X_train, X_test, y_train, y_test = load_svhn("dataset", training_label=training_label,benchmarking=False)
            elif data == "usps":
                X_train, X_test, y_train, y_test = load_usps("dataset",training_label=training_label, benchmarking=False)

            train = Dataset(X_train, y_train)
            test = Dataset(X_test, y_test)
            dataloader_train = DataLoader(train, batch_size=args.batch_size, 
                                    shuffle=True, num_workers=0)
            dataloader_test = DataLoader(test, batch_size=args.batch_size, 
                                    shuffle=False, num_workers=0)
            data = [dataloader_train, dataloader_test]

            for epoch in [20]: #,100,150,200]:
                # fit model
                args.num_epochs = epoch
                DAGMM = TrainerDAGMM(args, X_train, device)
                DAGMM.train()
                DAGMM.eval(DAGMM.model, test, device) # data[1]: test dataloader

                # model = DAGMM()
                # model.model_fit(X_train, y_train)
                # score = model.predict_score(X_train,X_test)
                # print(score)

                # predict scores
