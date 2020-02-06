from diabnet import data
from diabnet.model import Model
from diabnet.optim import RAdam
from torch.utils.data import DataLoader, random_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
import datetime
# RADAM
import math
# import torch
# from torch.optim.optimizer import Optimizer, required

def train(params, training_set, validation_set, epochs, fn_to_save_model, is_trial=False, device='cuda'):
    device=torch.device(device)

# Current best value is 0.3591878928244114 with parameters: 
# {'l1_neurons': 38, 
# 'l2_neurons': 32, 
# 'dropout_1': 0.1, 
# 'dropout_2': 0.1, 
# 'learning_rate': 2.703525027574716e-05, 
# 'weight_decay': 0.00023586800364526562}

##### one layer 
# Current best value is 0.3367238938808441 with parameters: 
# {'l1_neurons': 140, 
# 'dropout_1': 0.5, 
# 'learning_rate': 1.4787445433385137e-05, 
# 'weight_decay': 0.0002787682549966316}

##### two layers

    # if trial:
    #     l1_neurons = trial.suggest_int('l1_neurons',2,192)
    #     l2_neurons = 0
    #     l3_neurons = 0
    #     # dp0 = trial.suggest_discrete_uniform('dropout_0', 0.0, 1.0, 0.05)
    #     dp0 = 0
    #     dp1 = trial.suggest_discrete_uniform('dropout_1', 0.05, 1.0, 0.05)
    #     dp2 = 0
    #     dp3 = 0
    #     lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    #     wd = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    # else:
    #     l1_neurons = 100
    #     l2_neurons = 0
    #     l3_neurons = 0
    #     dp0 = 0
    #     dp1 = 0.05
    #     dp2 = 0.0
    #     dp3 = 0.0
    #     # lr = 1.5e-05
    #     lr = 3e-04
    #     wd = 0.00045

    # dataset = data.DiabDataset(fn_data, random_age=False)
    # print(len(dataset))
    # ltr, lva = int(0.7*len(dataset)), int(0.1*len(dataset))
    # trainset, valset, testset = random_split(dataset, [ltr, lva, len(dataset)-ltr-lva])

    trainloader = DataLoader(training_set, batch_size=384, shuffle=True)
    
    valloader = DataLoader(validation_set, batch_size=64, shuffle=False)

    # print("Number of features", training_set.dataset.n_feat)
    model = Model(
        training_set.dataset.n_feat, 
        params["l1_neurons"], 
        params["l2_neurons"], 
        params["l3_neurons"], 
        params["dp0"], 
        params["dp1"], 
        params["dp2"], 
        params["dp3"])
    model.to(device)

    loss_func = BCEWithLogitsLoss()
    loss_func.to(device)

    # optimizer = Adam(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    optimizer = RAdam(model.parameters(), lr=params["lr"])

    # lambda to L1 regularization at LC layer
    lambda1_dim1 = params["lambda1_dim1"]
    lambda2_dim1 = params["lambda2_dim1"]
    lambda1_dim2 = params["lambda1_dim2"]
    lambda2_dim2 = params["lambda2_dim2"]

    for e in range(epochs):
        # model.train()
        # trainloader.dataset.random_age = True

        training_loss, training_loss_reg, n_batchs = 0.0, 0.0, 0
        for i, sample in enumerate(trainloader):
            x, y_true = sample
            y_pred = model(x.to(device))
            # print(training_set.dataset.n_feat)
            # print(y_pred)
            # print(y_pred.shape)
            loss = loss_func(y_pred, y_true.to(device))

            # elastic net regularization
            lc_params = model.lc.weight
            l1_regularization_dim1 = lambda2_dim1*torch.sum(torch.norm(lc_params,1, dim=1))
            l2_regularization_dim1 = (1.0-lambda2_dim1)/2.0*torch.sum(torch.norm(lc_params,2, dim=1))
            l1_regularization_dim2 = lambda2_dim2*torch.sum(torch.norm(lc_params,1, dim=2))
            l2_regularization_dim2 = (1.0-lambda2_dim2)/2.0*torch.sum(torch.norm(lc_params,2, dim=2))
            
            dim1_loss = lambda1_dim1 * (l1_regularization_dim1 + l2_regularization_dim1)
            dim2_loss = lambda1_dim2 * (l1_regularization_dim2 + l2_regularization_dim2)

            loss_reg = loss + dim1_loss + dim2_loss

            optimizer.zero_grad()
            loss_reg.backward()
            optimizer.step()

            training_loss += loss.item()
            training_loss_reg += loss_reg.item()
            n_batchs += 1

        training_loss /= n_batchs
        training_loss_reg /= n_batchs

        if not is_trial:
            # print only the loss of the last batch of each epoch
            print("lreg DIM 1", dim1_loss)
            print("lreg DIM 2", dim2_loss)
            print("T epoch {}, loss {}, loss_with_regularization {}".format(e, training_loss, training_loss_reg))


        # loss_sum, acc_sum, bacc_sum = 0, 0, 0
        # loss_val_list, acc_val_list, bacc_val_list = [],[],[]
        validation_loss = 0.0
        cm = np.zeros((2,2))
        n_batchs = 0
        # valloader.dataset.random_age = False
        for s in range(30):   
            for i, sample in enumerate(valloader):
                x, y_true = sample
                y_pred = model(x.to(device))

                loss = loss_func(y_pred, y_true.to(device))
                # loss_val_list.append(loss.item())

                t = y_true.gt(0.5).cpu().detach().numpy()
                p = y_pred.gt(0).cpu().detach().numpy()
                # print(x)
                # print(y_pred)
                # print(torch.sigmoid(y_pred))
                # print(p)
                # return
                # acc_val_list.append(accuracy_score(t, p))
                # bacc_val_list.append(balanced_accuracy_score(t, p))
                cm += np.array(confusion_matrix(t, p))
                # auc = roc_auc_score(t, y_pred.detach().numpy())
                # print(auc)
                validation_loss += loss.item()
                
                n_batchs += 1 
                # return

        validation_loss /= n_batchs # 30 because we are using 30 samples "for...range(30)"
        validation_acc = np.sum(np.diag(cm)) / cm.sum()
        validation_bacc = np.mean(np.diag(cm) / cm.sum(axis=1))

        if not is_trial:
            print("V epoch {}, loss {}, acc {}, bacc {}".format(e, validation_loss, validation_acc, validation_bacc))
            print("line is true, column is pred")
            print(cm/30)
        # print("V epoch {}, loss {}, acc {}, bacc {}".format(e, loss_sum/(n*20), acc_sum/(n*20), bacc_sum/(n*20)))
    
    if is_trial:
        print("V epoch {}, loss {}, acc {}, bacc {}".format(e, validation_loss, validation_acc, validation_bacc))
        print(cm/30)
        print(params)
    # print("V epoch {}, loss {}, acc {}, bacc {}".format(e, loss_sum/(n*20), acc_sum/(n*20), bacc_sum/(n*20)))
    # print(cm/20)
    

    if fn_to_save_model != "":
        print("Saving model at {}".format(fn_to_save_model))
        torch.save(model, fn_to_save_model)
        with open(fn_to_save_model+'.txt', 'w') as f:
            f.write(str(datetime.datetime.now()))
            f.write("\nModel name: {}\n".format(fn_to_save_model))
            # f.write("\nAccuracy: {}\n".format(np.median(acc_val_list)))
            # f.write("\nBalanced Accuracy: {}\n".format(np.median(bacc_val_list)))
            f.write("\nConfusion matrix:\n{}\n".format(cm/30))
            f.write("\nT Loss: {}\n".format(training_loss))
            f.write("\nT Loss(reg): {}\n".format(training_loss_reg))
            f.write("\nV Loss: {}\n\n".format(validation_loss))
            for k in params:
                f.write("{} = {}\n".format(k,params[k]))
            f.close()
            
    return validation_loss



