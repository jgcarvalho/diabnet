from diabnet import data
from diabnet.train import train
from torch.utils.data import random_split

DATASET = './datasets/visits_sp_unique_train_positivo_1000_negative_0.csv'

def net():
    # neural network to SELECTED features with fixed dropout=0.5 during hyperopt
    for i in range(100):
        params = {
            "l1_neurons": 256,
            "l2_neurons": 0,
            "l3_neurons": 0,
            "dp0": 0,
            "dp1": 0.5,
            "dp2": 0,
            "dp3": 0,
            "lr": 0.0007,
            "wd": 0.000001,
            "lambda1_dim1":0.0003,
            "lambda2_dim1":0.05,
            "lambda1_dim2":0.0008,
            "lambda2_dim2":0.05
            # to explore new weights
            # "lambda1_dim1":0.0002,
            # "lambda2_dim1":0.03,False
            # "lambda1_dim2":0.0005,
            # "lambda2_dim2":0.03
        }

        epochs = 500
        fn_data = DATASET

        features = data.get_feature_names(DATASET, BMI=False, sex=True, parents_diagnostics=True)
        # print(features)
        
        dataset = data.DiabDataset(fn_data, features, random_age=False, soft_label=True)
        len_trainset = int(0.8*len(dataset))
        trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

        train(params, trainset, valset, epochs, 'diabnet/models/teste-sp-soft-label_tmp.pth', device='cuda')
        # break to train only one model
        break

# def net_01():
#     # neural network to SELECTED features
#     for i in range(100):
#         params = {
#             "l1_neurons": 96,
#             "l2_neurons": 0,
#             "l3_neurons": 0,
#             "dp0": 0.0,
#             "dp1": 0.05,
#             "dp2": 0,
#             "dp3": 0,
#             "lr": 0.001, #adam 0.0007 radam 0.001
#             "wd": 0.0000,
#             "lambda1_dim1":0.0003,
#             "lambda2_dim1":0.05,
#             "lambda1_dim2":0.0008,
#             "lambda2_dim2":0.05
#         }

#         epochs = 500
#         # fn_data = './datasets/train.csv'
#         fn_data = DATASET

#         features = selected_features()
        
#         dataset = data.DiabDataset(fn_data, features , random_age=False)
#         len_trainset = int(0.8*len(dataset))
#         trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

#         # train(params, trainset, valset, epochs, 'diabnet/models/model-layer01-FEAT-SELECT-50BW-{:03d}.pth'.format(i))
#         train(params, trainset, valset, epochs, 'diabnet/models/model-layer01-FEAT-SELECT-50BW-teste-elasticnet.pth')
#         break

# def net_01b():
#     # neural network to SELECTED features with fixed dropout=0.5 during hyperopt
#     for i in range(100):
#         params = {
#             "l1_neurons": 256,
#             "l2_neurons": 0,
#             "l3_neurons": 0,
#             "dp0": 0,
#             "dp1": 0.5,
#             "dp2": 0,
#             "dp3": 0,
#             "lr": 0.0013,
#             "wd": 0.00001,
#             "lambda1_dim1":0.0003,
#             "lambda2_dim1":0.05,
#             "lambda1_dim2":0.0008,
#             "lambda2_dim2":0.05
#         }

#         epochs = 1000
#         fn_data = './datasets/train.csv'

#         features = selected_features()
        
#         dataset = data.DiabDataset(fn_data, features , random_age=False)
#         len_trainset = int(0.8*len(dataset))
#         trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

#         train(params, trainset, valset, epochs, 'diabnet/models/model-layer01-FEAT-SELECT-50BW-teste-elasticnet-dropout.pth')
#         break

# def net_01_BMI():
#     # neural network to SELECTED features 
#     for i in range(100):
#         params = {
#             "l1_neurons": 96,
#             "l2_neurons": 0,
#             "l3_neurons": 0,
#             "dp0": 0,
#             "dp1": 0.05,
#             "dp2": 0,
#             "dp3": 0,
#             "lr": 0.00045,
#             "wd": 0.0001
#         }

#         epochs = 300
#         fn_data = './datasets/train2.csv'

#         features = selected_features_wBMI()
        
#         dataset = data.DiabDataset(fn_data, features , random_age=False)
#         len_trainset = int(0.8*len(dataset))
#         trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

#         train(params, trainset, valset, epochs, 'diabnet/models/model-layer01-FEAT-SELECT-50BW-BMI-{:03d}.pth'.format(i))

# def net_01b_BMI():
#     # neural network to SELECTED features with fixed dropout=0.5 during hyperopt
#     for i in range(100):
#         params = {
#             "l1_neurons": 256,
#             "l2_neurons": 0,
#             "l3_neurons": 0,
#             "dp0": 0,
#             "dp1": 0.5,
#             "dp2": 0,
#             "dp3": 0,
#             "lr": 0.00025,
#             "wd": 0.0001
#         }

#         epochs = 300
#         fn_data = './datasets/train2.csv'

#         features = selected_features_wBMI()
        
#         dataset = data.DiabDataset(fn_data, features , random_age=False)
#         len_trainset = int(0.8*len(dataset))
#         trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

#         train(params, trainset, valset, epochs, 'diabnet/models/model-layer01-FEAT-SELECT-50BW-BMI-fixedDP-{:03d}.pth'.format(i))

# def net_01_teste():
#     # neural network to SELECTED features

#     params = {
#         "l1_neurons": 256,
#         "l2_neurons": 0,
#         "l3_neurons": 0,
#         "dp0": 0,
#         "dp1": 0.05,
#         "dp2": 0,
#         "dp3": 0,
#         "lr": 0.00045,
#         "wd": 0.00000001
#     }

#     epochs = 300
#     fn_data = './datasets/train.csv'

#     features = selected_features()
    
#     dataset = data.DiabDataset(fn_data, features , random_age=False)
#     len_trainset = int(0.8*len(dataset))
#     trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

#     train(params, trainset, valset, epochs, 'diabnet/models/model-teste-1layer.pth')

# def net_02():
#     # neural network to SELECTED features with fixed dropout=0.5 during hyperopt
#     for i in range(100):
#         params = {
#             "l1_neurons": 256,
#             "l2_neurons": 0,
#             "l3_neurons": 0,
#             "dp0": 0,
#             "dp1": 0.01,
#             "dp2": 0,
#             "dp3": 0,
#             "lr": 0.007,
#             "wd": 0.000,
#             "lambda1_dim1":0.000,
#             "lambda2_dim1":0.05,
#             "lambda1_dim2":0.000,
#             "lambda2_dim2":0.05
#         }

#         epochs = 500
#         fn_data = './datasets/train2.csv'

#         n = 20
#         features = first_n_features(n)


#         dataset = data.DiabDataset(fn_data, features , random_age=False)
#         len_trainset = int(0.8*len(dataset))
#         trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

#         train(params, trainset, valset, epochs, 'diabnet/models/model-layer01-first-{}-feat-teste-elasticnet-dropout.pth'.format(n))
#         break

# def net_03():
#     # neural network to ALL features
#     params = {
#         "l1_neurons": 128,
#         "l2_neurons": 64,
#         "l3_neurons": 0,
#         "dp0": 0.0,
#         "dp1": 0.5,
#         "dp2": 0.0,
#         "dp3": 0,
#         "lr": 0.0007,
#         "wd": 0.00,
#         "lambda1_dim1":0.00002,
#         "lambda2_dim1":0.5,
#         "lambda1_dim2":0.000013,
#         "lambda2_dim2":0.65
#     }

#     epochs = 1000
#     fn_data = './datasets/train2.csv'

#     features = snp_age_features()
    
#     dataset = data.DiabDataset(fn_data, features , random_age=False)
#     len_trainset = int(0.8*len(dataset))
#     trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

#     train(params, trainset, valset, epochs, 'diabnet/models/model-teste-2layer-allfeat-elasticnet.pth')

# def net_04b():
#     # neural network to SELECTED features with fixed dropout=0.5 during hyperopt
#     for i in range(100):
#         params = {
#             "l1_neurons": 256,
#             "l2_neurons": 0,
#             "l3_neurons": 0,
#             "dp0": 0,
#             "dp1": 0.5,
#             "dp2": 0,
#             "dp3": 0,
#             "lr": 0.0013,
#             "wd": 0.000001,
#             # "lambda1_dim1":0.0003,
#             # "lambda2_dim1":0.05,
#             # "lambda1_dim2":0.0008,
#             # "lambda2_dim2":0.05
#             "lambda1_dim1":0.0001,
#             "lambda2_dim1":0.05,
#             "lambda1_dim2":0.0005,
#             "lambda2_dim2":0.05
#         }

#         epochs = 1000
#         fn_data = './datasets/train.csv'

#         features = selected_features_50B500W()
        
#         dataset = data.DiabDataset(fn_data, features , random_age=False)
#         len_trainset = int(0.8*len(dataset))
#         trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

#         train(params, trainset, valset, epochs, 'diabnet/models/model-layer01-FEAT-SELECT-50B500W-teste-elasticnet-dropout.pth')
#         break

# def net_05b():
#     # neural network to SELECTED features with fixed dropout=0.5 during hyperopt
#     for i in range(100):
#         params = {
#             "l1_neurons": 256,
#             "l2_neurons": 0,
#             "l3_neurons": 0,
#             "dp0": 0,
#             "dp1": 0.5,
#             "dp2": 0,
#             "dp3": 0,
#             "lr": 0.0013,
#             "wd": 0.0000001,
#             "lambda1_dim1":0.0001,
#             "lambda2_dim1":0.02,
#             "lambda1_dim2":0.0005,
#             "lambda2_dim2":0.02
#         }

#         epochs = 1000
#         fn_data = './datasets/train.csv'

#         features = selected_features_50B2000W()
        
#         dataset = data.DiabDataset(fn_data, features , random_age=False)
#         len_trainset = int(0.8*len(dataset))
#         trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

#         train(params, trainset, valset, epochs, 'diabnet/models/model-layer01-FEAT-SELECT-50B2000W-teste-elasticnet-dropout.pth')
#         break


if __name__ == "__main__":
    # net_01()
    # net_01b()
    # net_01_BMI()
    # net_01b_BMI()
    # net_01_teste()
    # net_02()
    # net_04b()
    # net_05b()
    # get_feature_names(DATASET, BMI=True)
    net()