import optuna
from torch.utils.data import random_split
import sys

sys.path.append('../')
from diabnet import data
from diabnet.train import train


def all_snp_age_features():
    feat_names = ['snp_{}'.format(i+1) for i in range(5173)]
    feat_names.append("AGE")
    return feat_names

def selected_features():
    # features selected by Fisher test (top 50 + bottom 50 + AGE)
    sel = ["snp_1126","snp_2444","snp_1","snp_1757","snp_9","snp_11","snp_2454","snp_10","snp_376","snp_8",
    "snp_715","snp_1417","snp_1485","snp_4","snp_2","snp_3","snp_6","snp_5","snp_469","snp_1208",
    "snp_1780","snp_790","snp_1301","snp_2208","snp_966","snp_992","snp_577","snp_1938","snp_1494","snp_2478",
    "snp_1843","snp_2241","snp_1045","snp_368","snp_1652","snp_299","snp_615","snp_1150","snp_1479","snp_1461",
    "snp_234","snp_1166","snp_7","snp_1283","snp_1773","snp_1272","snp_359","snp_954","snp_2371","snp_1509",
    "snp_2474","snp_2472","snp_2470","snp_2469","snp_2462","snp_2459","snp_2458","snp_2450","snp_2449","snp_2442",
    "snp_2439","snp_2434","snp_2432","snp_2431","snp_2430","snp_2428","snp_2427","snp_2420","snp_2419","snp_2417",
    "snp_2412","snp_2405","snp_2402","snp_2399","snp_2395","snp_2384","snp_2377","snp_2375","snp_2369","snp_2368",
    "snp_2362","snp_2358","snp_2357","snp_2354","snp_2352","snp_2349","snp_2348","snp_2337","snp_2326","snp_2323",
    "snp_2318","snp_2317","snp_2316","snp_2315","snp_2309","snp_2305","snp_2302","snp_2292","snp_2282","snp_2275","AGE"]
    return sel

def selected_features_wBMI():
    sel = selected_features().append("BMI")
    return sel

def objective(trial):
    params = {
        "l1_neurons": 128,
        "l2_neurons": 0,
        "l3_neurons": 0,
        "dp0": 0,
        # "dp1": trial.suggest_discrete_uniform('dropout_1', 0.05, 1.0, 0.05),
        "dp1": 0.05,
        "dp2": 0,
        "dp3": 0,
        "lr": trial.suggest_loguniform('learning_rate', 1e-6, 1e-3),
        "wd": 0,
        "lambda1_dim1":trial.suggest_loguniform('lambda1_dim1', 1e-6, 1e-2),
        "lambda2_dim1":trial.suggest_discrete_uniform('lambda2_dim1', 0.05, 1.05, 0.05),
        "lambda1_dim2":trial.suggest_loguniform('lambda1_dim2', 1e-6, 1e-2),
        "lambda2_dim2":trial.suggest_discrete_uniform('lambda2_dim2', 0.05, 1.05, 0.05)
    }

    fn_data = '../datasets/train2.csv'

    features = all_snp_age_features()

    dataset = data.DiabDataset(fn_data, features, label="T2D", random_age=False)
    len_trainset = int(0.8*len(dataset))
    trainset, valset = random_split(dataset, [len_trainset, len(dataset)-len_trainset])

    return train(params, trainset, valset, 200,'' ,is_trial=True, device='cuda')

if __name__ == "__main__":
    # epochs = 100
    # train(epochs, './all_data.csv', 'models/model-alpha-02.pth')
    study = optuna.create_study()
    study.optimize(objective, n_trials=1000)
    df = study.trials_dataframe()
    df.to_csv("./hyperopt-1layer-feat-all-elasticnet-full.csv")
    optuna.visualization.plot_intermediate_values(study)
    print(study.best_value)
    print(study.best_trial)
    

# Current best value is 0.3591878928244114 with parameters: {'l1_neurons': 38, 'l2_neurons': 32, 'dropout_1': 0.1, 'dropout_2': 0.1, 'learning_rate': 2.703525027574716e-05, 'weight_decay': 0.00023586800364526562}

# Current best value is 0.36119218170642853 with parameters: {'l1_neurons': 185, 'l2_neurons': 54, 'dropout_1': 0.45, 'dropout_2': 0.6500000000000001, 'learning_rate': 2.170617842710937e-05, 'weight_decay': 1.8875083252341037e-05}.

# FrozenTrial(number=103, state=<TrialState.COMPLETE: 1>, value=0.36309823393821716, datetime_start=datetime.datetime(2019, 7, 24, 17, 0, 26, 113108), datetime_complete=datetime.datetime(2019, 7, 24, 17, 2, 57, 761813), params={'l1_neurons': 118, 'l2_neurons': 91, 'learning_rate': 1.977230023013128e-05, 'weight_decay': 0.00021014222706884097}, distributions={'l1_neurons': IntUniformDistribution(low=32, high=512), 'l2_neurons': IntUniformDistribution(low=8, high=256), 'learning_rate': LogUniformDistribution(low=1e-06, high=0.001), 'weight_decay': LogUniformDistribution(low=1e-07, high=0.001)}, user_attrs={}, system_attrs={'_number': 103}, intermediate_values={}, params_in_internal_repr={'l1_neurons': 118, 'l2_neurons': 91, 'learning_rate': 1.977230023013128e-05, 'weight_decay': 0.00021014222706884097}, trial_id=103)
