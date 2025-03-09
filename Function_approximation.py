import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from scipy.optimize import minimize
import optuna
from sklearn.discriminant_analysis import StandardScaler
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope
import argparse
from src.fuzzier import FunctionApproximator
import pandas as pd
from src.plotter import plot_fuzzy_sets, plot_performance_metrics
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.metrics import r2_score, mean_squared_error    

def objective(params, df_train, df_test, ytest):
    
    kmeans_max = int(params['kmeans_max'])
    std_factor = params['std_factor']
    ridge_alpha_linear = params['ridge_alpha_linear']
    ridge_alpha_quadratic = params['ridge_alpha_quadratic']
    ridge_alpha_cubic = params['ridge_alpha_cubic']
    ridge_alpha_exp = params['ridge_alpha_exp']
    # print(f"kmeans_max: {kmeans_max}, std_factor: {std_factor}, ridge_alpha_linear: {ridge_alpha_linear}, ridge_alpha_quadratic: {ridge_alpha_quadratic}, ridge_alpha_cubic: {ridge_alpha_cubic}, ridge_alpha_exp: {ridge_alpha_exp}")
    fapp = FunctionApproximator(input_columns=['x'], output_column='y', fuzzy_type='type1',
                                mode='optimize', kmeans_max=kmeans_max, std_factor=std_factor,
                                fiting_mode='both', ridge_alphas=[ridge_alpha_linear, ridge_alpha_quadratic,
                                                                  ridge_alpha_cubic, ridge_alpha_exp])
    
    try:
        fapp.fit(df_train[['x']], df_train['y'])
        preds = fapp.predict(df_test[['x']])
        
        if isinstance(preds[0], tuple):
            y_pred = [(a + b) / 2 for (a, b) in preds]
        else:
            y_pred = preds

        rmse = np.sqrt(np.mean((np.array(y_pred) - ytest) ** 2))
        return {'loss': rmse, 'status': STATUS_OK}
    except:
        return {'loss': 1000000, 'status': STATUS_FAIL}

def main_function_approximator(optimization, mode='optimize',
                               kmeans_max=10, std_factor=0.2, 
                               fiting_mode='both', ridge_alpha=1, 
                               mode_function_approximation='not_visualize',
                               scaled='Nothing',
                               xtrain=None, xtest=None, ytrain=None, ytest=None,
                               kmeans_dim=1):
    
    if scaled == "Minmax":
        scaler = MinMaxScaler()
        scaler.fit(xtrain.reshape(-1, 1))
        xtrain = scaler.transform(xtrain.reshape(-1, 1))
        xtest = scaler.transform(xtest.reshape(-1, 1))
    elif scaled == "Standard":
        scaler = StandardScaler()
        scaler.fit(xtrain.reshape(-1, 1))
        xtrain = scaler.transform(xtrain.reshape(-1, 1))
        xtest = scaler.transform(xtest.reshape(-1, 1))
    elif scaled == "Nothing":
        xtrain = xtrain.reshape(-1, 1)
        xtest = xtest.reshape(-1, 1)
    df_train = pd.DataFrame(xtrain, columns=['x'])
    df_train['y'] = ytrain
    df_test = pd.DataFrame(xtest, columns=['x'])
    df_test['y'] = ytest

    if mode_function_approximation == 'visualize':
        kmeans_range = range(4, 31, 1)  # From 4 to 30 in steps of 2
        ridge_alphas = [10, 5, 2, 1, 0.5, 0.1, 0.05]
        std_factors = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        r2_scores = [[] for _ in range(len(ridge_alphas))]
        rmse_scores = [[] for _ in range(len(ridge_alphas))]
        
        for i, (r_alpha, s_factor) in enumerate(zip(ridge_alphas, std_factors)):
            for k in tqdm(kmeans_range, desc=f'Processing α={r_alpha}, σ={s_factor}'):
                fapp = FunctionApproximator(
                    input_columns=['x'], output_column='y', fuzzy_type='type1',
                    mode='optimize',kmeans_dim=kmeans_dim, kmeans_max=k, std_factor=s_factor, 
                    fiting_mode='both', ridge_alphas=[r_alpha, r_alpha, r_alpha, r_alpha]
                )

                fapp.fit(df_train[['x']], df_train['y'])
                preds = fapp.predict(df_test[['x']])
                
                if isinstance(preds[0], tuple):
                    y_pred = [(a + b)/2 for (a, b) in preds]
                else:
                    y_pred = preds
                
                rmse = np.sqrt(np.mean((np.array(y_pred) - ytest)**2))
                r2 = 1 - (np.sum((np.array(y_pred) - ytest)**2) / np.sum((ytest - np.mean(ytest))**2))
                
                rmse_scores[i].append(rmse)
                r2_scores[i].append(r2)
        
        # Plot the results
        plot_performance_metrics(kmeans_range, r2_scores, 'R2 Score', ridge_alphas, std_factors)
        plot_performance_metrics(kmeans_range, rmse_scores, 'RMSE', ridge_alphas, std_factors)
        
    if optimization:
        space = {
            'kmeans_max': hp.quniform('kmeans_max', 11, 30, 1),
            'std_factor': hp.uniform('std_factor', 0.05, 0.6),
            'ridge_alpha_linear': hp.uniform('ridge_alpha_linear', 0.000001,  0.0001),
            'ridge_alpha_quadratic': hp.uniform('ridge_alpha_quadratic', 0.000001, 0.0001),
            'ridge_alpha_cubic': hp.uniform('ridge_alpha_cubic', 0.000001, 0.0001),
            'ridge_alpha_exp': hp.uniform('ridge_alpha_exp', 0.000001, 0.0001)
        }
        trials = Trials()
        best = fmin(fn=lambda params: objective(params, df_train, df_test, ytest), space=space, algo=tpe.suggest, max_evals=100, trials=trials)
        
        print("Best parameters:", best)

        fapp = FunctionApproximator(input_columns=['x'], output_column='y', fuzzy_type='type1',
                                    mode='optimize',kmeans_dim=kmeans_dim ,kmeans_max=int(best['kmeans_max']),
                                    std_factor=best['std_factor'], fiting_mode='both',
                                    ridge_alphas=[best['ridge_alpha_linear'], best['ridge_alpha_quadratic'],
                                                  best['ridge_alpha_cubic'], best['ridge_alpha_exp']])
    
    else:
        fapp = FunctionApproximator(input_columns=['x'], output_column='y', fuzzy_type='type1',
                                    mode=mode, kmeans_dim=kmeans_dim, kmeans_max=kmeans_max, std_factor=std_factor, 
                                    fiting_mode=fiting_mode, ridge_alphas=ridge_alpha)
    
    fapp.fit(df_train[['x']], df_train['y'])
    fapp.write_rules()
    preds = fapp.predict(df_test[['x']])

    plot_fuzzy_sets(df_train, fapp.regressor.fuzzy_vars, ['x'], 'type1', folder = "plots", scaled=scaled, kmeans_dim=kmeans_dim)
    
    if isinstance(preds[0], tuple):
        y_pred = [(a + b)/2 for (a, b) in preds]
    else:
        y_pred = preds

    rmse = np.sqrt(np.mean((np.array(y_pred) - ytest)**2))
    r2 = 1 - (np.sum((np.array(y_pred) - ytest)**2) / np.sum((ytest - np.mean(ytest))**2))
    print("===============++++++++++++===============")
    print(f"Scaling: {scaled}, Kmeans_dim: {kmeans_dim}")
    print(f"R2 Score: {r2}, RMSE: {rmse}")


    xs = np.linspace(xtrain.min(), xtrain.max(), 100)
    df_xs = pd.DataFrame(xs, columns=['x'])
    all_preds = fapp.predict(df_xs)
    if isinstance(all_preds[0], tuple):
        all_preds = [(a + b)/2 for (a, b) in all_preds]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, all_preds, label='Approx')
    plt.scatter(xtrain, ytrain, color='blue')
    plt.scatter(xtest, ytest, color='red')
    plt.legend()
    plt.savefig(f"plots/function_approximation_scaling_{scaled}_kmeans_dim_{kmeans_dim}.png")
    plt.close()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    xtrain = np.array([-2, -1.928617603731178, -1.8388106323716946, -1.6505360588614146, 
                       -1.6466737987854083, -1.5257057561784362, -1.505126916572999, 
                       -1.2955420401570672, -1.2122777008048495, -1.0858122491745035, 
                       -0.9800400137767338, -0.9779517282635433, -0.9331787789830388, 
                       -0.7874813039993671, -0.619838545990298, -0.4886226045199771, 
                       -0.382651422395774, -0.33300018110610896, -0.08664409599647804, 
                        0.06743694482422713, 0.35682294786440316, 0.39378743445292974, 
                        0.5851822041845041, 0.6804321390567245, 0.7161701249297399, 
                        0.8723862165263401, 0.893034849006435, 0.9333918116453996, 
                        0.9400309986851507, 0.9910626847722677, 1.2247286373398238, 
                        1.2658191715020592, 1.3030635701944662, 1.358983543088455, 
                        1.3593742798418726, 1.3935574671917363, 1.4062142116954033, 
                        1.5388539859334647, 1.6971277518635222, 1.7861198603661284])
    ytrain = np.array([0.9999999999999992, 0.31487794940850466, -0.29666651103951447, 
                       0.08665913205587092, 0.10736120053455869, 0.5176377997731151, 
                       0.5071452098300027, -0.7893502291513371, -1.1844912264703529, 
                       -0.776306607295187, 0.18406664352546748, 0.20271657641210541, 
                       0.5565148316701979, 0.6320946748233786, -0.9758134393879618, 
                       -1.4751116374799857, -0.6382067658058621, 0.004499901071692179, 
                        1.6381326660484838, 0.35096099396214053, 0.12601644334980847, 
                        0.3104637030728553, 0.13238666461451254, -0.6095196293825365, 
                       -0.8693506614049537, -1.0456788261509466, -0.926602149118585, 
                       -0.6195847460262386, -0.5618087599760331, -0.08472370267127548, 
                        0.5525531397237378, 0.2090593933813838, -0.17218692090734544, 
                       -0.7699779902525705, -0.7739470584880559, -1.09399367705507, 
                       -1.1950009789254477, -1.3579923859092413, 0.4103275243772142, 
                        1.4064357239339786])

    xtest = np.array([-1.8652295067688538, -1.532894160073611, -1.1179362405895072, 
                      -1.1096094984898555, -0.76632706968909, -0.13255097331976584, 
                       0.3046084462091203, 0.44498061136975187, 0.5660072373556777, 
                       0.7206199322245301, 0.9000763586957743, 1.3457760241567094, 
                       1.4623462150245938, 1.5243013202317912, 1.8850451348731059])
    ytest = np.array([-0.16810844971419459, 0.5145467362049033, -0.9935548408929169, 
                      -0.9434730918975127, 0.4871193936655061, 1.7424111202167762, 
                      -0.14763621518213493, 0.48411576702695025, 0.25210056265709857, 
                      -0.8981040194704278, -0.8795096331494634, -0.6329426445881473, 
                      -1.4827692234839247, -1.429933468059694, 1.72646286123605])
    
    
    best_params = [{'kmeans_max': 25, 'std_factor': 0.10194332695792915, 'ridge_alpha': 1.9177804306137793}]
    best_params = {'kmeans_max': 25, 'ridge_alpha_cubic': 0.5649857108053228, 
                   'ridge_alpha_exp': 1.3821201141576656, 'ridge_alpha_linear': 0.006568371959581452, 
                   'ridge_alpha_quadratic': 0.7134353388568304, 'std_factor': 0.050020359232745115}
    best_params = {'kmeans_max': 9, 'ridge_alpha_cubic': 0.005649857108053228, 
                   'ridge_alpha_exp': 0.003821201141576656, 'ridge_alpha_linear': 0.006568371959581452, 
                   'ridge_alpha_quadratic': 0.00007134353388568304, 'std_factor': 0.080020359232745115}
    
    # for kmeans_dim in [1, 2]:
    #     for scaling in ['Minmax', 'Standard', "Nothing"]:
    scaling, kmeans_dim = "Minmax", 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization", type=bool, default=False)
    # parser.add_argument("--mode", type=str, default='optimize')
    parser.add_argument("--mode", type=str, default='FCM')
    parser.add_argument("--kmeans_max", type=int, default=best_params['kmeans_max'])
    parser.add_argument("--kmeans_dim", type=int, default=kmeans_dim)    
    parser.add_argument("--ridge_alpha", type=list, default=[best_params['ridge_alpha_linear'], 
                                                        best_params['ridge_alpha_quadratic'], 
                                                        best_params['ridge_alpha_cubic'], 
                                                                best_params['ridge_alpha_exp']])
    parser.add_argument("--fiting_mode", type=str, default='both')
    parser.add_argument("--std_factor", type=float, default=best_params['std_factor'])
    parser.add_argument("--mode_function_approximation", type=str, default='not_visualize')
    parser.add_argument("--scaled", type=str, default=scaling)
    args = parser.parse_args()

    main_function_approximator(optimization=args.optimization, mode=args.mode,kmeans_dim=args.kmeans_dim,
                            kmeans_max=args.kmeans_max, std_factor=args.std_factor,
                            fiting_mode=args.fiting_mode, ridge_alpha=args.ridge_alpha, 
                            scaled=args.scaled,
                            mode_function_approximation=args.mode_function_approximation,
                            xtrain=xtrain, xtest=xtest, ytrain=ytrain, ytest=ytest)
