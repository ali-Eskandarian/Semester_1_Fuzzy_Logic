import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os
from src.evaluation import evaluate_model
from src.fuzzier import FuzzyRegression
from src.plotter import *



def filter_low_correlation_features(data, input_columns, output_column, threshold=0.1):
    """
    Filters out input features with correlation below the specified threshold with the output column.
    """
    correlations = data[input_columns + [output_column]].corr()[output_column]
    return [col for col in input_columns if abs(correlations[col]) >= threshold]

def main(fuzzy_type, output_col, mode = "All features", scale = "Minmax", fuzzy_shape = "All gaussian"):
    print(f"Evaluating {output_col} with {fuzzy_type} fuzzy type")

    if mode == "All features":
        selected_input_columns = input_columns
    elif mode == "Filtered features":
        selected_input_columns = filter_low_correlation_features(data, input_columns, output_col)

    regressor = FuzzyRegression(selected_input_columns, output_col,std_factor=1/4, fuzzy_type=fuzzy_type, fuzzy_shape=fuzzy_shape)
    best_rmse_result, best_prediction, best_actual, rules, fuzzy_vars, avg_rmse, sk_pred, r2_score = evaluate_model(
        regressor, data, selected_input_columns, output_col, fuzzy_type, scale, num_iterations=1
    )
    
    plot_fuzzy_sets(data, fuzzy_vars, selected_input_columns, fuzzy_type, folder = "plots_3")
    plt.close()
    print(f"Best RMSE ({fuzzy_type}):", best_rmse_result)
    print(f"Average RMSE ({fuzzy_type}):", avg_rmse)
    print(f"R2 Score ({fuzzy_type}):", r2_score)

    plot_predictions_vs_actuals(best_actual, best_prediction, sk_pred, output_col, fuzzy_type , folder = "plots_3")
    plt.close()

    with open(f'fuzzy_rules_{fuzzy_type}_{scale}_{fuzzy_shape}_{output_col}.txt', 'w') as f:
        f.write(f"Generated rules ({fuzzy_type}):\n")
        for rule in rules:
            str_rule = "IF "
            for name, value in rule.items():
                print()
                if name == "output":
                    str_rule = str_rule[:-4]
                    str_rule += f"THEN Y is {value}."
                elif name == "count":
                    pass
                else:
                    str_rule += f"{name} is {value} and "
            f.write(str_rule + '\n')
    return best_rmse_result, avg_rmse, r2_score

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    data = pd.read_excel('ENB2012_data.xlsx')
    input_columns = [f"X{i}" for i in range(1, 9)]
    output_columns = ["Y2"]#, "Y2"]
    
    scales = ["Nothing"]#,"Minmax", "Standard"]#,"Robust"]
    fuzzy_shapes = ["Both Gaussian and Trapezoidal"]#All gaussian"]#, "Both Gaussian and Trapezoidal"]
    fuzzy_types_to_use = ['type2']#, 'type2'], 'type1']
    results = {}
    for i, output_col in enumerate(output_columns):
        types = {}
        for fuzzy_type in fuzzy_types_to_use:
            metrics = {}
            for k, scale in enumerate(scales):
                for l, fuzzy_shape in enumerate(fuzzy_shapes):
                    best_rmse, avg_rmse, r2_score = main(fuzzy_type, output_col, mode="Filtered features", scale=scale, fuzzy_shape=fuzzy_shape)
                    metrics[f"{scale}_{fuzzy_shape}"] = (best_rmse, avg_rmse, r2_score)
            types[fuzzy_type] = metrics
        results[f"{output_col}"] = types

    plot_performance_metrics_all(output_columns, results, scales, fuzzy_shapes,fuzzy_types_to_use, folder="plots_3")

    print("\nBest Configurations:")
    for output_col in output_columns:
        print(f"\nFor {output_col}:")
        for fuzzy_type in fuzzy_types_to_use:
            best_config = min(results[output_col][fuzzy_type].items(), 
                            key=lambda x: x[1][1])  
            scale, shape = best_config[0].split('_')
            print(f"  {fuzzy_type}: Scale={scale}, Shape={shape}, Avg RMSE={best_config[1][1]:.4f}")
