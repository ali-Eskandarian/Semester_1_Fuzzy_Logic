import matplotlib.pyplot as plt
import os
import numpy as np

def plot_fuzzy_sets(data, 
                    fuzzy_vars, 
                    selected_input_columns, 
                    fuzzy_type, 
                    folder = "plots_2",
                    scaled = "Minmax",
                    kmeans_dim = 1):
    for col in selected_input_columns:
        plt.figure(figsize=(10, 6))
        
        if scaled == "Nothing":
            L_min = min(data[col])
            L_max = max(data[col])
            x = np.linspace(L_min, L_max, 1000)
        elif scaled == "Minmax":
            x = np.linspace(0,1, 1000)
        elif scaled == "Standard":
            x = np.linspace(-3,3, 1000)
        elif scaled == "Robust":
            x = np.linspace(-1,1, 1000)
        
        if fuzzy_type == 'type1':
            for key in fuzzy_vars[col].keys():
                values = [fuzzy_vars[col][key](val) for val in x]
                plt.plot(x, values, label=key.capitalize())
        
        elif fuzzy_type == 'type2':
            for key in fuzzy_vars[col].keys():
                values = [fuzzy_vars[col][key](val) for val in x]
                plt.plot(x, values, label=key.replace('_', ' ').capitalize())
        
        plt.title(f'Fuzzy Sets for {col}')
        plt.xlabel(col)
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.savefig(f'{folder}/fuzzy_sets_{col}_{fuzzy_type}_scaling_{scaled}_kmeans_dim_{kmeans_dim}.png')
        plt.close()

def plot_distribution(data, columns, folder = "plots_2"):
    os.makedirs(folder, exist_ok=True)
    for column in columns:
        plt.figure()
        data[column].hist(bins=30)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(f'{folder}/{column}_distribution.png')
        plt.close() 

def plot_predictions_vs_actuals(best_actual, best_prediction, sk_pred, output_col, fuzzy_type , folder = "plots_2"):
    plt.figure()
    plt.scatter(best_actual, best_prediction, label='Predictions', alpha=0.5)
    plt.scatter(best_actual, sk_pred, color='red', label='Regression XGBoost')
    plt.plot([min(best_actual), max(best_actual)], [min(best_actual), max(best_actual)], '--', color='gray', label='Perfect Prediction')
    plt.title(f'Predicted vs Actual for {output_col} ({fuzzy_type})')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.savefig(f'{folder}/{output_col}_predicted_vs_actual_{fuzzy_type}.png')
    plt.close()

def plot_performance_metrics_all(output_columns, results, scales, fuzzy_shapes,fuzzy_types_to_use, folder="plots_2"):
    """
    Plots bar charts for RMSE and R2 score for each output column.
    """
    os.makedirs(folder, exist_ok=True)
    metrics = ['Best RMSE', 'Average RMSE', 'R2 Score']

    num_scales = len(scales)
    num_shapes = len(fuzzy_shapes)
    bar_width = 0.35 

    for output_col in output_columns:
        plt.figure(figsize=(12, 8))
        results_for_output_col = results[output_col]
        for fuzzy_type in fuzzy_types_to_use:
            results_for_fuzzy_type = results_for_output_col[fuzzy_type]
            for m, metric in enumerate(metrics):
                for key, value in results_for_fuzzy_type.items():
                    plt.bar(key, value[m])
                plt.xticks(rotation=45)
                plt.title(f'Performance {metric} for {output_col}')
                plt.ylabel('Scores')
                plt.tight_layout()
                plt.savefig(f'{folder}/performance_{metric}_{output_col}_{fuzzy_type}.png')
                plt.close()

def plot_performance_metrics(kmeans_range, metrics_data, metric_name, ridge_alphas, std_factors, folder="plots"):
    """Plot performance metrics (R2 or RMSE) for different parameter combinations"""
    plt.figure(figsize=(10, 6))
    
    
    for i, (ridge_alpha, std_factor) in enumerate(zip(ridge_alphas, std_factors)):
        label = f'α={ridge_alpha}, σ={std_factor}'
        plt.plot(kmeans_range, metrics_data[i], label=label, marker='o')
    
    plt.xlabel('Number of Clusters (kmeans_max)')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Number of Clusters')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder}/{metric_name.lower()}_analysis.png")
    plt.close()