import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from aced_hmm.ABC_test_metrics import compute_true_summary_statistics

# Define column mapping for display
column_mapping = {
    'n_InGeneralWard': 'G InGeneralWard',
    'n_InICU': 'I + V InICU',
    'n_OnVentInICU': 'V OnVentInICU',
    'n_TERMINAL': 'T sm. Death'
}

# List of columns we want to analyze
columns_of_interest = list(column_mapping.keys())

def bootstrap_mae_ci(true_values, pred_values, n_bootstrap=1000):
    """
    Calculate bootstrap confidence intervals for MAE.
    
    Args:
        true_values: Array of true values
        pred_values: Array of predicted values
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (mae, lower_bound, upper_bound)
    """
    mae = np.mean(np.abs(true_values - pred_values))
    maes = []
    
    # Perform bootstrap resampling
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(len(true_values), len(true_values), replace=True)
        bootstrap_true = true_values[indices]
        bootstrap_pred = pred_values[indices]
        
        # Calculate MAE for this bootstrap sample
        mae_boot = np.mean(np.abs(bootstrap_true - bootstrap_pred))
        maes.append(mae_boot)
    
    # Calculate confidence interval
    lower_bound = np.percentile(maes, 2.5)  # 2.5th percentile
    upper_bound = np.percentile(maes, 97.5)  # 97.5th percentile
    
    return mae, lower_bound, upper_bound


def get_method_stats(method_name, forecast_path, config_path, true_counts_path):
    """
    Get formatted statistics for a single method using ABC_test_metrics.
    
    Args:
        method_name: Name of the method
        forecast_path: Path to forecast template (without _mean.csv suffix)
        config_path: Path to configuration file
        true_counts_path: Path to true counts CSV file
        
    Returns:
        DataFrame with formatted statistics
    """
    # Get training/testing split information
    with open(config_path, 'r') as f:
        config = json.load(f)
        num_training_timesteps = config.get('num_training_timesteps', 0)
    
    # Load the true data
    true_df = pd.read_csv(true_counts_path)
    
    # Load the forecasted data
    forecast_df = pd.read_csv(forecast_path + '_mean.csv')
    
    # Process data for test period only
    true_test_df = true_df[true_df['timestep'] > num_training_timesteps].copy()
    forecast_test_df = forecast_df[forecast_df['timestep'] > num_training_timesteps].copy()
    
    # Apply summary statistics function to get derived columns
    true_processed = compute_true_summary_statistics(true_test_df, columns_of_interest)
    forecast_processed = compute_true_summary_statistics(forecast_test_df, columns_of_interest)
    
    # Prepare results with bootstrap confidence intervals
    results = []
    for col in columns_of_interest:
        true_values = true_processed[col].values
        pred_values = forecast_processed[col].values
        
        # Calculate MAE with bootstrap confidence intervals
        mae, lower, upper = bootstrap_mae_ci(true_values, pred_values)
        
        # Format for display
        results.append({
            'State': column_mapping[col],
            'Method': method_name,
            'MAE': round(mae, 1),
            'lower': round(lower, 1),
            'upper': round(upper, 1)
        })
    
    # Create and format DataFrame
    results_df = pd.DataFrame(results)
    
    # Create formatted table with MAE and CI
    formatted_df = pd.DataFrame(index=pd.MultiIndex.from_product(
        [[method_name], [column_mapping[col] for col in columns_of_interest]],
        names=['Method', 'State']
    ))
    
    for _, row in results_df.iterrows():
        formatted_df.loc[(row['Method'], row['State']), 'MAE'] = row['MAE']
        formatted_df.loc[(row['Method'], row['State']), 'CI'] = f"{row['lower']} - {row['upper']}"
    
    return formatted_df