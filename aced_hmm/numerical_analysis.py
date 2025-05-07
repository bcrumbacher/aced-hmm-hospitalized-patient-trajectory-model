from aced_hmm.ABC_test_metrics import compute_true_summary_statistics
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_forecast_metrics(forecasts_template_path, config_filepath, true_counts_filepath, smooth_terminal_counts=True,
                    expected_columns=['n_discharged_InGeneralWard', 'n_InGeneralWard', 'n_OffVentInICU', 
                                     'n_OnVentInICU', 'n_InICU', 'n_occupied_beds', 'n_TERMINAL']):
    """
    Calculate numerical evaluation metrics for forecasts:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² (Coefficient of Determination)
    - MAPE (Mean Absolute Percentage Error)
    - Forecast Bias
    """
    # Load the data
    true_df = pd.read_csv(true_counts_filepath)
    pred_df = pd.read_csv(forecasts_template_path + '_mean.csv')
    pred_lower_df = pd.read_csv(forecasts_template_path + '_percentile=002.50.csv')
    pred_upper_df = pd.read_csv(forecasts_template_path + '_percentile=097.50.csv')
    
    # Process the data as in the original function
    true_df_processed = compute_true_summary_statistics(true_df, expected_columns, smooth_terminal_counts)
    pred_df_processed = compute_true_summary_statistics(pred_df, expected_columns, smooth_terminal_counts)
    pred_lower_processed = compute_true_summary_statistics(pred_lower_df, expected_columns, smooth_terminal_counts)
    pred_upper_processed = compute_true_summary_statistics(pred_upper_df, expected_columns, smooth_terminal_counts)
    
    # Get training/testing split information
    with open(config_filepath, 'r') as f:
        config = json.load(f)
        num_training_timesteps = config['num_training_timesteps']
    
    # Create results dataframe to store all metrics
    metrics_df = pd.DataFrame(index=expected_columns)
    
    # Calculate metrics for both training and testing periods
    time_periods = {
        'training': (0, num_training_timesteps),
        'testing': (num_training_timesteps, len(true_df)),
        'overall': (0, len(true_df))
    }
    
    for period_name, (start_idx, end_idx) in time_periods.items():
        # Skip if we don't have data for this period
        if start_idx >= end_idx:
            continue
            
        for column in expected_columns:
            true_values = true_df_processed[column].iloc[start_idx:end_idx].values
            pred_values = pred_df_processed[column].iloc[start_idx:end_idx].values
            lower_bound = pred_lower_processed[column].iloc[start_idx:end_idx].values
            upper_bound = pred_upper_processed[column].iloc[start_idx:end_idx].values
            
            # Calculate basic error metrics
            mae = mean_absolute_error(true_values, pred_values)
            rmse = np.sqrt(mean_squared_error(true_values, pred_values))
            r2 = r2_score(true_values, pred_values)
            
            # Calculate Mean Absolute Percentage Error (MAPE)
            # Avoid division by zero
            nonzero_idx = true_values != 0
            if np.any(nonzero_idx):
                mape = np.mean(np.abs((true_values[nonzero_idx] - pred_values[nonzero_idx]) / true_values[nonzero_idx])) * 100
            else:
                mape = np.nan
                
            # Calculate forecast bias (positive = overestimation, negative = underestimation)
            bias = np.mean(pred_values - true_values)
            
            # Calculate prediction interval coverage (PIC)
            # Percentage of true values that fall within the prediction intervals
            in_interval = np.logical_and(true_values >= lower_bound, true_values <= upper_bound)
            pic = np.mean(in_interval) * 100
            
            # Calculate prediction interval width
            pi_width = np.mean(upper_bound - lower_bound)
            
            # Store metrics
            metrics_df.loc[column, f'MAE_{period_name}'] = mae
            metrics_df.loc[column, f'RMSE_{period_name}'] = rmse
            metrics_df.loc[column, f'R2_{period_name}'] = r2
            metrics_df.loc[column, f'MAPE_{period_name}'] = mape
            metrics_df.loc[column, f'Bias_{period_name}'] = bias
            metrics_df.loc[column, f'PIC_{period_name}'] = pic
            metrics_df.loc[column, f'PI_Width_{period_name}'] = pi_width
    
    # Calculate aggregate metrics across all variables
    for period_name in time_periods.keys():
        for metric in ['MAE', 'RMSE', 'R2', 'MAPE', 'Bias', 'PIC', 'PI_Width']:
            col_name = f'{metric}_{period_name}'
            if col_name in metrics_df.columns:
                metrics_df.loc['AVERAGE', col_name] = metrics_df[col_name].mean()
    
    return metrics_df

def evaluate_forecasts_for_specific_events(forecasts_template_path, config_filepath, true_counts_filepath, 
                                          event_thresholds=None, smooth_terminal_counts=True,
                                          expected_columns=['n_discharged_InGeneralWard', 'n_InGeneralWard', 'n_OffVentInICU', 
                                                          'n_OnVentInICU', 'n_InICU', 'n_occupied_beds', 'n_TERMINAL']):
    """
    Evaluate forecasts for specific events like capacity thresholds being crossed
    """
    # Define default thresholds if none provided
    if event_thresholds is None:
        event_thresholds = {
            'n_occupied_beds': [50, 100, 150],
            'n_InICU': [20, 40, 60],
            'n_OnVentInICU': [10, 20, 30]
        }
    
    # Load the data
    true_df = pd.read_csv(true_counts_filepath)
    pred_df = pd.read_csv(forecasts_template_path + '_mean.csv')
    
    # Process the data
    true_df_processed = compute_true_summary_statistics(true_df, expected_columns, smooth_terminal_counts)
    pred_df_processed = compute_true_summary_statistics(pred_df, expected_columns, smooth_terminal_counts)
    
    # Get training/testing split information
    with open(config_filepath, 'r') as f:
        config = json.load(f)
        num_training_timesteps = config['num_training_timesteps']
    
    # Focus on testing period
    true_test = true_df_processed.iloc[num_training_timesteps:]
    pred_test = pred_df_processed.iloc[num_training_timesteps:]
    
    results = {}
    
    # Evaluate threshold crossings
    for column, thresholds in event_thresholds.items():
        if column not in true_test.columns:
            continue
            
        results[column] = {}
        
        for threshold in thresholds:
            # Get days when true values cross threshold
            true_crosses = true_test[column] > threshold
            pred_crosses = pred_test[column] > threshold
            
            # Calculate metrics
            true_positive = (true_crosses & pred_crosses).sum()
            false_positive = (~true_crosses & pred_crosses).sum()
            false_negative = (true_crosses & ~pred_crosses).sum()
            true_negative = (~true_crosses & ~pred_crosses).sum()
            
            # Calculate precision, recall, F1 score
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else np.nan
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else np.nan
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
            
            # Time to threshold - for the first crossing
            if true_crosses.any():
                first_true_cross = true_crosses.idxmax()
                if pred_crosses.any():
                    first_pred_cross = pred_crosses.idxmax()
                    time_diff = first_pred_cross - first_true_cross
                else:
                    time_diff = np.nan
            else:
                time_diff = np.nan
                
            results[column][threshold] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'days_difference_first_crossing': time_diff,
                'true_positives': true_positive,
                'false_positives': false_positive,
                'false_negatives': false_negative,
                'true_negatives': true_negative
            }
    
    return pd.DataFrame({
        (col, threshold, metric): value 
        for col, thresholds in results.items() 
        for threshold, metrics in thresholds.items() 
        for metric, value in metrics.items()
    }).unstack().unstack()

def evaluate_parameter_convergence(samples_path, prior_path='priors/abc_prior_config_OnCDCTableReasonable.json'):
    """
    Evaluate how well the ABC posterior has converged compared to the prior
    """
    # Load samples
    with open(samples_path, 'r') as f:
        params_list = json.load(f)
    
    if not isinstance(params_list, list):
        params_list = [params_list]
        
    # Load prior configuration
    with open(prior_path, 'r') as f:
        prior_dict = json.load(f)
        
    # States in the model
    states = ['InGeneralWard', 'OffVentInICU', 'OnVentInICU']
    
    # Sample from the prior (reusing existing function)
    prior_samples = sample_params_from_prior(prior_dict, states, num_samples=1000)
    
    # Prepare parameter distributions as in the original function
    param_names = list(params_list[0].keys())
    param_distributions = {}
    
    for params in params_list:
        for name in param_names:
            if name not in param_distributions:
                if type(params[name]) == dict:
                    param_distributions[name] = {}
                    for key in params[name]:
                        param_distributions[name][key] = []
                else:
                    param_distributions[name] = []

            if type(params[name]) == dict:
                for key in params[name]:
                    param_distributions[name][key].append(np.asarray(params[name][key]))
            else:
                param_distributions[name].append(np.asarray(params[name]))
    
    # Flatten everything
    for name in param_distributions:
        if type(param_distributions[name]) == dict:
            for key in param_distributions[name]:
                temp = np.array([])
                for arr in param_distributions[name][key]:
                    temp = np.append(temp, arr)
                param_distributions[name][key] = np.copy(temp)
        else:
            temp = np.array([])
            for arr in param_distributions[name]:
                temp = np.append(temp, arr)
            param_distributions[name] = np.copy(temp)
    
    # Calculate convergence metrics
    convergence_metrics = {}
    
    for param in param_distributions:
        if param == 'proba_Die_after_Declining_OnVentInICU':
            continue
            
        if 'duration' in param:
            # For duration parameters (which are distributions themselves)
            prior_means = np.mean(prior_samples[param], axis=0)
            posterior_means = []
            
            for dur in param_distributions[param]:
                if dur not in ['lam', 'tau']:
                    posterior_means.append(np.mean(param_distributions[param][dur]))
            
            posterior_means = np.array(posterior_means)
            
            # Calculate KL divergence between the distributions
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            prior_means_safe = prior_means[:len(posterior_means)] + epsilon
            posterior_means_safe = posterior_means + epsilon
            
            # Normalize to get probability distributions
            prior_means_norm = prior_means_safe / np.sum(prior_means_safe)
            posterior_means_norm = posterior_means_safe / np.sum(posterior_means_safe)
            
            # KL(P||Q) = sum(P(i) * log(P(i)/Q(i)))
            kl_div = np.sum(posterior_means_norm * np.log(posterior_means_norm / prior_means_norm))
            
            # Earth Mover's Distance (EMD)
            emd = np.sum(np.abs(np.cumsum(posterior_means_norm) - np.cumsum(prior_means_norm)))
            
            convergence_metrics[param] = {
                'kl_divergence': kl_div,
                'emd': emd,
                'prior_variance': np.var(prior_means),
                'posterior_variance': np.var(posterior_means)
            }
        else:
            # For scalar parameters
            # Calculate statistics
            prior_mean = np.mean(prior_samples[param])
            prior_var = np.var(prior_samples[param])
            posterior_mean = np.mean(param_distributions[param])
            posterior_var = np.var(param_distributions[param])
            
            # Variance reduction factor
            var_reduction = (prior_var - posterior_var) / prior_var if prior_var > 0 else np.nan
            
            # 95% HDI width reduction
            prior_hdi_width = np.percentile(prior_samples[param], 97.5) - np.percentile(prior_samples[param], 2.5)
            posterior_hdi_width = np.percentile(param_distributions[param], 97.5) - np.percentile(param_distributions[param], 2.5)
            hdi_reduction = (prior_hdi_width - posterior_hdi_width) / prior_hdi_width if prior_hdi_width > 0 else np.nan
            
            convergence_metrics[param] = {
                'prior_mean': prior_mean,
                'posterior_mean': posterior_mean,
                'prior_variance': prior_var,
                'posterior_variance': posterior_var,
                'variance_reduction': var_reduction,
                'hdi_width_reduction': hdi_reduction
            }
    
    return pd.DataFrame(convergence_metrics).T

def run_numerical_evaluation(samples_path, config_path, input_summaries_template_path, true_stats):
    """
    Run the numerical evaluation and print results
    """
    print("========== FORECAST METRICS ==========")
    forecast_metrics = calculate_forecast_metrics(
        input_summaries_template_path, 
        config_path, 
        true_stats
    )
    print(forecast_metrics)
    
    print("\n========== EVENT DETECTION METRICS ==========")
    event_metrics = evaluate_forecasts_for_specific_events(
        input_summaries_template_path, 
        config_path, 
        true_stats
    )
    print(event_metrics)
    
    print("\n========== PARAMETER CONVERGENCE METRICS ==========")
    convergence_metrics = evaluate_parameter_convergence(samples_path)
    print(convergence_metrics)
    
    # Return all metrics for possible saving to file
    return {
        'forecast_metrics': forecast_metrics,
        'event_metrics': event_metrics,
        'convergence_metrics': convergence_metrics
    }

def save_metrics_to_file(metrics, output_path_prefix):
    """
    Save metrics to CSV files
    """
    for metric_name, df in metrics.items():
        output_path = f"{output_path_prefix}_{metric_name}.csv"
        df.to_csv(output_path)
        print(f"Saved {metric_name} to {output_path}")

# Add this to the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_path', default='results/US/MA-20201111-20210111-20210211/posterior_samples.json', type=str)
    parser.add_argument('--config_path', default='results/US/MA-20201111-20210111-20210211/config_after_abc.json', type=str)
    parser.add_argument('--input_summaries_template_path', default='results/US/MA-20201111-20210111-20210211/summary_after_abc', type=str)
    parser.add_argument('--true_stats', default='datasets/US/MA-20201111-20210111-20210211/daily_counts.csv', type=str)
    parser.add_argument('--output_path_prefix', default='results/US/MA-20201111-20210111-20210211/numerical_evaluation', type=str)
    parser.add_argument('--save_metrics', action='store_true', help='Whether to save metrics to CSV files')
    args = parser.parse_args()

    # Original visualization code
    # plot_params(args.samples_path)
    # plot_forecasts(args.input_summaries_template_path, args.config_path, args.true_stats)
    
    # New numerical evaluation code
    metrics = run_numerical_evaluation(
        args.samples_path,
        args.config_path,
        args.input_summaries_template_path,
        args.true_stats
    )
    
    if args.save_metrics:
        save_metrics_to_file(metrics, args.output_path_prefix)