# ACED-HMM Hospitalized Patient Trajectory Model

This repository contains the implementation of the ACED-HMM model for analyzing hospitalized patient trajectories. The model is trained using Approximate Bayesian Computation (ABC) with Markov Chain Monte Carlo (MCMC) and provides tools for data preparation, model training, posterior visualization, forecasting, and forecast evaluation.

## Getting Started

To get started with this repository, follow the steps below:

### 1. Install Dependencies

Before running any code, ensure that all required Python packages are installed. The dependencies are listed in the `requirements.txt` file. Use the following command to install them:

```bash
pip install -r requirements.txt
```

This will ensure that all necessary libraries are available for running the pipeline.

### 2. Run the Full Pipeline

The `getting_started_full_pipeline.ipynb` notebook provides a comprehensive walkthrough of the major functionalities of this repository. It includes:

1. **Data Preparation**: Generating state-level datasets using the provided data collection and preparation scripts.
2. **Model Training**: Fitting the posterior distribution of ACED-HMM parameters using ABC-MCMC.
3. **Visualization**: Visualizing prior and posterior distributions of model parameters.
4. **Forecasting**: Running forecasts using the trained posterior.
5. **Evaluation**: Summarizing and visualizing forecast results.

To run the notebook:

1. Open `getting_started_full_pipeline.ipynb` in Jupyter Notebook or JupyterLab.
2. Follow the instructions in each cell to execute the pipeline step by step.

### Notes

- The notebook uses example data from the state of Massachusetts with specific training and testing windows. You can modify the paths and parameters in the notebook to work with different states or time periods.
- For demonstration purposes, the training process in the notebook is configured to run for the full number of iterations. For faster training, decrease the number of iterations.
