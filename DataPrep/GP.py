import numpy as np
import pandas as pd
import GPy
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from scipy.stats import norm

def log_odds_ratio(gamma, propensity_score):
    return np.log((propensity_score * (1 - propensity_score * gamma)) / (gamma * (1 - propensity_score) * propensity_score))

def sensitivity_analysis(treated_indices, control_indices, gamma_range, matched_propensity_scores, individual_treatment_effects):
    results = []
    for gamma in gamma_range:
        log_OR = log_odds_ratio(gamma, matched_propensity_scores)
        W = norm.cdf(log_OR)  # Calculate weights for each observation
        treated_weights = W[treated_indices]
        control_weights = W[control_indices]
        weighted_treated_avg = np.average(individual_treatment_effects.squeeze(), weights=treated_weights.squeeze())
        weighted_control_avg = np.average(individual_treatment_effects.squeeze(), weights=control_weights.squeeze())
        weighted_ATE = weighted_treated_avg - weighted_control_avg
        results.append((gamma, weighted_ATE))
    return results


df = pd.read_excel('synthetic_data_with_infotreat.xlsx')

X_covariates = df[[f'X{i + 1}' for i in range(30)]].values
X_other = df[['Individual', 'Time', 'T']].values
X = np.hstack([X_other, X_covariates])
y = df['Y'].values

# Estimate propensity scores
log_reg = LogisticRegression()
log_reg.fit(X_covariates, X[:, 2])
propensity_scores = log_reg.predict_proba(X_covariates)[:, 1]

# Perform propensity score matching
df['propensity_score'] = propensity_scores
df_treated = df[df['T'] == 1]
df_control = df[df['T'] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(df_control['propensity_score'].values.reshape(-1, 1))

_, indices = nn.kneighbors(df_treated['propensity_score'].values.reshape(-1, 1))
matched_control_indices = indices.flatten()

matched_control = df_control.iloc[matched_control_indices]

matched_data = pd.concat([df_treated, matched_control])

matern_kernel = GPy.kern.Matern32(input_dim=33, active_dims=[1])  # Temporal correlation
rbf_kernel = GPy.kern.RBF(input_dim=33, active_dims=[0])  # Multilevel correlation
treatment_kernel = GPy.kern.Linear(input_dim=1, active_dims=[2])  # Treatment effect

# Split matched_data into train and test sets based on individuals
unique_individuals = matched_data['Individual'].unique()
train_individuals, test_individuals = train_test_split(unique_individuals, test_size=0.2, random_state=42)

train_data = matched_data[matched_data['Individual'].isin(train_individuals)]
test_data = matched_data[matched_data['Individual'].isin(test_individuals)]

# Train GPR model on the train set
X_train = train_data[['Individual', 'Time', 'T']].values
X_train = np.hstack([X_train, train_data[[f'X{i + 1}' for i in range(30)]].values])
y_train = train_data['Y'].values

kernel = matern_kernel * rbf_kernel + treatment_kernel
model = GPy.models.GPRegression(X_train, y_train.reshape(-1, 1), kernel)
model.optimize()

# Estimate counterfactual outcomes for the test set
X_test = test_data[['Individual', 'Time', 'T']].values
X_test = np.hstack([X_test, test_data[[f'X{i + 1}' for i in range(30)]].values])
y_test = test_data['Y'].values

X_cf1_test = X_test.copy()
X_cf1_test[:, 2] = 1
X_cf0_test = X_test.copy()
X_cf0_test[:, 2] = 0

Y1_pred_test, _ = model.predict(X_cf1_test)
Y0_pred_test, _ = model.predict(X_cf0_test)

# Calculate individual treatment effects for the test set
individual_treatment_effects_test = Y1_pred_test - Y0_pred_test

# Use the true potential outcomes (Y0 and Y1) to calculate the true treatment effects for the test set
true_treatment_effects_test = test_data['Y1'].values - test_data['Y0'].values

# Calculate the performance metric as the difference between the estimated treatment effects and the true treatment effects
performance_metric = np.mean(np.abs(true_treatment_effects_test - individual_treatment_effects_test))

print(f"Performance Metric (test set): {performance_metric}")