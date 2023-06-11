import numpy as np
import pandas as pd
import GPy
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

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

# Train GPR model on the matched dataset
X_matched = matched_data[['Individual', 'Time', 'T']].values
X_matched = np.hstack([X_matched, matched_data[[f'X{i + 1}' for i in range(30)]].values])
y_matched = matched_data['Y'].values

matern_kernel = GPy.kern.Matern32(input_dim=1, active_dims=[1])  # Temporal correlation
rbf_kernel = GPy.kern.RBF(input_dim=1, active_dims=[0])  # Multilevel correlation
treatment_kernel = GPy.kern.Linear(input_dim=1, active_dims=[2])  # Treatment effect

kernel = matern_kernel * rbf_kernel + treatment_kernel

model = GPy.models.GPRegression(X_matched, y_matched.reshape(-1, 1), kernel)
model.optimize()

X_cf1 = X_matched.copy()
X_cf1[:, 2] = 1
X_cf0 = X_matched.copy()
X_cf0[:, 2] = 0

Y1_pred, _ = model.predict(X_cf1)
Y0_pred, _ = model.predict(X_cf0)

individual_treatment_effects = Y1_pred - Y0_pred

treated_individuals = individual_treatment_effects[X_matched[:, 2] == 1]
ATT = np.mean(treated_individuals)
ATE = np.mean(individual_treatment_effects)

# Calculate c_phi using true values of Y1 and Y0
Y1_true = df['Y1'].values
Y0_true = df['Y0'].values
true_individual_treatment_effects = Y1_true - Y0_true
phi = X_covariates[:, 0]  # Assuming Φ is the first covariate X1

c_phi = np.corrcoef(true_individual_treatment_effects, phi)[0, 1]

print(f"ATT: {ATT}, ATE: {ATE}, c_phi: {c_phi}")
