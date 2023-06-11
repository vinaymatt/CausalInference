import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 100),
            nn.Tanh(),
            nn.Dropout(0.7),
            nn.BatchNorm1d(100),
            nn.Linear(100, 30),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)

class OutcomeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(31, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
        )

    def forward(self, x, t):
        xt = torch.cat((x, t), dim=1)
        return self.layers(xt)

def dgp(n_individuals=40, n_observations=20, n_clusters=2, n_base_features=10, n_covariates=30, ar1_decay=0.9, time_steps=None):

    base_features = torch.rand((n_individuals * n_observations, n_base_features))

    encoder = Encoder()
    X = encoder(base_features)

    propensity_scores = torch.sigmoid(X[:, 0] + X[:, 1] + X[:, 2])
    T = torch.bernoulli(propensity_scores).unsqueeze(1)

    Y1_network = OutcomeNetwork()
    Y0_network = OutcomeNetwork()
    Y1 = Y1_network(X, T)
    Y0 = Y0_network(X, 1 - T)

    Y = T * Y1 + (1 - T) * Y0

    sigma = torch.zeros((n_individuals * n_observations, n_individuals * n_observations))
    for i in range(n_individuals):
        for j in range(n_observations):
            for k in range(n_observations):
                time_diff = abs(time_steps[i * n_observations + j] - time_steps[i * n_observations + k]).item()
                sigma[i * n_observations + j, i * n_observations + k] = torch.tensor(ar1_decay, dtype=torch.float32) ** torch.tensor(time_diff, dtype=torch.float32)

    cluster_ids = np.random.randint(0, n_clusters, n_individuals)
    cluster_corr_matrix = torch.zeros_like(sigma)
    for i in range(n_individuals):
        for j in range(n_individuals):
            if cluster_ids[i] == cluster_ids[j]:
                cluster_corr_matrix[i * n_observations:(i + 1) * n_observations,
                j * n_observations:(j + 1) * n_observations] = 1

    longitudinal_sigma = sigma + 1e-6 * torch.eye(n_individuals * n_observations)
    cluster_sigma = cluster_corr_matrix + 1e-6 * torch.eye(n_individuals * n_observations)

    combined_sigma = torch.maximum(longitudinal_sigma, cluster_sigma)
    L = torch.linalg.cholesky(combined_sigma)

    error = torch.matmul(L, torch.randn((n_individuals * n_observations, 1)))

    Y += error
    Y1 += error
    Y0 += error

    X = X.view(n_individuals, n_observations, n_covariates)
    T = T.view(n_individuals, n_observations, 1)
    Y = Y.view(n_individuals, n_observations, 1)
    Y1 = Y1.view(n_individuals, n_observations, 1)
    Y0 = Y0.view(n_individuals, n_observations, 1)

    return X, T, Y, Y1, Y0

folder_name = 'simulation_results'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

ate_list = []

for sim in range(1, 1001):
    individual_number = np.repeat(np.arange(1, 41), 20)[:, np.newaxis]
    time_steps = np.random.poisson(lam=5, size=(40, 20))
    time_steps = np.cumsum(time_steps, axis=1)
    time_steps = time_steps.flatten()[:, np.newaxis]

    X, T, Y, Y1, Y0 = dgp(time_steps=time_steps)

    X_np = X.detach().numpy().reshape(-1, 30)
    T_np = T.detach().numpy().reshape(-1, 1)
    Y_np = Y.detach().numpy().reshape(-1, 1)

    Y1_np = Y1.detach().numpy().reshape(-1, 1)
    Y0_np = Y0.detach().numpy().reshape(-1, 1)

    data = np.hstack((individual_number, time_steps, X_np, T_np, Y_np, Y0_np, Y1_np))

    column_names = ['Individual', 'Time'] + [f'X{i + 1}' for i in range(30)] + ['T', 'Y', 'Y0', 'Y1']

    df = pd.DataFrame(data, columns=column_names)
    file_name = f'simulation_{sim}.xlsx'
    df.to_excel(os.path.join(folder_name, file_name), index=False)

    Y0_df = df['Y0'].values.reshape(-1, 20)
    Y1_df = df['Y1'].values.reshape(-1, 20)
    T_df = df['T'].values.reshape(-1, 20)

    individual_treatment_effect = Y1_df - Y0_df

    ATE = np.mean(individual_treatment_effect)
    print("Average Treatment Effect (ATE):", ATE)
    ate_list.append(ATE)

with open('ate_values.txt', 'w') as f:
    for ate in ate_list:
        f.write(f'{ate}\n')

individual_number = np.repeat(np.arange(1, 41), 20)[:, np.newaxis]
time_steps = np.random.poisson(lam=5, size=(40, 20))
time_steps = np.cumsum(time_steps, axis=1)
time_steps = time_steps.flatten()[:, np.newaxis]

X, T, Y, Y1, Y0 = dgp(time_steps=time_steps)

X_np = X.detach().numpy().reshape(-1, 30)
T_np = T.detach().numpy().reshape(-1, 1)
Y_np = Y.detach().numpy().reshape(-1, 1)

Y1_np = Y1.detach().numpy().reshape(-1, 1)
Y0_np = Y0.detach().numpy().reshape(-1, 1)

data = np.hstack((individual_number, time_steps, X_np, T_np, Y_np, Y0_np, Y1_np))

column_names = ['Individual', 'Time'] + [f'X{i + 1}' for i in range(30)] + ['T', 'Y', 'Y0', 'Y1']

df = pd.DataFrame(data, columns=column_names)
df.to_excel('synthetic_data_with_infotreat5.xlsx', index=False)

Y0_df = df['Y0'].values.reshape(-1, 20)
Y1_df = df['Y1'].values.reshape(-1, 20)
T_df = df['T'].values.reshape(-1, 20)

individual_treatment_effect = Y1_df - Y0_df

ATE = np.mean(individual_treatment_effect)

treated_individuals = individual_treatment_effect[T_df == 1]
ATT = np.mean(treated_individuals)

print("Average Treatment Effect (ATE):", ATE)
print("Average Treatment Effect on the Treated (ATT):", ATT)



