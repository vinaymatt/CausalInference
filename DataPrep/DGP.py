import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os

# Define the network architectures
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
            nn.Linear(31, 100),  # Change the input size to 31 to include the treatment
            nn.Tanh(),
            nn.Linear(100, 1),
        )

    def forward(self, x, t):
        xt = torch.cat((x, t), dim=1)  # Concatenate x and t along columns
        return self.layers(xt)

def dgp(n_individuals=40, n_observations=20, n_clusters=2, n_base_features=10, n_covariates=30, ar1_decay=0.9, time_steps=None):
    # Generate base features
    base_features = torch.rand((n_individuals * n_observations, n_base_features))

    # Compute the covariate matrix X using the encoder network
    encoder = Encoder()
    X = encoder(base_features)

    # Assign treatment based on a propensity score function
    propensity_scores = torch.sigmoid(X[:, 0] + X[:, 1] + X[:, 2])  # Example: logistic regression model based on the first three covariates
    T = torch.bernoulli(propensity_scores).unsqueeze(1)

    # Generate potential outcomes using two separate networks
    Y1_network = OutcomeNetwork()
    Y0_network = OutcomeNetwork()
    Y1 = Y1_network(X, T)  # Update to include treatment
    Y0 = Y0_network(X, 1 - T)  # Update to include treatment

    # Compute observed outcomes
    Y = T * Y1 + (1 - T) * Y0

    # Simulate longitudinal correlation
    sigma = torch.zeros((n_individuals * n_observations, n_individuals * n_observations))
    for i in range(n_individuals):
        for j in range(n_observations):
            for k in range(n_observations):
                # Use time_steps to compute the difference between time points
                time_diff = abs(time_steps[i * n_observations + j] - time_steps[i * n_observations + k]).item()
                sigma[i * n_observations + j, i * n_observations + k] = torch.tensor(ar1_decay, dtype=torch.float32) ** torch.tensor(time_diff, dtype=torch.float32)

    # Simulate multilevel correlation
    cluster_ids = np.random.randint(0, n_clusters, n_individuals)
    cluster_corr_matrix = torch.zeros_like(sigma)
    for i in range(n_individuals):
        for j in range(n_individuals):
            if cluster_ids[i] == cluster_ids[j]:
                cluster_corr_matrix[i * n_observations:(i + 1) * n_observations,
                j * n_observations:(j + 1) * n_observations] = 1

    # Ensure positive-definiteness
    longitudinal_sigma = sigma + 1e-6 * torch.eye(n_individuals * n_observations)
    cluster_sigma = cluster_corr_matrix + 1e-6 * torch.eye(n_individuals * n_observations)

    # Combine covariance matrices
    combined_sigma = torch.maximum(longitudinal_sigma, cluster_sigma)
    L = torch.linalg.cholesky(combined_sigma)

    error = torch.matmul(L, torch.randn((n_individuals * n_observations, 1)))

    # Add error t the observed outcomes
    Y += error
    Y1 += error
    Y0 += error

    # Reshape the data
    X = X.view(n_individuals, n_observations, n_covariates)
    T = T.view(n_individuals, n_observations, 1)
    Y = Y.view(n_individuals, n_observations, 1)
    Y1 = Y1.view(n_individuals, n_observations, 1)
    Y0 = Y0.view(n_individuals, n_observations, 1)

    return X, T, Y, Y1, Y0

# Create a folder to store the simulation results
folder_name = 'simulation_results'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

ate_list = []

# Run 50 simulations
for sim in range(1, 1001):
    # Individual number and time step
    individual_number = np.repeat(np.arange(1, 41), 20)[:, np.newaxis]
    # Generate unique time steps for each simulation
    time_steps = np.random.poisson(lam=5, size=(40, 20))
    time_steps = np.cumsum(time_steps, axis=1)  # Cumulative sum to make time steps sequential
    time_steps = time_steps.flatten()[:, np.newaxis]

    # Generate synthetic data
    X, T, Y, Y1, Y0 = dgp(time_steps=time_steps)

    # Convert tensors to NumPy arrays and reshape
    X_np = X.detach().numpy().reshape(-1, 30)
    T_np = T.detach().numpy().reshape(-1, 1)
    Y_np = Y.detach().numpy().reshape(-1, 1)

    # Counterfactual outcomes
    Y1_np = Y1.detach().numpy().reshape(-1, 1)
    Y0_np = Y0.detach().numpy().reshape(-1, 1)

    # Combine arrays into a single array
    data = np.hstack((individual_number, time_steps, X_np, T_np, Y_np, Y0_np, Y1_np))

    # Create column names
    column_names = ['Individual', 'Time'] + [f'X{i + 1}' for i in range(30)] + ['T', 'Y', 'Y0', 'Y1']

    # Create DataFrame and save as Excel file
    df = pd.DataFrame(data, columns=column_names)
    file_name = f'simulation_{sim}.xlsx'
    df.to_excel(os.path.join(folder_name, file_name), index=False)

    # Extract potential outcomes from the data frame
    Y0_df = df['Y0'].values.reshape(-1, 20)
    Y1_df = df['Y1'].values.reshape(-1, 20)
    T_df = df['T'].values.reshape(-1, 20)

    # Compute the treatment effect for each individual and time step
    individual_treatment_effect = Y1_df - Y0_df

    # Compute the average treatment effect (ATE)
    ATE = np.mean(individual_treatment_effect)
    print("Average Treatment Effect (ATE):", ATE)
    ate_list.append(ATE)

# Save ATEs to a text file
with open('ate_values.txt', 'w') as f:
    for ate in ate_list:
        f.write(f'{ate}\n')

# Individual number and time step
individual_number = np.repeat(np.arange(1, 41), 20)[:, np.newaxis]
time_steps = np.random.poisson(lam=5, size=(40, 20))
time_steps = np.cumsum(time_steps, axis=1)  # Cumulative sum to make time steps sequential
time_steps = time_steps.flatten()[:, np.newaxis]

# Generate synthetic data
X, T, Y, Y1, Y0 = dgp(time_steps=time_steps)

# Convert tensors to NumPy arrays and reshape
X_np = X.detach().numpy().reshape(-1, 30)
T_np = T.detach().numpy().reshape(-1, 1)
Y_np = Y.detach().numpy().reshape(-1, 1)

# Counterfactual outcomes
Y1_np = Y1.detach().numpy().reshape(-1, 1)
Y0_np = Y0.detach().numpy().reshape(-1, 1)


# Combine arrays into a single array
data = np.hstack((individual_number, time_steps, X_np, T_np, Y_np, Y0_np, Y1_np))

# Create column names
column_names = ['Individual', 'Time'] + [f'X{i + 1}' for i in range(30)] + ['T', 'Y', 'Y0', 'Y1']

# Create DataFrame and save as Excel file
df = pd.DataFrame(data, columns=column_names)
df.to_excel('synthetic_data_with_infotreat5.xlsx', index=False)

# Extract potential outcomes from the data frame
Y0_df = df['Y0'].values.reshape(-1, 20)
Y1_df = df['Y1'].values.reshape(-1, 20)
T_df = df['T'].values.reshape(-1, 20)

# Compute the treatment effect for each indiv
# dual and time step
individual_treatment_effect = Y1_df - Y0_df

# Compute the average treatment effect (ATE)
ATE = np.mean(individual_treatment_effect)

# Compute the average treatment effect on the treated (ATT)
treated_individuals = individual_treatment_effect[T_df == 1]
ATT = np.mean(treated_individuals)

print("Average Treatment Effect (ATE):", ATE)
print("Average Treatment Effect on the Treated (ATT):", ATT)



