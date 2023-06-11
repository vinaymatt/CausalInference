import torch
import torch.nn as nn
from torch.optim import Adam
import gpytorch
import pandas as pd
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from scipy.optimize import linear_sum_assignment
from itertools import product
from sklearn.utils import resample
from scipy.stats import wasserstein_distance
import ot
import os

def compute_wasserstein(treated_outcomes, untreated_outcomes):
    n_treatment, n_control = treated_outcomes.shape[0], untreated_outcomes.shape[0]
    weights_treatment = np.ones(n_treatment) / n_treatment
    weights_control = np.ones(n_control) / n_control

    cost_matrix = ot.dist(treated_outcomes.detach().numpy(), untreated_outcomes.detach().numpy(), metric='euclidean')

    wasserstein_dist = ot.sinkhorn2(weights_treatment, weights_control, cost_matrix, reg=1e-1)

    return torch.tensor(wasserstein_dist, requires_grad=True, dtype=torch.float32)

# Create a folder to store the ATE and error results
results_folder_name = 'ate_and_error_results'
folder_name = 'simulation_results'
if not os.path.exists(results_folder_name):
    os.makedirs(results_folder_name)

ate_and_error_list = []
pred_treatment_effect_list = []
pehe_values = []
mse_list = []

for sim in range(1, 2):
    file_name = f'simulation_{sim}.xlsx'
    file_path = os.path.join(folder_name, file_name)

    data = pd.read_excel(file_path)

    unique_individuals = data['Individual'].unique()
    train_individuals = unique_individuals[:int(0.8 * len(unique_individuals))]
    test_individuals = unique_individuals[int(0.8 * len(unique_individuals)):]

    train_data = data[data['Individual'].isin(train_individuals)]
    test_data = data[data['Individual'].isin(test_individuals)]

    X_train = train_data.drop(columns=['Individual', 'T', 'Y', 'Y0', 'Y1']).values
    T_train = train_data['T'].values
    Y_train = train_data['Y'].values
    time_train = train_data['Time'].values
    Y1_test = test_data['Y1'].values
    Y0_test = test_data['Y0'].values

    X_test = test_data.drop(columns=['Individual', 'T', 'Y', 'Y0', 'Y1']).values
    T_test = test_data['T'].values
    Y_test = test_data['Y'].values
    time_test = test_data['Time'].values

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    T_train_tensor = torch.tensor(T_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    time_train_tensor = torch.tensor(time_train, dtype=torch.float32)

    train_data = TensorDataset(X_train_tensor, T_train_tensor, Y_train_tensor, time_train_tensor)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    class CFRModel(nn.Module):
        def __init__(self, input_size, representation_size=32):
            super(CFRModel, self).__init__()
            self.shared_repr = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, representation_size),
                nn.ReLU(),
            )
            self.treatment_head = nn.Sequential(
                nn.Linear(representation_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.control_head = nn.Sequential(
                nn.Linear(representation_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x, t):
            phi = self.shared_repr(x)
            treated_outcome = self.treatment_head(phi)
            control_outcome = self.control_head(phi)
            h = torch.where(t.view(-1, 1) == 1, treated_outcome, control_outcome)
            return h, phi


    class TimeGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, train_time, train_phi, likelihood):
            train_time = train_time.unsqueeze(-1)
            super(TimeGPModel, self).__init__(torch.cat((train_time, train_phi), dim=-1), train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

        def forward(self, x, time_batch, phi_batch):
            x = torch.cat((time_batch, phi_batch), dim=-1)
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def cfr_loss(y_pred, y_true, t, phi_batch, alpha, beta, gamma, gp_model, mll, output, y_pred_cf, lambda_wass):
        treatment_pred = y_pred[t == 1]
        treatment_true = y_true[t == 1]
        control_pred = y_pred[t == 0]
        control_true = y_true[t == 0]

        treatment_pred_mean = torch.mean(treatment_pred)
        control_pred_mean = torch.mean(control_pred)
        balance_term = torch.abs(treatment_pred_mean - control_pred_mean)

        factual_loss = torch.mean((treatment_true - treatment_pred) ** 2) + torch.mean(
            (control_true - control_pred) ** 2)

        gp_reg_term = -mll(output, y_true)

        y_pred_treated, _ = model(x_batch, torch.ones_like(t_batch))
        y_pred_untreated, _ = model(x_batch, torch.zeros_like(t_batch))
        wasserstein_term = compute_wasserstein(y_pred_treated, y_pred_untreated)

        loss = factual_loss + alpha * balance_term + beta * gp_reg_term + lambda_wass * wasserstein_term
        return loss

    input_size = X_train.shape[1]
    output_size = 1
    model = CFRModel(input_size, output_size)
    model.train()
    phi_list = []
    for x_batch, t_batch, y_batch, time_batch in train_loader:
        y_pred, phi_batch = model(x_batch, t_batch)
        phi_list.append(phi_batch)

    phi_train = torch.cat(phi_list, dim=0)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = TimeGPModel(X_train_tensor, Y_train_tensor, time_train_tensor, phi_train, likelihood)


    optimizer = Adam(list(model.parameters()) + list(gp_model.parameters()), lr=0.001)
    alpha = 0.1
    beta = 1.0
    gamma = 0.1
    lambda_wass = 0.1

    num_epochs = 100
    for epoch in range(num_epochs):
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        epoch_loss = 0.0
        for x_batch, t_batch, y_batch, time_batch in train_loader:
            optimizer.zero_grad()
            y_pred, phi_batch = model(x_batch, t_batch)

            gp_model.set_train_data(time_batch.unsqueeze(-1), y_pred, strict=False)

            output = gp_model(time_batch.unsqueeze(-1), time_batch, phi_batch)

            y_pred_treated, _ = model(x_batch, torch.ones_like(t_batch))
            y_pred_untreated, _ = model(x_batch, torch.zeros_like(t_batch))
            y_pred_cf = torch.stack([y_pred_untreated, y_pred_treated], dim=1).squeeze()

            loss = cfr_loss(y_pred, y_batch, t_batch, phi_batch, alpha, beta, gamma, gp_model, mll, output, y_pred_cf,
                            lambda_wass)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")

    model.eval()
    gp_model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    T_test_tensor = torch.tensor(T_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    time_test_tensor = torch.tensor(time_test, dtype=torch.float32)

    test_data = TensorDataset(X_test_tensor, T_test_tensor, Y_test_tensor, time_test_tensor)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    y_pred_list = []
    phi_list = []
    with torch.no_grad():
        for x_batch, t_batch, y_batch, time_batch in test_loader:
            y_pred, phi_batch = model(x_batch, t_batch)

            y_pred_list.append(y_pred)
            phi_list.append(phi_batch)

    y_pred_test = torch.cat(y_pred_list, dim=0)
    phi_test = torch.cat(phi_list, dim=0)

    mse = torch.mean((y_pred_test - Y_test_tensor) ** 2)
    print(f"Mean Squared Error: {mse.item()}")
    mse_list.append(mse.item())

    def estimate_ate(y_pred, y_true, t):
        treated_outcomes = y_pred[t == 1]
        untreated_outcomes = y_pred[t == 0]
        treated_true = y_true[t == 1]
        untreated_true = y_true[t == 0]
        ate = torch.mean(treated_true) - torch.mean(untreated_true)
        return ate


    def error_in_ate(y_pred_treatment, y_pred_control, m1, m0):
        n = len(y_pred_treatment)
        ate_pred = y_pred_treatment - y_pred_control
        ate_true = torch.mean(m1) - torch.mean(m0)
        error = torch.abs(torch.sum(ate_pred - ate_true) / n)
        return error

    with torch.no_grad():
        y_pred_treated, _ = model(X_test_tensor, torch.ones_like(T_test_tensor))
        y_pred_untreated, _ = model(X_test_tensor, torch.zeros_like(T_test_tensor))

    pred_treatment_effect = y_pred_treated - y_pred_untreated
    pred_treatment_effect_list.append(pred_treatment_effect.numpy())

    y_pred_cf = torch.stack([y_pred_untreated, y_pred_treated], dim=1)

    ate = estimate_ate(y_pred_test, Y_test_tensor, T_test_tensor)
    error = error_in_ate(y_pred_treated, y_pred_untreated, Y_test_tensor[T_test_tensor == 1],
                         Y_test_tensor[T_test_tensor == 0])

    ate_and_error_list.append((ate.item(), error.item()))
    true_treatment_effect_test = Y1_test - Y0_test
    squared_error = (pred_treatment_effect - true_treatment_effect_test) ** 2

    true_ate_test = np.mean(Y1_test - Y0_test)

    pehe = torch.mean(squared_error).item()
    pehe_values.append(pehe)

with open(os.path.join(results_folder_name, 'ate_and_error_values5.txt'), 'w') as f:
    for ate, error in ate_and_error_list:
        f.write(f'{ate} {error}\n')


with open(os.path.join(results_folder_name, 'pehe_values1.txt'), 'w') as f:
    for pehe in pehe_values:
        f.write(f'{pehe}\n')


with open(os.path.join(results_folder_name, 'mse_values1.txt'), 'w') as f:
    for mse in mse_list:
        f.write(f'{mse}\n')


