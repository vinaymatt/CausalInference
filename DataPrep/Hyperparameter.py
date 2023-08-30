import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import gpytorch
import pandas as pd
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.model_selection import KFold
from tqdm import tqdm

data = pd.read_excel('synthetic_data_with_infotreat.xlsx')

unique_individuals = data['Individual'].unique()
train_individuals = unique_individuals[:int(0.8 * len(unique_individuals))]
test_individuals = unique_individuals[int(0.8 * len(unique_individuals)):]

train_data = data[data['Individual'].isin(train_individuals)]
test_data = data[data['Individual'].isin(test_individuals)]

X_train = train_data.drop(columns=['Individual', 'Time', 'T', 'Y', 'Y0', 'Y1']).values
T_train = train_data['T'].values
Y_train = train_data['Y'].values
time_train = train_data['Time'].values

X_test = test_data.drop(columns=['Individual', 'Time', 'T', 'Y', 'Y0', 'Y1']).values
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
    def __init__(self, input_size, output_size):
        super(CFRModel, self).__init__()
        self.shared_repr = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.outcome_function = nn.Linear(32, output_size)

    def forward(self, x, t):
        phi = self.shared_repr(x)
        h = self.outcome_function(phi)
        return h

class TimeGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(TimeGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def cfr_loss(y_pred, y_true, t, alpha, beta, gp_model, likelihood, output):
    treatment_pred = y_pred[t == 1]
    treatment_true = y_true[t == 1]
    control_pred = y_pred[t == 0]
    control_true = y_true[t == 0]

    treatment_pred_mean = torch.mean(treatment_pred)
    control_pred_mean = torch.mean(control_pred)
    balance_term = treatment_pred_mean - control_pred_mean

    factual_loss = torch.mean((treatment_true - treatment_pred) ** 2) + torch.mean((control_true - control_pred) ** 2)

    mll = ExactMarginalLogLikelihood(likelihood, gp_model)
    gp_reg_term = -mll(output, y_true)
    
    loss = factual_loss + alpha * balance_term + beta * gp_reg_term
    return loss


def train_and_evaluate(alpha, beta, kernel, X_train_fold_tensor, T_train_fold_tensor, Y_train_fold_tensor, time_train_fold_tensor):

    input_size = X_train.shape[1]
    output_size = 1
    model = CFRModel(input_size, output_size)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = TimeGPModel(time_train_fold_tensor.unsqueeze(-1), Y_train_fold_tensor, likelihood, kernel)

    optimizer = Adam(list(model.parameters()) + list(gp_model.parameters()), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        for x_batch, t_batch, y_batch, time_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(x_batch, t_batch)
            gp_model.set_train_data(time_batch.unsqueeze(-1), y_batch, strict=False)  # Update GP model's training inputs
            gp_model.train()
            likelihood.train()
            output = gp_model(time_batch.unsqueeze(-1))

            loss = cfr_loss(y_pred, y_batch, t_batch, alpha, beta, gp_model, likelihood, output)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        model.eval()
        y_pred = model(torch.tensor(X_test, dtype=torch.float32), torch.tensor(T_test, dtype=torch.float32))
        mse = torch.mean((torch.tensor(Y_test, dtype=torch.float32) - y_pred) ** 2)

    return mse.item()


kernels = [gpytorch.kernels.RBFKernel(), gpytorch.kernels.MaternKernel(nu=1.5), gpytorch.kernels.MaternKernel(nu=2.5)]
alphas = [0.1, 1.0, 10.0]
betas = [0.1, 1.0, 10.0]

best_kernel = None
best_alpha = None
best_beta = None
best_mse = float('inf')

kf = KFold(n_splits=5)

for kernel in tqdm(kernels, desc="Kernels"):
    for alpha in tqdm(alphas, desc="Alphas", leave=False):
        for beta in tqdm(betas, desc="Betas", leave=False):
            mse_sum = 0
            for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                T_train_fold, T_val_fold = T_train[train_index], T_train[val_index]
                Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[val_index]
                time_train_fold, time_val_fold = time_train[train_index], time_train[val_index]

                X_train_fold_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
                T_train_fold_tensor = torch.tensor(T_train_fold, dtype=torch.float32)
                Y_train_fold_tensor = torch.tensor(Y_train_fold, dtype=torch.float32)
                time_train_fold_tensor = torch.tensor(time_train_fold, dtype=torch.float32)

                mse_sum += train_and_evaluate(alpha, beta, kernel, X_train_fold_tensor, T_train_fold_tensor, Y_train_fold_tensor, time_train_fold_tensor)

            mse_avg = mse_sum / 5

            if mse_avg < best_mse:
                best_kernel = kernel
                best_alpha = alpha
                best_beta = beta
                best_mse = mse_avg

print('Best kernel:', best_kernel)
print('Best alpha:', best_alpha)
print('Best beta:', best_beta)
print('Best MSE:', best_mse)
