import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Read the Excel file
df = pd.read_excel('synthetic_data_with_info.xlsx')

# Extract columns as NumPy arrays
X_np = df[[f'X{i + 1}' for i in range(30)]].values
T_np = df['T'].values.reshape(-1, 1)
Y_np = df['Y'].values.reshape(-1, 1)

# Define the number of individuals, observations, and covariates
n_individuals = 40
n_observations = 20
n_covariates = 30

# Convert NumPy arrays to tensors
X = torch.tensor(X_np, dtype=torch.float32).view(n_individuals, n_observations, n_covariates)
T = torch.tensor(T_np, dtype=torch.float32).view(n_individuals, n_observations, 1)
Y = torch.tensor(Y_np, dtype=torch.float32).view(n_individuals, n_observations, 1)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out


# Combine X and T as input features
XT = torch.cat([X, T], dim=-1)

# LSTM parameters
input_size = 31
hidden_size = 50
num_layers = 10

# Train the LSTM network
lstm = LSTM(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.001)

num_epochs = 3000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    lstm_out = lstm(XT)
    loss = criterion(lstm_out, Y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Convert LSTM output to numpy array
lstm_out_np = lstm_out.squeeze().detach().numpy().reshape(-1, 1)

# Standardize the LSTM output
scaler = StandardScaler()
lstm_out_scaled = scaler.fit_transform(lstm_out_np)

# Fit a logistic regression model to estimate propensity scores
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(lstm_out_scaled, T_np.ravel())

# Compute propensity scores
propensity_scores = logistic_model.predict_proba(lstm_out_scaled)[:, 1].reshape(-1, 1)

# IPTW
weights = T_np / propensity_scores - (1 - T_np) / (1 - propensity_scores)
weighted_outcome = Y_np * weights

# Compute the ATE
ATE = np.mean(weighted_outcome)
print("Estimated ATE:", ATE)

