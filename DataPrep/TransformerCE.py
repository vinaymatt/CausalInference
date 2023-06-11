import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

class CausalEffectDataset(Dataset):
    def __init__(self, X, T, Y):
        self.X = X
        self.T = T
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        t = self.T[idx]
        y = self.Y[idx]
        return x, torch.tensor(t, dtype=torch.float32), y

class CausalEffectTransformer(nn.Module):
    def __init__(self, n_covariates, pretrained_model_name="roberta-base"):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name)
        self.fc_in = nn.Linear(n_covariates + 1, self.pretrained_model.config.hidden_size)
        self.fc_out = nn.Linear(self.pretrained_model.config.hidden_size, 1)

    def forward(self, x, t):
        x = self.fc_in(torch.cat((x, t.unsqueeze(-1)), dim=-1))
        x = x.unsqueeze(1)
        x = self.pretrained_model(inputs_embeds=x).last_hidden_state[:, 0, :]
        x = self.fc_out(x)
        return x

class CausalInferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, t):
        treated = (t == 1).float()
        control = (t == 0).float()
        y1 = y_pred * treated
        y0 = y_pred * control
        loss = (y_true - y1 - y0).pow(2).mean()
        return loss

def main():
    # Load the synthetic data
    df = pd.read_excel('synthetic_data_with_infotreat3.xlsx')

    # Normalize the continuous covariates
    scaler = StandardScaler()
    X = scaler.fit_transform(df.iloc[:, 2:32])

    # Normalize the time attribute
    time_scaler = StandardScaler()
    time = time_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    # Concatenate the time attribute with the existing covariates
    X = np.hstack((X, time))

    # Train-test split
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, df['T'].values, df['Y'].values, test_size=0.2, random_state=42)

    # Create datasets and data loaders
    train_dataset = CausalEffectDataset(X_train, T_train, Y_train)
    test_dataset = CausalEffectDataset(X_test, T_test, Y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model, loss function, and optimizer
    n_covariates = 31
    pretrained_model_name = "roberta-base"
    model = CausalEffectTransformer(n_covariates, pretrained_model_name=pretrained_model_name).to(device)
    criterion = CausalInferenceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_dataloader:
            x, t, y = batch
            x, t, y = x.to(device).float(), t.to(device).float(), y.to(device).float()
            optimizer.zero_grad()
            y_pred = model(x, t)
            loss = criterion(y_pred, y, t)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                x, t, y = batch
                x, t, y = x.to(device).float(), t.to(device).float(), y.to(device).float()
                y_pred = model(x, t)
                loss = criterion(y_pred, y, t)
                running_test_loss += loss.item()

        print(
            f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_dataloader):.4f}, Test Loss: {running_test_loss / len(test_dataloader):.4f}")

    # Estimate causal effects
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        T_tensor = torch.tensor(df['T'].values, dtype=torch.float32).to(device)
        Y1_pred = model(X_tensor, torch.ones_like(T_tensor).to(device)).cpu().numpy()
        Y0_pred = model(X_tensor, torch.zeros_like(T_tensor).to(device)).cpu().numpy()

    # Compute the average treatment effect (ATE) and average treatment effect on the treated (ATT)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    T_test_tensor = torch.tensor(T_test, dtype=torch.float32).to(device)
    Y1_pred_test = model(X_test_tensor, torch.ones_like(T_test_tensor).to(device)).detach().cpu().numpy()
    Y0_pred_test = model(X_test_tensor, torch.zeros_like(T_test_tensor).to(device)).detach().cpu().numpy()
    ITE_test = Y1_pred_test - Y0_pred_test
    ATE = np.mean(ITE_test)
    ATT = np.mean(ITE_test[T_test == 1])

    print("Estimated Average Treatment Effect (ATE):", ATE)
    print("Estimated Average Treatment Effect on the Treated (ATT):", ATT)


if __name__ == '__main__':
    main()

