import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, BertTokenizer

# Load and preprocess data
df = pd.read_excel('synthetic_data_with_infotreat3.xlsx')

# Preprocess the data
def preprocess_data(df, tokenizer):
    df = df.copy()

    # Split into features and labels
    X = df.iloc[:, 2:32]
    y = df['Y'].values

    # Tokenize the features
    input_ids, attention_masks = [], []

    for record in X.values:
        encoded = tokenizer.encode_plus(
            str(record),
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    y = torch.tensor(y)

    return input_ids, attention_masks, y

# Split the data into train and validation sets
def split_data(input_ids, attention_masks, y, random_state=42):
    X_train_ids, X_val_ids, X_train_masks, X_val_masks, y_train, y_val = train_test_split(input_ids, attention_masks, y, test_size=0.2, random_state=random_state)
    return X_train_ids, X_val_ids, X_train_masks, X_val_masks, y_train, y_val

# Define custom Dataset class for your EHR data
class EHRDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

# Custom T-BERT model
class CustomTBERT(nn.Module):
    def __init__(self, config):
        super(CustomTBERT, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.propensity_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )

        self.outcome_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        propensity_scores = self.propensity_head(pooled_output)
        outcome_predictions = self.outcome_head(pooled_output)

        return propensity_scores, outcome_predictions

# Define your training and validation functions
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        input_ids, attention_masks, labels = batch
        input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        propensity_scores, outcome_predictions = model(input_ids, attention_masks)
        loss = criterion(outcome_predictions, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks, labels = batch
            input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)
            labels = labels.to(device).unsqueeze(1)

            propensity_scores, outcome_predictions = model(input_ids, attention_masks)
            loss = criterion(outcome_predictions, labels)

            running_loss += loss.item()

    return running_loss / len(dataloader)


def estimate_ate(y_true, propensity_scores, outcome_preds, treatment_col):
    treated = (treatment_col == 1)
    untreated = (treatment_col == 0)

    treatment_outcomes = outcome_preds[treated].mean()
    control_outcomes = outcome_preds[untreated].mean()
    weighted_outcomes = outcome_preds - propensity_scores
    weighted_control_outcomes = weighted_outcomes[untreated].mean()

    ate = treatment_outcomes - control_outcomes
    att = treatment_outcomes - weighted_control_outcomes

    return ate.item(), att.item()

def main():
    # Tokenizer and model initialization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig()

    # Preprocess and split the data
    input_ids, attention_masks, y = preprocess_data(df, tokenizer)
    X_train_ids, X_val_ids, X_train_masks, X_val_masks, y_train, y_val = split_data(input_ids, attention_masks, y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize your custom T-BERT model
    config = BertConfig()
    model = CustomTBERT(config)
    model.to(device)

    # Set your hyperparameters
    learning_rate = 1e-4
    epochs = 30
    batch_size = 64

    # Create data loaders
    train_dataset = EHRDataset(X_train_ids, X_train_masks, y_train)
    val_dataset = EHRDataset(X_val_ids, X_val_masks, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define optimizer and loss functions
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_dataloader, criterion, device)
        print(f'Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}')

    # Estimate causal effect
    model.eval()
    with torch.no_grad():
        propensity_scores, outcome_predictions = model(X_val_ids.to(device), X_val_masks.to(device))
        propensity_scores = propensity_scores.squeeze().cpu()
        outcome_predictions = outcome_predictions.squeeze().cpu()

    treatment_col = df['treatment'].values
    _, _, _, _, _, y_val_treatment = train_test_split(input_ids, attention_masks, y, treatment_col, test_size=0.2,
                                                      random_state=42)
    ate, att = estimate_ate(y_val, propensity_scores, outcome_predictions, y_val_treatment)
    print(f'ATE: {ate:.4f} - ATT: {att:.4f}')

    # Save model
    model_path = 't_bert_model.pt'
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()
