#4/9/2026

import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

input_labels = []
output_labels = []

def format_data():
    standarization = lambda a, b, c: (a - b) / c
    columns_to_remove = ["planet_name", "host_star", "discovery_method", "disc_year", "disc_facility", "controversial_flag", "is_recent_discovery", "orbital_period_cat", "dist_category"]
    all_columns = pd.read_csv("planet.csv", nrows=0).columns.tolist()
    columns_to_use = [col for col in all_columns if col not in columns_to_remove]
    data = pd.read_csv('planet.csv', usecols = columns_to_use)
    null_columns =  data.columns[data.isnull().any()]
    #df_fixed['type1'], codes = pd.factorize(df_fixed['type1'])
    data['planet_type'], _ = pd.factorize(data['planet_type'])
    data['star_type'] , _ = pd.factorize(data['star_type'])
    data['habitable_zone_flag'] = data['habitable_zone_flag'].replace({
    "False" : 0,
    "True": 1
    })
    data['multi_planet_system'] = data['multi_planet_system'].replace({
    "False" : 0,
    "True": 1
    })
    for col in null_columns:
        if col in data.columns:
           mean = data[col].mean()
           data[col] = data[col].fillna(mean)
    
    column_to_move = data.pop('planet_type')
    data['planet_type'] = (column_to_move)
    std_values = data.iloc[:, :18].std()
    mean_values = data.iloc[:,:18].mean()
    for index, col in enumerate(data.columns[:18]):
        data[col] = standarization(data[col],mean_values[index],  std_values[index])
    
    return data
    #data.to_csv('output')
    
    
    
           
    

def sort_data(data):
    data = data.sample(frac = 1).reset_index(drop = True)
    for index, row in data.iterrows():
        print(f"Row {index}")
        current_planet = tuple(row.to_dict().values())
        input_labels.append(current_planet[:-1])
        output_labels.append(current_planet[-1])
        
data = format_data()
sort_data(data)

print("=== Hyperparameters ===")
dropout = float(input("Dropout (default 0.25): ") or 0.25)
lr = float(input("Learning rate (default 0.01): ") or 0.01)
weight_decay = float(input("Weight decay (default 1e-4): ") or 1e-4)
epochs = int(input("Epochs (default 1000): ") or 1000)

class MultiClassificationModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer_1 = nn.Linear(in_features, in_features * 2)
        self.layer_2 = nn.Linear(in_features *2, in_features )
        self.layer_3 = nn.Linear(in_features, out_features)
        self.dropout =  nn.Dropout(p = dropout)
        
    def forward(self, x):
        layer1_output = self.dropout(torch.relu(self.layer_1(x)))
        layer2_output = self.dropout(torch.relu(self.layer_2(layer1_output)))
        return self.layer_3(layer2_output)

total_len = len(input_labels)

train_end = int(0.7 * total_len)
val_end = int(0.9 * total_len)  # 70% + 20% = 90%

x_train = torch.tensor(input_labels[:train_end], dtype=torch.float32)
x_val = torch.tensor(input_labels[train_end:val_end], dtype=torch.float32)
x_test = torch.tensor(input_labels[val_end:], dtype=torch.float32)

y_train = torch.tensor(output_labels[:train_end], dtype=torch.long)
y_val = torch.tensor(output_labels[train_end:val_end], dtype=torch.long)
y_test = torch.tensor(output_labels[val_end:], dtype=torch.long)

# Tiny training set — easiest way to cause overfitting

plt.ion()
fig, ax = plt.subplots()
train_losses, val_losses = [], []

model = MultiClassificationModel(in_features=len(data.columns)-1, out_features=data['planet_type'].nunique())
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)

for epoch in range(epochs):
    model.train()
    y_hat = model(x_train)
    loss = loss_fn(y_hat, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(x_val), y_val)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if epoch % 10 == 0:
        ax.clear()
        ax.plot(train_losses, label='Train loss', color='steelblue')
        ax.plot(val_losses, label='Val loss', color='tomato')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title(f'Epoch {epoch} | Train={loss:.4f} | Val={val_loss:.4f}')
        plt.pause(0.01)

plt.ioff()
plt.show()
