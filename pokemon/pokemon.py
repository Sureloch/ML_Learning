import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv('pokemon.csv')

input_labels = []
output_labels = []
codes = []


def format_data(df):
    df_fixed = df.drop_duplicates(subset = ["number"])
    df_fixed = df_fixed.drop(columns=["type2", "name", "total", "number", "generation", "legendary"])
    mean_lambda = lambda a , b, c: (a - b) / c
    std_values = df_fixed.iloc[:, 1:].std()
    mean_values = df_fixed.iloc[:,1:].mean()
    for index, col in enumerate(df_fixed.columns[1:]):
        df_fixed[col] = mean_lambda(df_fixed[col],mean_values[index],  std_values[index])
    
    df_fixed['type1'], codes = pd.factorize(df_fixed['type1'])
    return df_fixed, codes 
    
def sort_data(df):
    df = df.sample(frac = 1).reset_index(drop = True)
    for index, row in df.iterrows():
        print(f"Row {index}")
        current_Pokemon = tuple(row.to_dict().values())
        input_labels.append(current_Pokemon[1:])
        output_labels.append(current_Pokemon[0])
 


class MultiClassificationModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer_1 = nn.Linear(in_features, in_features * 2)
        self.layer_2 = nn.Linear(in_features *2, in_features )
        self.layer_3 = nn.Linear(in_features, out_features)
        self.dropout =  nn.Dropout(p = 0.0)
        
    def forward(self, x):
        layer1_output = self.dropout(torch.relu(self.layer_1(x)))
        layer2_output = self.dropout(torch.relu(self.layer_2(layer1_output)))
        return self.layer_3(layer2_output)

df, codes = format_data(df)
sort_data(df)
train_to = int(0.7 * len(input_labels))
x_train = torch.tensor(input_labels[:train_to] , dtype = torch.float32)
x_test = torch.tensor(input_labels[train_to:] , dtype = torch.float32)

y_train = torch.tensor(output_labels[:train_to] , dtype = torch.long)
y_test = torch.tensor(output_labels[train_to:] , dtype = torch.long)
print(y_train.shape)
print(y_test.shape)
model = MultiClassificationModel(in_features = 6, out_features = len(codes))

epochs = 1000
lr = 0.01

#set up optimizer and loss calculatio
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)

for epoch in range(epochs):
    y_hat = model(x_train)
    loss = loss_fn(y_hat, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Training Epoch {epoch:02d}: Loss ={loss.item():.4f}")

model.eval()
with torch.no_grad():
    y_hat = model(x_test)
    loss = loss_fn(y_hat, y_test)
    print(f"Test Epoch Loss ={loss.item():.4f}")
    
    predicted = torch.argmax(y_hat, dim=1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy = {accuracy.item():.4f}")