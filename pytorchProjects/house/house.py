#4/5/2026
#First regression model
#using fetch_california_housing api to get data
#objective is to make a regression model to predict price of house

#importing libs. 
#pd for reading csv, torch for ML, fetch_california_housing for data
from sklearn.datasets import fetch_california_housing
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math


#makes the housing data as a dataframe
housing = fetch_california_housing(as_frame = True)
df = housing.frame
#makes input list and output list
input_labels = []
output_labels = []



#method to format and standardize data 
def format_and_standardize_data(data):
    #formula for standardization = (x - mean) / std
    std_values = data.std()
    print(std_values)
    mean_values = data.mean()
    print(mean_values)
    print(f"Max value of house: {data['MedHouseVal'].max()}\nMin Value of house: {data['MedHouseVal'].min()}\nMean Value of house: {data['MedHouseVal'].mean()}")
    data['MedInc'] = (data['MedInc'] - mean_values[0]) / std_values[0]
    data['HouseAge'] = (data['HouseAge'] - mean_values[1])/std_values[1]
    data['AveRooms'] = (data['AveRooms'] - mean_values[2])/std_values[2]
    data['AveBedrms'] = (data['AveBedrms'] - mean_values[3])/std_values[3]
    data['Population'] = (data['Population'] - mean_values[4])/std_values[4]
    data['AveOccup'] = (data['AveOccup'] - mean_values[5])/std_values[5]
    data['Latitude'] = (data['Latitude'] - mean_values[6])/std_values[6]
    data['Longitude'] = (data['Longitude'] - mean_values[7])/std_values[7]

format_and_standardize_data(df)

#shuffles the data
df = df.sample(frac = 1).reset_index(drop = True)

#sort input labels and output label
for index,row in df.iterrows():
    #print(f"Row: {index}")
    current_house = tuple(row.to_dict().values())
    input_labels.append(current_house[:-1])
    output_labels.append(current_house[-1])

#makes two sets of input/ output tuple    
train_to = int(0.7 * len(input_labels)) 
x_train = torch.tensor(input_labels[:train_to], dtype = torch.float32)
x_test = torch.tensor(input_labels[train_to:], dtype = torch.float32)

y_train = torch.tensor(output_labels[:train_to], dtype = torch.float32).unsqueeze(1)
y_test = torch.tensor(output_labels[train_to:], dtype = torch.float32).unsqueeze(1)

#Our model constructor: Note <Dropout is here because i thought the machine is memorizing, so using dropout to remove (p*100) percent of labels
class RegressionModel(nn.Module):
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

#create said model
model = RegressionModel(in_features = 8, out_features = 1)

#set up loops and learning rate
epochs = 10000
lr = 0.001
#set up optimizer and loss calculation
#Using MSE due to output being a mean value
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

#standard train loop
for epoch in range(epochs):
    y_hat = model(x_train)
    loss = loss_fn(y_hat, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Training Epoch {epoch:02d}: Loss ={loss.item():.4f}")

#Realized that testing do not need loops
model.eval()
with torch.no_grad():
    y_hat = model(x_test)
    loss = loss_fn(y_hat, y_test)
    print(f"Test Epoch Loss ={loss.item():.4f}")
    print(f"{math.sqrt(loss)}")
    
model.train()