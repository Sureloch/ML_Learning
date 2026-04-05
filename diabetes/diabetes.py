#4/2/26
#Intial start to using torch
# using the https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# objective is to make a simple classification model to predict if the person has diabetes


#importing libs. 
#pd for reading csv, torch for ML
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

#reads the data in the file.
#makes input list and output list
df = pd.read_csv('diabetes.csv')
input_labels = []
output_labels = []
for index, row in df.iterrows():
    print(f"Row {index}")
    current_Patient= tuple(row.to_dict().values())
    input_labels.append(current_Patient[:-1])
    output_labels.append(current_Patient[-1])


#makes two sets of input/ output tuple
x_train = torch.tensor(input_labels[:614], dtype = torch.float32)
x_test = torch.tensor(input_labels[614:], dtype = torch.float32)

y_train = torch.tensor(output_labels[:614], dtype = torch.float32).unsqueeze(1)
y_test = torch.tensor(output_labels[614:], dtype = torch.float32)


#NOTE: LinearRegressionModel is not a good name.  
class LinearRegressionModel(nn.Module): # nn from torch.nn
     def __init__(self, in_features, out_features):
            #nerons go from input(8) -> Hidden(16) -> Hidden(8) -> output(1)
             super().__init__()
             self.linear_layer1 = nn.Linear(in_features, in_features * 2)
             self.linear_layer2 = nn.Linear(in_features * 2, in_features)
             self.linear_layer3 = nn.Linear(in_features, out_features)

     def forward(self, x):
            #Relu used to ignore negative values
            #sigmoid used to clamp output between [0.0 and 1.0]
            layer1_output = torch.relu(self.linear_layer1(x))
            layer2_output = torch.relu(self.linear_layer2(layer1_output))
            return torch.sigmoid(self.linear_layer3(layer2_output))
  
#create said model
model = LinearRegressionModel(in_features = 8, out_features = 1)

#set up loops and learning rate
epochs = 1000
lr = 0.01
#set up optimizer and loss calculation
#Using BCE due to the output being binary (0 or 1)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr)

#standard train loop
for epoch in range(epochs):
    y_hat = model(x_train)
    loss = loss_fn(y_hat, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epochs % 10 == 0:
        print(f"Epoch {epoch:02d}: Loss ={loss.item():.4f}")
    
    
    
#Test Loop
for epoch in range(epochs):
  y_hat = model(x_test)
  predictions = y_hat >= 0.5
  adjusted_predictions = predictions.squeeze(1)
  comparision = torch.eq(adjusted_predictions, y_test)
  print(len(comparision))
  count = comparision.sum().item()
  print(f"Number of True is {count}")
    
  print(f"Accuracy: {count/len(comparision):.4f}")  
    



