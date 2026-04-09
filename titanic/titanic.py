#4/4/2026
#Diving into non-uniform data 
#using the kaggle.com/c/titanic train.csv file
#objective is to make a simple classification model to predict if the person would have survived or not


#importing libs. 
#pd for reading csv, torch for ML
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
import os

#method to clean up data for training ML
def clean_up_csv(data):
    #Block makes sex binary
    data['Sex'] = data['Sex'].replace({
    'female' : 1,
    'male' : 0
    })
    #if embarked location is unknown, make it 0.
    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].fillna(0)
    #Make a char be an integer
    data['Embarked'] = data['Embarked'].replace({
    'S' : 0,
    'C' : 1,
    'Q' : 2
    })
    #Use the mean age of the dataset to fill empty age values
    raw_mean_age = data['Age'].mean()
    mean_age = int(round(raw_mean_age,2))
    if 'Age' in data.columns:
        data['Age'] = data['Age'].fillna(mean_age)
    
    #Format fare to 2 decimal points
    data['Fare'] = data['Fare'].round(2)
    
    
    #Normalization
    data['Age'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())
    data['Fare'] = (data['Fare'] - data['Fare'].min()) / (data['Fare'].max() - data['Fare'].min())
    data['Parch'] = (data['Parch'] - data['Parch'].min()) / (data['Parch'].max() - data['Parch'].min())
    data['SibSp'] = (data['SibSp'] - data['SibSp'].min()) / (data['SibSp'].max() - data['SibSp'].min())
   
#makes input list and output list   
input_labels = []
output_labels = []

#Do the data clean-up and 
columns_to_be_removed = ["Ticket" , "Name", "PassengerId", "Cabin"]
all_columns = pd.read_csv("train.csv", nrows=0).columns.tolist()
columns_to_use = [col for col in all_columns if col not in columns_to_be_removed]
data = pd.read_csv('train.csv', usecols = columns_to_use)
clean_up_csv(data)
#Shuffles the data
data = data.sample(frac=1).reset_index(drop=True)

#sort input labels and output label
for index, row in data.iterrows():
    print(f"Row {index}")
    current_Passenger = tuple(row.to_dict().values())
    input_labels.append(current_Passenger[1:])
    output_labels.append(current_Passenger[0])
    
#makes two sets of input/ output tuple    
x_train = torch.tensor(input_labels[:624] , dtype = torch.float32)
x_test = torch.tensor(input_labels[624:] , dtype = torch.float32)

y_train = torch.tensor(output_labels[:624] , dtype = torch.float32).unsqueeze(1)
y_test = torch.tensor(output_labels[624:] , dtype = torch.float32)

#Our model constructor
class ClassificationModel(nn.Module): # nn from torch.nn
    def __init__(self, in_features, out_features):
        #nerons go from input(7) -> Hidden(14) -> Hidden(7) -> output(1)
        super().__init__()
        self.layer_1 = nn.Linear(in_features, in_features *2)
        self.layer_2 = nn.Linear(in_features * 2, in_features)
        self.layer_3 = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        #Relu used to ignore negative values
            #sigmoid used to clamp output between [0.0 and 1.0]
        layer1_output = torch.relu(self.layer_1(x))
        layer2_output = torch.relu(self.layer_2(layer1_output))
        return torch.sigmoid(self.layer_3(layer2_output))

#create said model
model = ClassificationModel(in_features = 7, out_features = 1)
#set up loops and learning rate
epochs = 1000
lr = .01
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

y_hat = model(x_test)
predictions = y_hat >= 0.5
adjusted_predictions = predictions.squeeze(1)
comparision = torch.eq(adjusted_predictions, y_test)
print(len(comparision))
count = comparision.sum().item()
print(f"Number of True is {count}")
print(model.layer_1.weight)
print(f"Accuracy: {count/len(comparision):.4f}")  
    