#4/7/2026
#Mutliclass Model
#using the https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
#objective is to make a MultiClassModel to predict what genre is that song


#importing libs. 
#pd for reading csv, torch for ML
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

#make dataframe from csv
df = pd.read_csv('music.csv')
#makes input list and output list 
input_labels = []
output_labels = []
#method to clean up data for training ML
def format_and_standardize_data(df):
    #drop the wav file name
    df_formated = df.drop(df.columns[0],axis =1)
    #replace string label with ints
    df_formated['label'] = df['label'].replace({
    'blues' : 0,
    'classical' : 1,
    'country' : 2,
    'disco' : 3,
    'hiphop': 4,
    'jazz': 5,
    'metal' : 6,
    'pop' : 7,
    'reggae' : 8,
    'rock' : 9
    })
    #standarization    
    std_values = df_formated.iloc[:, :-1].std()
    mean_values = df_formated.iloc[:, :-1].mean()
    print(std_values)
    print(mean_values)
    for index, col in enumerate(df_formated.columns[:-1]):
        print(index)
        df_formated[col] = (df_formated[col] - mean_values[index]) / std_values[index]
    
    return df_formated

#save modified df and shuffle it
df = format_and_standardize_data(df)
df = df.sample(frac = 1).reset_index(drop = True)
#df.to_csv('formateda.cvs', index = False)

#sort input labels and output label
for index, row in df.iterrows():
    print(f"Row {index}")
    current_Song = tuple(row.to_dict().values())
    input_labels.append(current_Song[:-1])
    output_labels.append(current_Song[-1])
    
#makes two sets of input/ output tuple       
train_to = int(0.7 * len(input_labels))     
    
x_train = torch.tensor(input_labels[:train_to] , dtype = torch.float32)
x_test = torch.tensor(input_labels[train_to:] , dtype = torch.float32)

y_train = torch.tensor(output_labels[:train_to] , dtype = torch.long)
y_test = torch.tensor(output_labels[train_to:] , dtype = torch.long)
print(y_train.shape)
print(y_test.shape)


#Our model constructor
class MultiClassificationModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer_1 = nn.Linear(in_features, in_features * 2)
        self.layer_2 = nn.Linear(in_features *2, in_features )
        self.layer_3 = nn.Linear(in_features, out_features)
        self.dropout =  nn.Dropout(p = 0.25)
        
    def forward(self, x):
        layer1_output = self.dropout(torch.relu(self.layer_1(x)))
        layer2_output = self.dropout(torch.relu(self.layer_2(layer1_output)))
        return self.layer_3(layer2_output)

#since we return the probaility of each class, out_features is  10        
model = MultiClassificationModel(in_features = 58, out_features = 10)

#set up loops and learning rate
epochs = 1000
lr = 0.01

#set up optimizer and loss calculatio
loss_fn = nn.CrossEntropyLoss()
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

#evaluations 
model.eval()
with torch.no_grad():
    y_hat = model(x_test)
    loss = loss_fn(y_hat, y_test)
    print(f"Test Epoch Loss ={loss.item():.4f}")
    
    predicted = torch.argmax(y_hat, dim=1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy = {accuracy.item():.4f}")
