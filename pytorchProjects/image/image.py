import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_set = torchvision.datasets.CIFAR10(root = './data',train = True, download = False, transform = transform)
test_set = torchvision.datasets.CIFAR10(root = './data',train = False, download = False, transform = transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle = False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog' , 'horse', 'ship', 'truck')

def view_classification(image, probabilities):
    probabilities = probabilities.data.numpy().squeeze()
    
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    
    image = image.permute(1, 2, 0)
    denormalized_image = image / 2 + 0.5
    ax1.imshow(denormalized_image)
    ax1.axis('off')
    ax2.barh(np.arange(10), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()  

class ConvNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3,64,3)
        self.conv2 = nn.Conv2d(64,128,3)
        
        self.pool = nn.MaxPool2d(2, stride = 2)
        
        self.fc1 = nn.Linear(128 * 6 * 6 , 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = ConvNeuralNet().to(device)
summary(net, (3,32,32))

loss_fn = nn.NLLLoss()
optimizer = optim.Adam(net.parameters() , lr = 0.001)
epochs = 10
for epoch in range(epochs):
   running_loss =0.0
   for i, data in enumerate(train_loader):
       inputs , lables = data[0].to(device), data[1].to(device)
       optimizer.zero_grad()
       outputs = net(inputs)
       loss = loss_fn(outputs, lables)
       loss.backward()
       optimizer.step()
       running_loss += loss.item()
       if i % 2000 == 1999:
           print(f'[{epoch + 1}/{epochs} , {i * 1:5d}] loss: {running_loss / 2000:.3f}')
           running_loss = 0.0
           
           
print("finished training")
images, _ = next(iter(test_loader))

image = images[0]
batched_image = image.unsqueeze(8).to(device)
with torch.no_grad():
    log_probabilities = net(batched_image)

probabilities = torch.exp(log_probabilities).squeeze().cpu()
view_classification(image, probabilities)