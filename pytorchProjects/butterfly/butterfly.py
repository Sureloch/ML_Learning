import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models  # CHANGED: need this for pretrained model
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# CHANGED: 224x224 instead of 32x32, and ImageNet normalization values instead of 0.5
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),   # keep
    transforms.RandomRotation(10),       # reduce a LOT
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inference_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
train_set = torchvision.datasets.ImageFolder(root='./train', transform=transform)
test_set = torchvision.datasets.ImageFolder(
    root='./test',
    transform=inference_transform
)

# CHANGED: batch size 32 instead of 4 — much better GPU utilization
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
classes = train_set.classes

# CHANGED: entire model section replaced with pretrained ResNet18
net = models.resnet18(weights='IMAGENET1K_V1')

for param in net.parameters():
    param.requires_grad = False

# Unfreeze LAST block (this is the key)
for param in net.layer4.parameters():
    param.requires_grad = True

# Replace classifier (keep this)
net.fc = nn.Linear(net.fc.in_features, len(classes))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

# CHANGED: CrossEntropyLoss instead of NLLLoss
# CrossEntropyLoss expects raw logits, so we also removed log_softmax from the model
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()),
    lr=1e-4
)

epochs = 10
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # CHANGED: print every 100 batches instead of 2000
            print(f'[{epoch + 1}/{epochs}, {i+1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0




image = Image.open("comet.jpg").convert("RGB")
image_tensor = inference_transform(image)

batched_image = image_tensor.unsqueeze(0).to(device)

net.eval()
with torch.no_grad():
    outputs = net(batched_image)

probabilities = torch.softmax(outputs, dim=1).squeeze().cpu()

top3_probs, top3_indices = torch.topk(probabilities, 3)
for prob, idx in zip(top3_probs, top3_indices):
    print(f"{classes[idx]:30s}  {prob.item()*100:.1f}%")

net.eval()
correct = 0
total = 0
class_correct = {}
class_total = {}

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        for label, pred in zip(labels, predicted):
            name = classes[label.item()]
            class_correct[name] = class_correct.get(name, 0) + (pred == label).item()
            class_total[name] = class_total.get(name, 0) + 1
            correct += (pred == label).item()
            total += 1

print(f'\nOverall accuracy: {100*correct/total:.1f}%\n')
print(f"{'Class':<35} {'Correct':>7} {'Total':>7} {'Acc':>7}")
print("-" * 58)
for name in sorted(class_total.keys()):
    acc = 100 * class_correct[name] / class_total[name]
    print(f"{name:<35} {class_correct[name]:>7} {class_total[name]:>7} {acc:>6.0f}%")