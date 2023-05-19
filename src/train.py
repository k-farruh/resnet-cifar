import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Define transformation for data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR100 dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Define ResNet50 model
resnet50 = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT")

num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 100)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Freeze the first few layers of the model
for name, param in resnet50.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet50.to(device)
num_epochs = 100
best_acc = 0.0

for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    resnet50.train()

    for i, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        train_acc += (predictions == targets).sum().item()

    train_loss /= len(trainloader)
    train_acc /= len(trainloader)

    # Validate the model
    val_loss = 0.0
    val_acc = 0.0
    resnet50.eval()

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = resnet50(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            val_acc += (predictions == targets).sum().item()

    val_loss /= len(testloader)
    val_acc /= len(testloader)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
        epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))

    # Save the model with best accuracy
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(resnet50.state_dict(), './model/resnet50_cifar100.pth')

    scheduler.step()

print('Best accuracy:', best_acc)
