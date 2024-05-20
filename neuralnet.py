import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch

training_transform = transforms.Compose([
	transforms.RandomRotation(30),
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.Flowers102(root='data', split='train', download=True, transform=training_transform)
val_dataset = datasets.Flowers102(root='data', split='val', download=True, transform=transform)
test_dataset = datasets.Flowers102(root='data', split='test', download=True, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Neural Net Model
class FlowerNet(nn.Module):
    def __init__(self):
        super(FlowerNet,self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 102)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        
        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = self.bn3(self.pool(F.relu(self.conv3(x))))
        x = self.bn4(self.pool(F.relu(self.conv4(x))))
        x = self.bn5(self.pool(F.relu(self.conv5(x))))

        x = x.view(-1, 512 * 7 * 7)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlowerNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001,weight_decay=1e-5)
num_epochs = 20 
patience = 10

# Training and Evaluation
def train_eval(model, traindataloader, validateloader, TrCriterion, optimizer, epochs, deviceFlag_train):
    model.to(deviceFlag_train)
    
    for e in range(epochs):
        since = time.time()
        model.train()
        best_val_loss = float('inf')
        training_loss_running = 0
        itrs = 0
        
        for inputs, labels in traindataloader:
            itrs += 1
            inputs = inputs.to(deviceFlag_train)
            labels = labels.to(deviceFlag_train)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            train_loss = TrCriterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            training_loss_running += train_loss.item()
            
            if itrs % 4590 == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss, val_acc = validation(model, validateloader, TrCriterion)
                print(f'Epoch: {e + 1}/{epochs}, Train Loss: {training_loss_running / 4590}, Validation Loss: {validation_loss}, Validation Acc: {val_acc}')
                training_loss_running = 0
                model.train()
                
                if validation_loss < best_val_loss:
                        best_val_loss = validation_loss
                        epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f'Early stopping triggered at epoch {e+1}')
                        return
                
        print(f'Epoch {e + 1} completed in {round(time.time() - since, 4)} sec')

# Function for validation phase
def validation(model, validateloader, ValCriterion):
    val_loss_running = 0
    acc = 0
    for images, labels in validateloader:
        images = images.to(device)
        labels = labels.to(device)
        output = model.forward(images)
        val_loss_running += ValCriterion(output, labels).item()
        output = torch.exp(output)
        equals = (labels.data == output.max(dim=1)[1])
        acc += equals.float().mean().item()
    return val_loss_running / len(validateloader), acc / len(validateloader)*100

########                            ##########
########           TESTING          ##########
########                            ##########
########                            ##########

# Start Training and Testing
print('Starting Training and validation')
train_eval(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
print('Starting Testing')

model.eval()
with torch.no_grad():
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on test images: {100 * correct / total}%')

