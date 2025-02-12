import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vision_expert import VisionTransformer
from vision_expert import Config

config = Config()

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

model = VisionTransformer(config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
device_ids = list(range(num_gpus))

model = nn.DataParallel(model, device_ids=device_ids)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
accumulation_steps = 4

for epoch in range(40):
    running_loss = 0.0
    optimizer.zero_grad()  
    
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
    
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps
        loss.backward()
        
        running_loss += loss.item() 
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {running_loss :.4f}")