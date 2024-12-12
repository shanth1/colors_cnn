import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import yaml

# ====== Определение датасета ======
class ColorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ====== Определение модели ======

class ColorCNN(nn.Module):
    def __init__(self, config, num_classes):
        super(ColorCNN, self).__init__()
        layers = []
        in_channels = 3  # Входные каналы (RGB)
        for i in range(config['model']['number_of_layers']):
            out_channels = config['model']['filters'][i]
            kernel_size = config['model']['kernel_sizes'][i]
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels * (config['model']['input_size'][0] // (2 ** config['model']['number_of_layers'])) * (config['model']['input_size'][1] // (2 ** config['model']['number_of_layers'])), num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ====== Функции для обучения и валидации ======

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_preds = torch.tensor(0, dtype=torch.float32, device=device)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct_preds += torch.sum(preds == labels.data)

        total_samples += labels.size(0)
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds.double() / total_samples
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct_preds = torch.tensor(0, dtype=torch.float32, device=device)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct_preds += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds.double() / total_samples
    return epoch_loss, epoch_acc

# ====== Обучение ======

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if config['model']['use_gpu']:
        config['use_gpu'] = torch.cuda.is_available()

    colors_dir = config['data']['colors_dir']
    colors_df = pd.read_excel(colors_dir, engine='openpyxl')
    num_classes = len(colors_df)

    transform = transforms.Compose([
        transforms.Resize(config['model']['input_size']),
        transforms.ToTensor()
    ])

    train_dataset = ColorDataset(config['data']['train_dir'], transform=transform)
    test_dataset = ColorDataset(config['data']['val_dir'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['model']['batch_size'], shuffle=False)


    model = ColorCNN(config, num_classes)

    device = torch.device('cuda' if config['model']['use_gpu'] else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])

    best_acc = 0.0
    for epoch in range(config['model']['num_epochs']):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config['model']["num_epochs"]}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config["model"]["model_path"])
