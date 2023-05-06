import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from scipy.io import loadmat

# 自定义数据集类
class MatDataset(Dataset):
    def __init__(self, normal_dir, patient_dir):
        self.normal_dir = normal_dir
        self.patient_dir = patient_dir
        self.normal_files = os.listdir(normal_dir)
        self.patient_files = os.listdir(patient_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.normal_files) + len(self.patient_files)

    def __getitem__(self, idx):
        if idx < len(self.normal_files):
            file_path = os.path.join(self.normal_dir, self.normal_files[idx])
            label = 1
        else:
            file_path = os.path.join(self.patient_dir, self.patient_files[idx - len(self.normal_files)])
            label = 0

        mat_data = loadmat(file_path)['data']
        mat_data = self.transform(mat_data)
        return mat_data, label

# 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 58 * 58, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 58 * 58)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            true_labels.extend(target.cpu().numpy())
            predicted_labels.extend(pred.squeeze().cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_labels)

    return accuracy, recall, f1, auc
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normal_dir = "normal"
    patient_dir = "patient"

    dataset = MatDataset(normal_dir, patient_dir)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy, recall, f1, auc = test(model, device, test_loader)
        print(
            "Epoch: {}, Accuracy: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}, AUC: {:.2f}".format(epoch, accuracy, recall,
                                                                                                f1, auc))
if __name__ == '__main__':
    main()