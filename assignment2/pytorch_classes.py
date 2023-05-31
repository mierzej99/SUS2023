import torch
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm


# dataset class
class ImageDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        photo = Image.open(self.pairs[idx][0])
        encoding = Image.open(self.pairs[idx][1])
        label = self.labels[idx]
        if self.transform is not None:
            photo = self.transform(photo)
            encoding = self.transform(encoding)
        return photo, encoding, label


class TwoImagesTwoConvNets(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_c1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=32, stride=32)
        self.enc_c1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=32, stride=32)
        self.act_1 = nn.Tanh()

        self.img_c2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.enc_c2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.act_2 = nn.Tanh()

        self.img_c3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.enc_c3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.act_3 = nn.Tanh()
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(6 * 6 * 10 * 2, 150)
        self.act_4 = nn.ReLU()
        self.fc2 = nn.Linear(150, 100)
        self.act_5 = nn.ReLU()
        self.fc3 = nn.Linear(100, 1)

        self.reset_parameters()
        # 512-> 16 -> 14 -> 12 -> 6

    def forward(self, img, enc):
        img = self.img_c1(img)
        enc = self.enc_c1(enc)
        img = self.act_1(img)
        enc = self.act_1(enc)

        img = self.img_c2(img)
        enc = self.enc_c2(enc)
        img = self.act_2(img)
        enc = self.act_2(enc)

        img = self.img_c3(img)
        enc = self.enc_c3(enc)
        img = self.act_3(img)
        enc = self.act_3(enc)

        img = self.p1(img)
        enc = self.p1(enc)

        img = img.view(-1, 6 * 6 * 10)
        enc = enc.view(-1, 6 * 6 * 10)
        x = torch.cat((img, enc), 1)
        x = self.fc1(x)
        x = self.act_4(x)
        x = self.fc2(x)
        x = self.act_5(x)
        x = self.fc3(x)
        x = nn.Sigmoid()(x)
        return x

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


########## Functions

def training(num_epochs, learning_rate, train_dataset, test_dataset, batch_size=8):
    model = TwoImagesTwoConvNets()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_DL = torch.utils.data.DataLoader(train_dataset, batch_size=8)

    ### graphs
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_test = [0] * num_epochs
    accuracy_hist_test = [0] * num_epochs

    for epoch in tqdm(range(num_epochs)):
        if epoch == 30:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate / 10)
        for imgs, encs, labels in train_DL:
            pred = model(imgs, encs)[:, 0]
            loss = criterion(pred, labels.to(torch.float))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist_train[epoch] += loss.item()
            correct = ((pred >= 0.5).int() == labels).float()
            accuracy_hist_train[epoch] += correct.mean()

        loss_hist_train[epoch] /= len(train_dataset) / batch_size
        accuracy_hist_train[epoch] /= len(train_dataset) / batch_size
        loss_hist_test[epoch], accuracy_hist_test[epoch] = loss_and_acc(model, test_dataset, criterion)

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 2, 1)
    plt.plot(loss_hist_train, lw=4)
    plt.plot(loss_hist_test, lw=4)
    plt.legend(['Train loss', 'Test loss'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    ax.set_ylim(bottom=0, top=1)
    ax = fig.add_subplot(1, 2, 2)
    plt.plot(accuracy_hist_train, lw=4)
    plt.plot(accuracy_hist_test, lw=4)
    plt.legend(['Train acc.', 'Test acc.'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    ax.set_ylim(bottom=0, top=1)
    return model


def acc(model, dataset):
    DL = torch.utils.data.DataLoader(dataset, batch_size=8)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, enc, labels in DL:
            pred = model(imgs, enc)[:, 0]
            total += labels.shape[0]
            correct += ((pred >= 0.5).int() == labels).sum().item()
    return correct / total


def loss_fn(model, dataset, criterion):
    loss = 0
    DL = torch.utils.data.DataLoader(dataset, batch_size=8)
    with torch.no_grad():
        for imgs, enc, labels in DL:
            pred = model(imgs, enc)[:, 0]
            loss += criterion(pred, labels.to(torch.float))
    return loss / len(dataset)

def loss_and_acc(model, dataset, criterion):
    loss = 0
    total = 0
    correct = 0
    DL = torch.utils.data.DataLoader(dataset, batch_size=8)
    with torch.no_grad():
        for imgs, encs, labels in DL:
            pred = model(imgs, encs)[:, 0]
            loss += criterion(pred, labels.to(torch.float))
            total += labels.shape[0]
            correct += ((pred >= 0.5).int() == labels).sum().item()
    return (loss/(len(dataset)/8)).item(), correct / total
