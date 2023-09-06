import torch.utils.data

from dl import MNISTModel
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train(model, dataloader, criterion, optimizer):
    epoch_loss = 0
    epoch_corrects = 0
    num_preds = 0
    model.train()
    for batch in tqdm(dataloader):
        # Forward pass
        image, label = batch
        logits = model(image)

        loss = criterion(logits, label)

        epoch_loss += loss.item()
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)

        correct_preds = (predictions == label).sum()
        num_preds += predictions.shape[0]
        epoch_corrects += correct_preds

    epoch_loss /= len(dataloader)
    accuracy = epoch_corrects / num_preds
    return epoch_loss, epoch_corrects


def validation(model, dataloader, criterion):
    epoch_loss = 0
    epoch_corrects = 0
    num_preds = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Forward pass
            image, label = batch
            logits = model(image)

            loss = criterion(logits, label)

            epoch_loss += loss.item()

            # No backward pass, because we are in validation loop

            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)

            correct_preds = (predictions == label).sum()
            num_preds += predictions.shape[0]
            epoch_corrects += correct_preds

        epoch_loss /= len(dataloader)
        accuracy = epoch_corrects / num_preds
        return epoch_loss, epoch_corrects


if __name__ == '__main__':
    model = MNISTModel()
    num_epochs = 20
    patience = 5
    checkpoint_path = 'best_checkpoint.pt'
    print(model)

    train_dataset = torchvision.datasets.MNIST('./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST('./data/', train=False, transform=transforms.ToTensor(), download=True)

    print('The length of original train set is:', len(train_dataset))
    print('The length of original test set is:', len(test_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
    print('The length of train set after the split is:', len(train_dataset))
    print('The length of val set after the split is:', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()
    print('Starting Training')
    print('=' * 20)
    best_val_loss = torch.inf
    current_patience = 0
    for epoch in range(num_epochs):
        if current_patience >= 5:
            break
        print(f'Epoch {epoch} / {num_epochs}')
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
        writer.add_scalar('Loss/Train', train_loss)
        writer.add_scalar('Accuracy/Train', train_accuracy)
        print(f'Train Loss: {train_loss}, Train accuracy: {train_accuracy}')

        val_loss, val_accuracy = validation(model, val_loader, criterion)
        current_patience += 1
        if val_loss < best_val_loss:
            current_patience = 0
            best_val_loss = val_loss
            torch.save(model, checkpoint_path)
        writer.add_scalar('Loss/Validation', val_loss)
        writer.add_scalar('Accuracy/Validation', val_accuracy)
        print(f'Validation Loss: {val_loss}, Validation accuracy: {val_accuracy}')

    print(f'Starting Testing')
    print('='*20)
    best_model = torch.load(checkpoint_path)
    test_loss, test_accuracy = validation(best_model, test_loader, criterion)
    print(f'Test Loss: {test_loss}, Test accuracy: {test_accuracy}')








