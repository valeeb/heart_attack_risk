import numpy as np
import torch

def train(dataloader, model, loss_fn, optimizer,device='cpu'):
    """
    trains a model to the data given by dataloader
    dataloader: training set, must be a pytorch DataLoader object
    model: CNN model, a NeuralNetwork object
    loss_fn: loss function as defined by pytorch.nn
    optimizer: optimizer defined by pytorch.optim
    """

    loss_values=[]
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_values.append(loss.item())
    
    return np.mean(loss_values)


def test(dataloader, model, loss_fn, device='cpu'):
    """
    tests a model to the data given by dataloader
    dataloader: testing set, must be a pytorch DataLoader object
    model: CNN model, a NeuralNetwork object
    loss_fn: loss function as defined by pytorch.nn
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct,test_loss