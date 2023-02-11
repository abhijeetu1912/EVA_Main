import sys
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from EVA_Main.utils.utils import get_device, Transforms, denormalize_image, show_random_images, get_missclassified_records, show_missclassified_images, show_performance_plots
from models.resnet import ResNet18


class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.train_accs = []

    def train(self):
        # use model in training mode
        self.model.train()
        # track number of correct classifications & data points processed
        correct = 0
        processed = 0    
        #progress bar
        pbar = tqdm(self.train_loader)
        # iterate over batches
        for batch_id, (inputs, targets) in enumerate(pbar):
            # set device
            images, targets = inputs.to(self.device), targets.to(self.device)
            # zero out or flush gradients from last batch
            self.optimizer.zero_grad()
            # predict output
            outputs = self.model(images)
            # calculate loss
            loss = self.criterion(outputs, targets)
            # append batch train loss
            self.train_losses.append(loss.item())
            # backpropogate loss
            loss.backward()
            # update weights 
            self.optimizer.step()
            # predict class
            pred = outputs.argmax(dim=1, keepdim=True)
            # count correctly classified data points
            correct += pred.eq(targets.view_as(pred)).sum().item()
            # count number of data points processed till this iteration
            processed += len(inputs)
            # append train accuracy
            self.train_accs.append(100 * correct / processed)
            # show progress bar
            pbar.set_description(
                desc=f"Loss = {loss.item():3.2f} | Batch = {batch_id} | Accuracy = {self.train_accs[-1]:0.2f}"
            )


class Tester:
    def __init__(self, model, test_loader, criterion, device, split = 'Test'):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.test_losses = []
        self.test_accs = []
        self.split = split

    def test(self):
        #use model in evaluation mode
        self.model.eval()
        # number of correctly classified data points
        correct = 0
        # batch test loss
        batch_loss = 0
        # execute without gradients
        with torch.no_grad():
            #iterate over batches
            for data, target in self.test_loader:
                # set device
                data, target = data.to(self.device), target.to(self.device)
                # predict output
                outputs = self.model(data)
                # calculate and add up loss over batches
                batch_loss += self.criterion(outputs, target).item() 
                # predict class
                pred = outputs.argmax(dim=1, keepdim=True)
                # update number of correctly classified ata points
                correct += pred.eq(target.view_as(pred)).sum().item()
            # append average test loss of epoch
            self.test_losses.append(batch_loss / len(self.test_loader.dataset))
            # append test accuracy
            self.test_accs.append(100 * correct / len(self.test_loader.dataset))
            # print epoch statistics
            print('\nTest set: Average loss: {:.5f}, Accuracy: {:.2f}\n'.format(
                  self.test_losses[-1], correct, self.test_accs[-1]))


def evaluate(model, data_loader, criterion, device, split = 'Test'):
    #use model in evaluation mode
    model.eval()
    # number of correctly classified data points
    correct = 0
    # batch test loss
    batch_loss = 0
    # execute without gradients
    with torch.no_grad():
        #iterate over batches
        for data, target in data_loader:
            # set device
            data, target = data.to(device), target.to(device)
            # predict output
            outputs = model(data)
            # calculate and add up loss over batches
            batch_loss += criterion(outputs, target).item() 
            # predict class
            pred = outputs.argmax(dim=1, keepdim=True)
            # update number of correctly classified ata points
            correct += pred.eq(target.view_as(pred)).sum().item()
        # average loss
        loss = batch_loss / len(data_loader.dataset))
        # accuracy
        accuracy = 100 * correct / len(data_loader.dataset)
        # print epoch statistics
        print('\n{} set: Average loss: {:.5f}, Accuracy: {:.2f}\n'.format(
              split, loss, correct, accuracy))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', default = 20, type = int, description = 'Number of epochs for training')
    parser.add_argument('--batch_size', default = 16, type = int, description = 'Batch size')
    parser.add_argument('--optimizer', default = 'SGD', type = int, description = 'Batch size')
    parser.add_argument('--lr', default = 1e-3, type = float, description = 'Learning rate')
    parser.add_argument('--momentum', default = 0.9, type = float, description = 'Momentum')
    parser.add_argument('--lr_scheduler', default = False, type = bool, description = 'If lr scheduler needs to be used')
    parser.add_argument('--step_size', default = 5, type = int, description = 'Step size for StepLR')
    parser.add_argument('--augmentation', default = False, type = bool, description = 'If data augmentation will be applied')
    parser.add_argument('--weight_decay', default = 0.1, type = float, description = 'L2 weight decay for regularization')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    # data loader
    train_data = torchvision.datasets.CIFAR10(root = './data', train = True, 
                                         download = True, transform = Transforms)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, 
                              shuffle = True, num_workers = 2)

    test_data = torchvision.datasets.CIFAR10(root = './data', train = False, 
                                        download = True, transform = Transforms)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, 
                             shuffle = False, num_workers = 2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # plot example images
    show_random_images(train_loader, classes)

    # device
    device = get_device()
    print("Device being used: ", device)

    # initiate model
    model = ResNet18().to(device)

    # print model summary
    print(summary(model, input_size=(3, 32, 32)))

    # losss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    
    # lr scheduler
    if args.lr_scheduler:
        scheduler = StepLR(optimizer, step_size = args.step_size, gamma = 0.1)

    # intitiate trainer and tester instances
    train = Trainer(model, train_loader, criterion, optimizer, device)
    test = Tester(model, test_loader, criterion, split = 'Test', device)

    # model training
    for epoch in range(args.epochs):
        print("EPOCH:", epoch, "  Learning Rate: ", scheduler.optimizer.param_groups[0]['lr'])
        train.train()
        test.test()
        if args.lr_scheduler:
            scheduler.step()

    # model evaluation
    evaluate(model, train_loader, criterion, device, split = 'Test')
    evaluate(model, test_loader, criterion, device, split = 'Test')

    # show model performance plots
    show_performance_plots(train, test, args.epochs)

    # get miss classified images, their truue labels and predicted labels
    miss_images, miss_labels, miss_pred_labels = get_missclassified_records(model, test_loader, device)

    # show miss classified images
    show_missclassified_images(miss_images, miss_labels, miss_pred_labels, classes, num_images = 10)



