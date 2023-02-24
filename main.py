import sys
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.utils.data import DataLoader
from torchsummary import summary
from utils.utils import get_device, Transforms, Transforms_A8, show_random_images, get_missclassified_records, show_missclassified_images, show_performance_plots, plot_gradcam, lrFinder, lr_plot
from models.resnet import ResNet18
from models.custom_resnet import CustomResNet
from models.model_a9 import Model_A9


class Trainer:
    def __init__(self, args, model, train_loader, criterion, optimizer, device, scheduler):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.train_accs = []
        self.scheduler = scheduler
        self.lr_scheduler_flag = args.lr_scheduler.lower()
        if self.lr_scheduler_flag == "onecyclelr":
           self.lr_tracker = []

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
            # scheduler step
            if self.lr_scheduler_flag == "onecyclelr":
               self.scheduler.step()
               # track learning rate
               self.lr_tracker.append(self.scheduler.optimizer.param_groups[0]['lr'])


class Tester:
    def __init__(self, model, test_loader, criterion, device):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.test_losses = []
        self.test_accs = []

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
                  self.test_losses[-1], self.test_accs[-1]))


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
        loss = batch_loss / len(data_loader.dataset)
        # accuracy
        accuracy = 100 * correct / len(data_loader.dataset)
        # print epoch statistics
        print('\n{} set: Average loss: {:.5f}, Accuracy: {:.2f}\n'.format(
              split, loss, accuracy))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', default = 20, type = int, help = 'Number of epochs for training')
    parser.add_argument('--batch_size', default = 16, type = int, help = 'Batch size')
    parser.add_argument('--optimizer', default = 'SGD', type = str, help = 'Batch size')
    parser.add_argument('--lr', default = 1e-3, type = float, help = 'Learning rate')
    parser.add_argument('--momentum', default = 0.9, type = float, help = 'Momentum')
    parser.add_argument('--lr_scheduler', default = 'StepLR', type = str, help = 'Which LR scheduler needs to be used')
    parser.add_argument('--step_size', default = 5, type = int, help = 'Step size for StepLR')
    parser.add_argument('--end_lr', default = 4, type = int, help = 'End LR for LR Finder')
    parser.add_argument('--num_iter', default = 200, type = int, help = 'Number of iterations for LR Finder')
    parser.add_argument('--trials', default = 5, type = int, help = 'Number of trials for LR Finder')
    parser.add_argument('--boundary', default = 4, type = int, help = 'Search space to limit LR range for LR Finder')
    parser.add_argument('--boundary_factor', default = 0.5, type = float, help = 'Factor by which search space to limit LR range for LR Finder is reduced')
    parser.add_argument('--div_factor', default = 100, type = int, help = 'div factor for starting lr')
    parser.add_argument('--final_div_factor', default = 10000, type = int, help = 'div factor for min lr')
    parser.add_argument('--pct_start', default = 5/24, type = float, help = 'epoch in which max lr should occur')
    parser.add_argument('--three_phase', default = False, type = bool, help = 'If three phase lr annhilation should be implemented')
    parser.add_argument('--augmentation', default = False, type = bool, help = 'If data augmentation will be applied')
    parser.add_argument('--weight_decay', default = 0.1, type = float, help = 'L2 weight decay for regularization')
    parser.add_argument('--save_plots', default = '/content/drive/MyDrive/EVA8/Plots/', type = str, help = 'folder to save plots')
    parser.add_argument('--num_images', default = 10, type = int, help = 'number of images to plot')
    parser.add_argument('--assignment_num', default = 7, type = int, help = 'Assignment number')
    parser.add_argument('--gradcam_layer', default = 'layer3.1.conv2', type = str, help = 'Layer to bew used for GradCam should be applied or not')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    print(args.gradcam_layer, type(args.gradcam_layer))
    # choose transformation function
    if args.assignment_num == 7:
        transformations = Transforms
    if args.assignment_num == 8:
        transformations = Transforms_A8
    else:
        transformations = Transforms_A8

    # data loader
    if args.lr_scheduler.lower() == "onecyclelr":
       train_data_lr = torchvision.datasets.CIFAR10(root = './data', train = True, 
                                   download = True, transform = transformations(train = False))
       train_loader_lr = DataLoader(train_data_lr, batch_size = args.batch_size, 
                                    shuffle = True, num_workers = 2)

    train_data = torchvision.datasets.CIFAR10(root = './data', train = True, 
                             download = True, transform = transformations(train = args.augmentation))
    train_loader = DataLoader(train_data, batch_size = args.batch_size, 
                              shuffle = True, num_workers = 2)

    test_data = torchvision.datasets.CIFAR10(root = './data', train = False, 
                                        download = True, transform = transformations(train = False))
    test_loader = DataLoader(test_data, batch_size = args.batch_size, 
                             shuffle = False, num_workers = 2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # plot example images
    show_random_images(train_loader, classes, args.save_plots, num_images = args.num_images)

    # device
    use_cuda, device = get_device()
    print("Device being used: ", device)

    # initiate model
    if args.assignment_num == 7:
        model = ResNet18().to(device)
    elif args.assignment_num == 8:
        model = CustomResNet().to(device)
    elif args.assignment_num == 9:
        model = Model_A9().to(device)
    else:
        model = ResNet18().to(device)

    # print model summary
    print(summary(model, input_size=(3, 32, 32)))

    # losss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    if args.optimizer.lower() == 'sgd':
       optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
       print("Using SGD Optimiser")
    elif args.optimizer.lower() == 'adam':
       optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
       print("Using Adam Optimiser")
    elif args.optimizer.lower() == 'rmsprop': 
       optimizer = optim.RMSprop(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
       print("Using RMSProp Optimiser")
    else:
       optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
       print("Using SGD Optimiser by default !!!")
       


    # lr scheduler
    if args.lr_scheduler.lower() in ("true", "steplr"):
        scheduler = StepLR(optimizer, step_size = args.step_size, gamma = 0.1)
    elif args.lr_scheduler.lower() == "onecyclelr":
        # find max lr
        max_lr = lrFinder(model, train_loader_lr, optimizer, criterion, device, start_lr=0, 
                          end_lr=args.end_lr, num_iter=args.num_iter, trials=args.trials,
                          boundary=args.boundary, boundary_factor=args.boundary_factor)
        print("\nMax LR given by LR Finder: {}\n".format(max_lr))                  
        scheduler = OneCycleLR(optimizer, max_lr = max_lr, steps_per_epoch = len(train_loader), 
                               pct_start = args.pct_start, epochs = args.epochs, 
                               div_factor = args.div_factor, final_div_factor = args.final_div_factor, 
                               three_phase = False, anneal_strategy = 'linear')
    else:
        scheduler = None

    # intitiate trainer and tester instances
    train = Trainer(args, model, train_loader, criterion, optimizer, device, scheduler)
    test = Tester(model, test_loader, criterion, device)

    # model training
    for epoch in range(args.epochs):
        print("EPOCH:", epoch)
        train.train()
        test.test()
        if args.lr_scheduler.lower() == "steplr":
            scheduler.step()

    # model evaluation
    evaluate(model, train_loader, criterion, device, split = 'Train')
    evaluate(model, test_loader, criterion, device, split = 'Test')

    # show lr plot for one cycle policy
    if args.lr_scheduler.lower() == "onecyclelr":
       lr_plot(train.lr_tracker, args.save_plots)

    # show model performance plots
    show_performance_plots(train, test, args.epochs, args.save_plots)

    # get miss classified images, their truue labels and predicted labels
    miss_images, miss_labels, miss_pred_labels = get_missclassified_records(model, test_loader, 
                                                                            device, args.num_images)

    # show miss classified images
    show_missclassified_images(miss_images, miss_labels, miss_pred_labels, classes, 
                               args.save_plots, args.num_images)

    # grad cam
    if args.gradcam_layer not in ["None", "none"]:
        # show grad cam output of miss classified images against true label
        plot_gradcam(model, device, args.gradcam_layer, miss_images, miss_labels, classes, 
                     args.save_plots, num_images = 10, use_cuda = True, true_label = True)

        # show grad cam output of miss classified images against predicted label
        plot_gradcam(model, device, args.gradcam_layer, miss_images, miss_pred_labels, classes, 
                     args.save_plots, num_images = 10, use_cuda = True, true_label = False)




