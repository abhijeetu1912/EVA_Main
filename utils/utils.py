from functools import reduce
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch_lr_finder import LRFinder


# device
def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return use_cuda, device


# Data transformation/augmentation
class Transforms:
    """
    Padding of 4 -> Random crop of size 32*32
    Cut out of size 16*16
    """
    def __init__(self, train = True):
        if train:
            self.transformations = A.Compose(
                [
                    A.OneOf([
                        A.Sequential([
                            A.PadIfNeeded(min_height = 32 + 2*4, min_width = 32 + 2*4, 
                                          always_apply = True),
                            A.RandomCrop(height = 32, width = 32, always_apply = True),
                        ], p = 1),
                        A.Sequential([
                            A.CoarseDropout(max_height = 16, max_width = 16, min_height = 16, 
                                            min_width = 16, min_holes = 1, max_holes = 1, 
                                            fill_value = [125.31, 122.95, 113.87], 
                                            always_apply = True),
                        ], p = 1)
                    ], p = 0.5), 
                    A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transformations = A.Compose(
                [
                    A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)),
                    ToTensorV2(),
                ]
            )

    def __call__(self, img):
        return self.transformations(image = np.array(img))["image"]


# de-normalize image back to normal
def denormalize_image(img, means = (0.4914, 0.4822, 0.4465), 
                      stds = (0.2470, 0.2435, 0.2616)):
    img = img.astype(np.float32)
    for i in range(img.shape[0]):
        img[i] = (img[i] * stds[i]) + means[i]
    orig_img = np.transpose(img, (1, 2, 0))
    return orig_img


# save random images
def show_random_images(data_loader, classes, folder, num_images = 10):
    images, labels = next(iter(data_loader))
    fig = plt.figure(figsize = (5 * num_images // 5, 5))
    for i in range(num_images):
        sub = fig.add_subplot(num_images // 5, 5, i + 1)
        npimg = denormalize_image(images[i].cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        sub.set_title("{}".format(classes[labels[i]]))
    plt.tight_layout()
    plt.savefig(folder+'training_images.png')


# save performance plots
def show_performance_plots(train, test, epochs, folder):
    fig, axs = plt.subplots(2,2,figsize=(15, 7))

    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].plot(range(len(train.train_losses)), train.train_losses)

    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].set_title("Training Accuracy")
    axs[1, 0].plot(range(len(train.train_accs)), train.train_accs)

    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title("Test Loss")
    axs[0, 1].plot(range(epochs), test.test_losses)

    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].set_title("Test Accuracy")
    axs[1, 1].plot(range(epochs), test.test_accs)

    plt.tight_layout()
    plt.savefig(folder+'performance_plot.png')


# Retrieve model layer
def get_module_by_name(module: Union[torch.Tensor, nn.Module],
                       access_string: str):
    """
    Retrive layer by name
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


# prdict classes from model
def predict_label(model, images):
    outputs = model(images)
    pred_labels = outputs.argmax(dim = 1, keepdim = True)
    return pred_labels

# find miss classified examples
def get_missclassified_records(model, data_loader, device, num_images = 10):
    images = []
    labels = []
    pred_labels = []
    model.eval()
    with torch.no_grad():
         for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            preds = predict_label(model, data)                           # get the index of the max log-probability
            for i in range(len(target)):
                if preds[i] != target[i]:
                   images.append(data[i])
                   labels.append(target[i])
                   pred_labels.append(preds[i])
    return images[:num_images], labels[:num_images], pred_labels[:num_images]


# save miss classified images
def show_missclassified_images(images, labels, pred_labels, classes, folder, num_images = 10):
    fig = plt.figure(figsize = (5 * num_images // 5, 5))
    for i in range(num_images):
        sub = fig.add_subplot(num_images // 5, 5, i + 1)
        npimg = denormalize_image(images[i].cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        sub.set_title("True: {} \nPredicted: {}".format(classes[labels[i]], classes[pred_labels[i]]))
    plt.tight_layout()
    plt.savefig(folder+'misclassified_images.png')


# plot gradcxam heatmap with image
def plot_gradcam(model, device, layer, images, labels, classes, folder, num_images = 10, use_cuda = True, true_label = True):
    # layer to use for gradcam
    target_layers = [get_module_by_name(model, layer)]
    # grad cam object
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    # plot 10 gradcam images
    fig = plt.figure(figsize = (5 * num_images // 5, 5))
    for i in range(num_images):
        # calculate gradcam output
        input_tensor = images[i].unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(labels[i])]
        rgb_img = denormalize_image(images[i].cpu().numpy().squeeze())
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # add subplots
        sub = fig.add_subplot(num_images // 5, 5, i + 1)
        plt.imshow(visualization, cmap="gray")
        if true_label:
           sub.set_title("True: {}".format(classes[labels[i]]))
        else:
           sub.set_title("Predicted: {}".format(classes[labels[i]]))
    plt.tight_layout()
    if true_label:
        plt.savefig(folder+'gradcam_true_label.png')
    else:
        plt.savefig(folder+'gradcam_predicted_label.png')

# LR finder for one cycle policy
def lrFinder(model, train_loader, optimizer, criterion, device, start_lr=0, 
              end_lr=4, num_iter=200, trials=5, boundary=2, boundary_factor=0.5):
    try:
        for i in range(trials):
            lr_finder = LRFinder(model, optimizer, criterion, device)
            lr_finder.range_test(train_loader= train_loader, start_lr= start_lr, end_lr=end_lr, num_iter=200)
            lr_finder.reset()

            min_loss = min(lr_finder.history['loss'])
            ler_rate = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
        
            if i != (trials-1):
               start_lr = max(0.0, ler_rate - boundary)
               end_lr = ler_rate + boundary
               boundary *= boundary_factor
    except:
        lr_finder.reset()
    lr_finder.reset()
    return ler_rate 


# data augmentation for assignment 8
class Transforms_A8:
    def __init__(self, train=True):
        if train:
            self.transformations = A.Compose(
                [
                    A.Sequential([
                        A.PadIfNeeded(min_height = 32 + 2*4, min_width = 32 + 2*4, always_apply = True), 
                        A.RandomCrop(height = 32, width = 32, always_apply = True),
                        ], p = 0.5),
                    A.HorizontalFlip(p = 0.5), 
                    A.CoarseDropout(max_height = 8, max_width = 8, min_height = 8, min_width = 8, 
                                    min_holes = 1, max_holes = 1, fill_value = [125.31, 122.95, 113.87], p = 0.5),
                    A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transformations = A.Compose(
                [
                    A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)),
                    ToTensorV2(),
                ]
            )

    def __call__(self, img):
        return self.transformations(image = np.array(img))["image"]


# one ccyle lr plot
def lr_plot(lr_tracker, folder):
    idx = [x for x in range(len(lr_tracker))]
    plt.plot(idx, lr_tracker)
    plt.title("Learning Rate vs Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Learning Rate")
    plt.tight_layout()
    plt.savefig(folder+'one_cycle_lr_plot.png')