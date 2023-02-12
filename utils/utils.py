from functools import reduce
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2


# device
def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


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
                                            fill_value = [15.7198, 15.4241, 14.2844], 
                                            always_apply = True),
                        ], p = 1)
                    ], p = 0.5), 
                    A.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transformations = A.Compose(
                [
                    A.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ]
            )

    def __call__(self, img):
        return self.transformations(image = np.array(img))["image"]


# de-normalize image back to normal
def denormalize_image(img, means = (0.5, 0.5, 0.5), 
                      stds = (0.5, 0.5, 0.5)):
    img = img.astype(np.float32)
    for i in range(img.shape[0]):
        img[i] = (img[i] * stds[i]) + means[i]
    orig_img = np.transpose(img, (1, 2, 0))
    return orig_img


# save random images
def show_random_images(data_loader, classes, num_images = 10):
    images, labels = next(iter(data_loader))
    fig = plt.figure(figsize = (5 * num_images // 5, 5))
    for i in range(num_images):
        sub = fig.add_subplot(num_images // 5, 5, i + 1)
        npimg = denormalize_image(images[i].cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        sub.set_title("{}".format(classes[labels[i]]))
    plt.tight_layout()
    plt.savefig('plots/training_images.png')


# save performance plots
def show_performance_plots(train, test, epochs):
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
    plt.savefig('plots/performance_plot.png')


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
def get_missclassified_records(model, data_loader, device):
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
    return images, labels, pred_labels


# save miss classified images
def show_missclassified_images(images, labels, pred_labels, classes, num_images = 10):
    fig = plt.figure(figsize = (5 * num_images // 5, 5))
    for i in range(num_images):
        sub = fig.add_subplot(num_images // 5, 5, i + 1)
        npimg = denormalize_image(images[i].cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        sub.set_title("True: {} \nPredicted: {}".format(classes[labels[i]], classes[pred_labels[i]]))
    plt.tight_layout()
    plt.savefig('plots/misclassified_images.png')

