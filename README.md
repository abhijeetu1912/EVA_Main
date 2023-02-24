# EVA_Main

## Code Structure
1. models folder - contains resnet 18 & 34 model code
2. utils - contains helper function like plot images, plot gradcam, plot performance plots, computer & show gradcam etc.
3. main.py - Code to run the model


--------------------------------------------------------------------------------------------------------------------------------

## How to Run

!python main.py \ \
--epochs 20 \ \
--batch_size 64 \ \
--optimizer SGD \ \
--lr 1e-3 \ \
--momentum 0.9 \ \
--lr_scheduler True \ \
--step_size 5 \ \
--augmentation True \ \
--weight_decay 0.1 \ \
--save_plots /content/drive/MyDrive/EVA8/S7/Plots/ \ \
--num_images 10

### Arguments for main.py

--epochs : Number of epochs for training \
--batch_size : Batch size for training \
--optimizer SGD : Optimizer used for training \
--lr : Learning rate of optimizer \
--momentum : Momentum to be used for optimizer \
--lr_scheduler : Whether learning rate scheduler needs to be used or not \
--step_size 5 : Step size for learning rate scheduler \
--augmentation : Whether data ugmentation needs to be applied or not \
--weight_decay : L2 weight decay parameter value \
--save_plots : Folder where images & folders needs to be saved \
--num_images : Numbers of images to be displayed

