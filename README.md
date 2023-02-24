# EVA_Main

## Code Structure
1. models folder - contains resnet 18 & 34 model, custom resnet model and ultimus model
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
--lr_scheduler SGD \ \
--step_size 5 \ \
--augmentation True \ \
--weight_decay 0.1 \ \
--save_plots /content/drive/MyDrive/EVA8/S7/Plots/ \ \
--num_images 10 \
--gradcam_layer none

### Arguments for main.py

--epochs', default = 20, type = int, help = 'Number of epochs for training' \
--batch_size', default = 16, type = int, help = 'Batch size' \
--optimizer', default = 'SGD', type = str, help = 'Batch size' \
--lr', default = 1e-3, type = float, help = 'Learning rate' \
--momentum', default = 0.9, type = float, help = 'Momentum' \
--lr_scheduler', default = 'StepLR', type = str, help = 'Which LR scheduler needs to be used' \
--step_size', default = 5, type = int, help = 'Step size for StepLR' \
--end_lr', default = 4.0, type = float, help = 'End LR for LR Finder' \
--num_iter', default = 200, type = int, help = 'Number of iterations for LR Finder' \
--trials', default = 5, type = int, help = 'Number of trials for LR Finder' \
--boundary', default = 4.0, type = float, help = 'Search space to limit LR range for LR Finder' \
--boundary_factor', default = 0.5, type = float, help = 'Factor by which search space to limit LR range for LR Finder is reduced' \
--div_factor', default = 100, type = int, help = 'div factor for starting lr' \
--pct_start', default = 5/24, type = float, help = 'epoch in which max lr should occur' \
--three_phase', default = False, type = bool, help = 'If three phase lr annhilation should be implemented' \
--augmentation', default = False, type = bool, help = 'If data augmentation will be applied' \
--weight_decay', default = 0.1, type = float, help = 'L2 weight decay for regularization' \
--save_plots', default = '/content/drive/MyDrive/EVA8/Plots/', type = str, help = 'folder to save plots' \
--num_images', default = 10, type = int, help = 'number of images to plot' \
--assignment_num', default = 7, type = int, help = 'Assignment number' \
--gradcam_layer', default = 'layer3.1.conv2', type = str, help = 'Layer to bew used for GradCam should be applied or not' \


