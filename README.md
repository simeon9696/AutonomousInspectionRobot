# AutonomousInspectionRobot

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

This repository deals with the Deep Learning aspect of AIR. Specifically the Generative Adversarial Networks (GANs) that will be used to generate images of obstacles that the system would encounter during flight. These images would then be used to train an object classifier.

## Getting Started

This project is built with Python v3.8.3 and uses the following dependancies. Please install the packages below before attempting to run any of the code in this repository.

- PyTorch
    `pip install torch torchvision #linux`  
- Numpy  
    `pip install numpy`  
- Matplotlib  
    `pip install matplotlib`  

If you're planning to use PyTorch on windows please visit the [configuration tool](https://pytorch.org/get-started/locally/) on PyTorch to generate the appropriate install command

## Usage

- Place images into `data/*YOUR_FOLDER_NAME*/images`
- Change line 123 to `data/*YOUR_FOLDER_NAME*/images` i.e `dataroot = data/*YOUR_FOLDER_NAME*/images`
- Run `gan.py`

## Next Steps

- Upgrade the gan to generate full 1024 x 1024 images
- Speed up the code by leveraging TPU synatx 