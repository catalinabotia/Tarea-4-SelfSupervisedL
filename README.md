# Tarea 4 Self-supervised learning
Self-supervised learning module for the Advanced Machine Learning course at Universidad de los Andes.

## Objectives:
1. Learn to use one of the State-of-the-art methods in the self-supervised and representation learning world.
2. Learn how to implement a downstream task in order to evaluate the quality of the features learned by the network
3. Understand the implementation of contrastive learning.
4. Have fun!

## Getting Started
Clone this repository, create a new environment and run the following:
```
$ conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```
To install the remaining requirements necessary to run the code:
```
$ pip install -r requirements.txt
```
## Dataset
The dataset used for this homework will be the CIFAR-10 dataset which will be downloaded once you run the code.

## Running the code
### To train
```
$ python train.py --config CONFIGFILE_NAME --gpu-id GPUID
```
### To evaluate
```
$ python eval.py --config CONFIGFILE --gpu-id GPUID
```
## Homework 
1. Implement the linear classifier from scratch for the downstream task. Train the model and evaluate it. (2 points)
2. Add more layers to the linear classifier and experiment with some of the parameters. Do the results improve? Discuss (1 point)
3. Change the backbone used in the encoder with another backbone and run the model again. Compare the results obtained using this new backbone. (1 point)
4. About the evaluation method, do you think that the way the representations are being evaluated (how well they perform in the downstream tasks) is a good way of evaluating them? Discuss advantages and disadvantages of evaluating the representations this way. (1 point)

## Bonus
Train SimCLR on either MNIST or KMNIST and evaluate on the dataset that was not used for training (For example if I train on MNIST I will evaluate on KMNIST and viceversa). Discuss your results.

## Deadline
September 28 2020 - 11:59 pm

## Credits
This repository was adapted from Guillaume Jeanneret's pytorch implementation of SimCLR.
