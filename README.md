# pybullet-test

PyBullet setup used to train and run the reinforcement learning algorithm for the pruning experiment described in the paper "Precision fruit tree pruning using a learned hybrid vision/interaction controller".

The pipeline for setting up the framework consists of four steps: Preparing the models, training the GAN, training the reinforcement learning framework, and running it on a real robot.

## Preparing the models

(Model preparation TBD - From visual file, generate V-HACD collision mesh, generate annotations)

## Training the GAN

This system requires using the pix2pix framework, which can be found here: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

The GAN training consists of generating randomized textures and then applying them to the environment. The following files are involved in this process:

- training_utils.py - Contains utils for taking a folder full of base textures and generating more textures by randomizing the colors on them.
- generate_training_data.py - Takes the generated textures and environment and outputs image pairs which can be used for training.

## Training the framework

TBD

## Running the framework on a real setup

TBD
