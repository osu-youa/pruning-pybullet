# pruning-pybullet

PyBullet setup used to train and run the reinforcement learning algorithm for the pruning experiment described in the paper "Precision fruit tree pruning using a learned hybrid vision/interaction controller".

The pipeline for setting up the framework consists of four steps: Preparing the models, training the GAN, training the reinforcement learning framework, and running it on a real robot.

## Preparing the models

For each tree you want to use in the simulation, you need to generate an annotation file which contains information about the pruning targets and their respective collision joints (approximated as a dense collection of points, due to PyBullet's lack of ability to do "ghost" collisions). This process is taken care of in process_mesh_annotations.py. The steps are:

- From your tree model (.obj), create a model which contains the faces that represent the targets. Save it as \[basename\]-annotations.obj.
- Create a collision model (must be a watertight mesh or a collection of watertight objects). V-HACD is a good way to do this. (See process_mesh_annotations.py for some parameters used for the Blender V-HACD addon.) Save it as \[basename\]-collision.obj.
- Take your 3 .obj files and place them in models/trees. Then run process_mesh_annotations.py, which should generate a file of the format \[basename\].annotations, which is a Python Pickle object containing the target points and their collision approximations.

## Training the GAN

This system requires using the pix2pix framework, which can be found here: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

The GAN training consists of generating randomized textures and then applying them to the environment. The following files are involved in this process:

- training_utils.py - Contains utils for taking a folder full of base textures and generating more textures by randomizing the colors on them.
- generate_training_data.py - Takes the generated textures and environment and outputs image pairs which can be used for training.

## Training the framework

The training of the framework is taken care of in pybullet_env.py. First, make sure your models are being loaded properly and that the collision works (set action to 'eval', use_trained to False). Then run the training by setting action to 'train'. 

## Running the framework on a real setup

The framework can be run on a Realsense camera in stream_camera_nn.py.

- If you want to test that it's working, set action to 'eval'. When run, the neural network will take the input from the camera, run it through the GAN, and output a control action in the visualization.
- For actually connecting it to a system, set action to 'server'. In server mode, the script will set up a socket connection that outputs the control actions as encoded NumPy arrays. All that is necessary is to setup a client script that will decode the arrays and output them to the robot accordingly.
