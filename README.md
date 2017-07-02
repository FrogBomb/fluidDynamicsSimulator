# fluidDynamicsSimulator
A simple, interactive fluid dynamics simulator and recorder.

## Installation
You will need to install python 2.7, numpy, pygame, and OpenCV 2

This step is simple if you are using anaconda. 
Make sure you have properly installed anaconda, then run the following lines:
```cmd
~> conda create -n py27 python=2.7 anaconda
~> activate py27
(py27) ~> conda install -c menpo opencv=2.4.11
(py27) ~> conda install -c cogsci pygame=1.9.2a0

```
(numpy should already be installed)

If you are already using anaconda, be sure you are in an environment with python 2.7,
and just run the following
```cmd
(py27) ~> conda install -c menpo opencv=2.4.11
(py27) ~> conda install -c cogsci pygame=1.9.2a0
```

## To Run Demo
Assuming your python 2.7 environment is named `py27`,
run the following in the base directory of the repository:
```cmd
~/fluidDynamicsSimulator> activate py27
(py27) ~/fluidDynamicsSimulator> python __init__.py
```

## Basic controls
* _Left Click_: Paint
* _Right Click + Drag_: Generate velocity vectors
