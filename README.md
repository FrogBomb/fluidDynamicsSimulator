# fluidDynamicsSimulator
A simple, interactive fluid dynamics simulator and recorder. 
This was mostly written in 2014, but has been lightly cleaned up and published in 2017. There were improved versions of this (a little faster, better fluid editing tools), but I can't find those versions at this point. 

## Installation
To Run, you will need to install python 2.7, numpy, pygame, and OpenCV 2

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
Assuming you are already in an environment with python 2.7,
run the following in the base directory of the repository:
```cmd
(py27) ~/fluidDynamicsSimulator> python __init__.py
```

## Basic controls
* _Left Click_: Paint
* _Right Click + Drag_: Generate velocity vectors
