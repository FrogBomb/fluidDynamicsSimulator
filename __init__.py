# -*- coding: utf-8 -*-
"""
FLUID DYNAMICS SIMULATOR

Requires Python 2.7, pygame, numpy, and OpenCV 2

Tom Blanchet (c) 2013 - 2014 (revised 2017)
"""
import cv2
from fluidDoubleCone import FluidDoubleCone, RGB
from fluidDoubleConeView import FluidDCViewCtrl

def slow_example():
    inImg = cv2.imread("gameFace.jpg")
    game_face_dCone = FluidDoubleCone(inImg, RGB)
    ui = FluidDCViewCtrl(game_face_dCone, 615, 737)

def fast_example():
    inImg = cv2.imread("fastKid.png")
    fast_kid_dCone = FluidDoubleCone(inImg, RGB)
    ui = FluidDCViewCtrl(fast_kid_dCone, 200*2, 200*2)

    ui.mainloop()

if __name__ == "__main__":
    fast_example()
