# -*- coding: utf-8 -*-
"""
FLUID DYNAMICS SIMULATOR

Requires Python 2.7 and OpenCV 2

Tom Blanchet (c) 2013 - 2014 (revised 2017)
"""
import cv2
from fluidDoubleCone import FluidDoubleCone, RGB
from fluidDoubleConeView import FluidDCViewCtrl

def main():
    inImg = cv2.imread("gameFace.jpg")
    game_face_dCone = FluidDoubleCone(inImg, RGB)
    ui = FluidDCViewCtrl(game_face_dCone, 615, 737)
    ui.mainloop()

if __name__ == "__main__":
    main()
