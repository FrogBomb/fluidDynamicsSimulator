# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:52:32 2013

@author: Tom
"""
import cv2
import Tkinter as tk
import tkFileDialog
import tkColorChooser
import pygame
import os
from fluidDoubleCone import *
import time
BUTTONHEIGHT = 32
FPS = 30
class FluidDCViewCtrl:
    def __init__(self, fluidDC, width, height):

        self.width = width
        self.height = height
        self.master = tk.Tk()
        self.frame = tk.Frame(self.master, width=width, height=height)
        self.frame.grid(sticky = tk.N, rowspan = 5)
        self.pickColorButton = tk.Button(self.master, text="Pick Color", \
            command = self.getNewColor)
        self.pickColorButton.grid(sticky = tk.N, row = 0, column = 1, columnspan = 3)
        self.updateDCButton = tk.Button(self.master, text="Update Fluid",\
            command = self.updateFluidDCwUndo)
        self.updateDCButton.grid(row = 1, column = 1)
        self.playDCButton = tk.Button(self.master, text="Play",\
            command = self.playFluid)
        self.playDCButton.grid(row = 1, column = 2)
        self.playSlider = tk.Scale(label = "Play seconds", orient = tk.HORIZONTAL,\
            from_ = 1.0/FPS, to = 10, resolution = 1.0/FPS)
        self.playSlider.grid(row=1, column = 3)
        self.undoButton = tk.Button(self.master, text="Undo",\
            command = self.undo)
        self.undoButton.grid(row = 2, column = 1)
        self.pickFileButton = tk.Button(self.master, text="Pick Record File",\
            command = self.setup_record)
        self.pickFileButton.grid(row = 2, column = 2)

        self.displayFile = tk.Entry(self.master, state = "readonly")
        self.displayFile.grid(row = 3, column = 2)

        self.recording = tk.IntVar()
        self.recordCheck = tk.Checkbutton(self.master, text = "Record",\
            variable = self.recording)

        self.recordCheck.grid(row = 3, column = 1)

        self.master.grid_rowconfigure(4, minsize = self.height-3*BUTTONHEIGHT)
        self.master.protocol('WM_DELETE_WINDOW', self.quit)

        self.undoCopys = []
        self.screen = None
        self.playing = False

        self.curColor = np.array([0, 0, 0])
        self.curRadius = 1
        self.fluidDC = fluidDC

        self.surface = None
        self.updateSurface()
        self.hasQuit = False

        self.videoWriter = cv2.VideoWriter()
        self.recordFile = ""

    def mainloop(self):
        os.environ['SDL_WINDOWID'] = str(self.frame.winfo_id())
        os.environ['SDL_VIDEODRIVER'] = 'windib'
        self.master.update()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        while(not self.hasQuit):
            self.update()

    def undo(self):
        if len(self.undoCopys) == 0:
            return
        self.fluidDC = self.undoCopys.pop()
        self.updateSurface()

    def setUndoCopy(self):
        self.undoCopys.append(self.fluidDC.copy())
        if len(self.undoCopys)>10:
            del self.undoCopys[0]

    def update(self, blockEvents = False):
        if not blockEvents:
            pygame.event.pump()
            self.pygameEventsHandler(pygame.event.get())
            pygame.event.clear()
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()
        self.master.update()

    def playFluid(self):
        self.setUndoCopy()
        if not self.playing:
            self.playing = True
            lastTime = time.time()
            for i in range(int(self.playSlider.get()*FPS)+1):
                self.updateFluidDC()
                self.update()
                if time.time()- lastTime<1.0/FPS:
                    time.sleep(1.0/FPS-(time.time() - lastTime))
                lastTime = time.time()
        self.playing = False


    def getRecordFile(self):
        tmp = tkFileDialog.asksaveasfilename()
        if tmp:
            if self.videoWriter.isOpened():
                cv2.destroyAllWindows()
                self.videoWriter.release()
            self.displayFile.config(state = tk.NORMAL)
            self.displayFile.delete(0, tk.END)
            self.displayFile.insert(0, tmp)
            self.displayFile.config(state = "readonly")
            self.recordFile = tmp
            self.videoWriter = cv2.VideoWriter(self.recordFile,\
                -1, FPS, (self.fluidDC.width, self.fluidDC.height), True)
            if not self.videoWriter.isOpened():
                print "Invalid File"
                self.displayFile.config(state = tk.NORMAL)
                self.displayFile.delete(0, tk.END)
                self.displayFile.config(state = "readonly")

    def updateFluidDC(self):
        self.fluidDC.update(1.0/FPS)
        self.updateSurface()
        if self.recording.get()==1:
            self.recordFrame()

    def updateFluidDCwUndo(self):
        self.setUndoCopy()
        self.updateFluidDC()

    def updateSurface(self):
        img = cv2.cvtColor(self.fluidDC.fMat.astype(np.uint8), cv2.COLOR_LAB2BGR)
        img = cv2.transpose(img)
        self.surface = pygame.transform.scale(pygame.surfarray.make_surface(img),\
                (self.width, self.height))

    def setup_record(self):
        self.getRecordFile()

#    def save(self):
#        self.videoWriter.release()
#        self.displayFile.config(state = tk.NORMAL)
#        self.displayFile.delete(0, tk.END)
#        self.displayFile.config(state = "readonly")

    def recordFrame(self):
        if self.videoWriter.isOpened():
            img = cv2.cvtColor(self.fluidDC.fMat.astype(np.uint8), cv2.COLOR_LAB2RGB)
            self.videoWriter.write(img)

#    def start_record(self):
#        self.recording = True
#
#    def end_record(self):
#        self.recording = False

    def translatePointToFluidDC(self, point):
        return point[1]*self.fluidDC.height/float(self.height),\
                                point[0]*self.fluidDC.width/float(self.width)


    def rightClick(self):
        self.setUndoCopy()
        pygame.mouse.get_rel()
        mousePoses = set()
        stayInWhile = True
        while(stayInWhile):
            if pygame.event.wait().type == pygame.MOUSEBUTTONUP:
                stayInWhile = False
            mousePoses.add(pygame.mouse.get_pos())
        mouseVel = pygame.mouse.get_rel()
        mousePoses = [self.translatePointToFluidDC(mousePos) for mousePos in mousePoses]
        mouseVel = self.translatePointToFluidDC(mouseVel)
        for mousePos in mousePoses:
            self.fluidDC.velMat[int(mousePos[0]), int(mousePos[1])] = np.array(mouseVel)


    def leftClick(self):
        self.setUndoCopy()
        stayInWhile = True
        while(stayInWhile):
            if pygame.event.wait().type == pygame.MOUSEBUTTONUP:
                stayInWhile = False
            mousePos = pygame.mouse.get_pos()
            mousePos = self.translatePointToFluidDC(mousePos)
            color = self.curColor.astype(np.uint8).reshape(1, 1, 3)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)[0,0]
            self.fluidDC.addFluid(color, mousePos, self.curRadius)
            self.updateSurface()
            self.update(blockEvents = True)


    def pygameEventsHandler(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:

                pressedButtons = pygame.mouse.get_pressed()
                if pressedButtons[0]:
                    self.leftClick()
                elif pressedButtons[2]:
                    self.rightClick()


    def getNewColor(self):
        newColor = tkColorChooser.askcolor(parent=self.master)
        self.curColor = np.array(newColor[0])

    def quit(self):
        self.hasQuit = True
        self.master.destroy()
        pygame.display.quit()
        cv2.destroyAllWindows()
        self.videoWriter.release()
