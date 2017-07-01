# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 08:04:35 2013

@author: Tom
"""
import cv2
import numpy as np
DEFAULTDEFF = .000001
RGB = "RGB"
LAB = "LAB"
BGR = "BGR"


def LAB2RGB(labImage):
    return cv2.cvtColor(labImage, cv2.COLOR_LAB2RGB)

def RGB2LAB(rgbImage):
    return cv2.cvtColor(rgbImage, cv2.COLOR_RGB2LAB)

class FluidDoubleCone():
    """This simulates a double cone with fluid dynamics.
        The edge conditions for the sides simply wrap around,
        while the top and bottom each attach to an extra point.
        Pixel "sizes" can be adjusted to approximate several isomorphic shapes.
        (In particular, a sphere.)
        Color blending is done in the LAB color range. This conversion is done
        through openCV. """
    def __init__(self, inImg, imgColorSpace):

        height = len(inImg)
        width = len(inImg[0])
        self.height = height
        self.width = width
        self.vf = np.zeros((height, width, 2)) ##These are forces that get reapplied every generation
        if imgColorSpace != LAB:
            self.fMat = cv2.cvtColor(inImg, eval("cv2.COLOR_"+imgColorSpace+"2LAB"))
        else:
            self.fMat = inImg
        self.deff = DEFAULTDEFF
        self.sMat = np.ones((height, width)) ##pixel size matrix
        self.velMat = np.zeros((height, width, 2))
        self.set_bnd(0, self.fMat)
        self.set_bnd(1, self.velMat)
        self.isSphere = False
        self.sphereW = 0
        self.sphereR = 0

    def copy(self):
        retCopy = FluidDoubleCone(self.fMat, LAB)
        retCopy.setVectorField(self.velMat.copy())
        if self.isSphere:
            retCopy.makeLikeSphere(self.sphereW*self.sphereR, self.sphereR)
        return retCopy

    def __getitem__(self, index):
        if len(index)!=2:
            if index == "top":
                return self.fTop, self.veltop, self.vfTop
            raise IndexError("Index must be of the form fDC[x, y]")
        return self.fMat[index[0]+index[1]*self.width],\
                self.velMat[index[0]+index[1]*self.width],\
                self.vf[index[0]+index[1]*self.width],\
                self.deffMat[index[0]+index[1]*self.width]

    def addFluid(self, fluid, location, radius):
        self.fMat[int(location[0]), int(location[1])] = np.array(fluid)

#    def removeFluid(self, location, radius): TODO
#        return

    def addCoriolis(self, dt):
                ##TODO: Look into this function more. May be able to optimize
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                f = 2*self.sphereW*np.sin(np.pi*(i/float(self.height-1)-1/2.0))
                self.velMat[i, j, 0] += -dt*self.velMat[i, j, 1]*f
                self.velMat[i, j, 1] += dt*self.velMat[i, j, 0]*f


    def setFluidMat(self, fluidMat):
        self.fMat = fluidMat

    def setVectorField(self, vectorField):
        self.velMat = vectorField

    def spinLikeSphere(self, spinSpeed):
        for i in range(self.height):
            for j in range(self.width):
                self.velMat[i,j,1] += spinSpeed*np.cos(np.pi*(i/float(self.height)-1/2.0))

    def makeLikeSphere(self, spinSpeed, radius):
        self.isSphere = True
        self.sphereR = radius
        self.sphereW = spinSpeed/float(radius)
                ##TODO: Look into this function more. May be able to optimize
        for i in range(self.height):
            for j in range(self.width):
#                self.velMat[i,j,1] = spinSpeed*np.cos(np.pi*(i/float(self.height)-1/2.0))
                self.sMat[i, j] = radius*(np.cos(np.pi*(i/float(self.height)-1/2.0)))
        self.set_bnd(0, self.sMat)

    def centerSquareToRight(self, speed):
                ##TODO: Look into this function more. May be able to optimize
        for i in range(self.height/3, 2*self.height/3):
            for j in range(self.width/3, 2*self.width/3):
                self.velMat[i,j,1] = speed

    def allToRight(self, speed):
                ##TODO: Look into this function more. May be able to optimize
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                self.velMat[i,j,1] += speed

    def swirlAt(self, speed, radius, location):
                ##TODO: Look into this function more. May be able to optimize
        def swirlFunc(loc):
            x = loc[0]
            y = loc[1]
            if x**2 + y**2 > swirlFunc.radius**2:
                return np.array([0,0])
            else:
                return swirlFunc.speed*np.array([y, -x])/float(swirlFunc.radius)
        swirlFunc.speed = speed
        swirlFunc.radius = radius
        for i in range(self.height):
            for j in range(self.width):
                self.velMat[i, j] = swirlFunc([i-location[0], j-location[1]])

    def moveHorizontalStripToRight(self, speed, width, location):
        ##TODO: Look into this function more. May be able to optimize
        for i in range(int(location-width), int(location+width)):
            for j in range(self.width):
                self.velMat[i, j] = speed

    def update(self, dt):
        newMat = np.zeros((self.height, self.width, 2))
        if self.isSphere:
            self.addCoriolis(dt)
        self.vel_step(newMat, dt)
        self.dens_step(np.zeros((self.height, self.width, 3)), dt)
#        self.fMat = self.fMat.astype(np.uint8)


    def makeInt(self):
        self.fMat = self.fMat.astype(int)

    def add_source(self,x, s, dt):
        return x+dt*s

    def deffuse(self, b, x, dt):

        a = dt*self.deff*self.height*self.width
        newFM = x.copy()
        defInvMat = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])*a

        for k in xrange(20):
            newFM = cv2.filter2D(newFM,-1,defInvMat)
            newFM = (x + newFM)/(1.0+4*a)
            self.set_bnd(b, x)
        return newFM


    def advect(self, b, d, dt):
        newMat = d.copy()
        N = int((self.height*self.width)**.5)
        dt0 = dt*N
        jIndexArray = np.array([[[i] for i in range(self.width)]]*self.height)
        iIndexArray = np.array([[[i]]*self.width for i in range(self.height)])
        x= iIndexArray-dt0*self.velMat[0:,0:,:1]#/self.sMat
        y = jIndexArray-dt0*self.velMat[0:,0:,1:]#/self.sMat
        tmp = x<0.5
        x[tmp] = 0.5
        tmp = x>self.height-2+0.5
        x[tmp] = self.height-2+0.5

        i0 = x.astype(int)

        i1 = i0 + 1

        tmp = y<0.5
        y[tmp] = 0.5
        tmp = y>self.width-2+0.5
        y[tmp] = self.width-2+0.5

        j0 = y.astype(int)

        j1 = (j0 + 1)

        s1 = x-i0
        s0 = 1-s1
        t1 = (y-j0)%self.width
        t0 = (1-t1)%self.width
        i0.shape = (self.height, self.width)
        j0.shape = (self.height, self.width)
        i1.shape = (self.height, self.width)
        j1.shape = (self.height, self.width)
        newMat = s0*(t0*(d[i0,j0])\
                    +t1*(d[i0,j1]))\
                    +s1*(t0*(d[i1,j0])\
                    +t1*(d[i1,j1]))
        self.set_bnd(b, newMat)
        return newMat

#    def deffuseVel(self, b, dt):
#        return
#
#    def advectVel(self, b, dt):
#        return

    def dens_step(self, x0, dt):
        self.fMat = self.add_source(self.fMat, x0, dt)
        self.fMat = self.deffuse(0, self.fMat, dt)
        self.fMat = self.advect(0, self.fMat, dt)

    def vel_step(self, v0, dt):
        self.velMat = self.add_source(self.velMat, v0, dt)
        self.velMat = self.deffuse(1, self.velMat, dt)
        self.project(dt)
        self.velMat = self.advect(1, self.velMat, dt)
        self.project(dt)


    def project(self, dt):
        div = np.zeros((self.height, self.width))
        p = np.zeros((self.height, self.width))
        N = (self.height*self.width)**.5
        h = 1.0/N
        horMat = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        verMat = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
        div = -0.5*h*(cv2.filter2D(self.velMat[0:,0:, 1],-1, horMat)\
                        +cv2.filter2D(self.velMat[0:,0:,0],-1, verMat))
        self.set_bnd(0, div)
        self.set_bnd(0, p)
        invMat = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        for k in range(20):
            p = (div + cv2.filter2D(p,-1,invMat))/4.0
            self.set_bnd(0, p)

        horVelMat = .5*cv2.filter2D(p, -1, horMat)/h
        verVelMat = .5*cv2.filter2D(p, -1, verMat)/h
        self.velMat -= np.dstack([verVelMat, horVelMat])

    def set_bnd(self, b, x):
        ##TODO: Look into this function more. May be able to optimize
        if b == 1:##1 if velocity matrix
            for i in range(1, self.width-1):
                x[0,i,0] = -x[1,i,0]
                x[self.height-1,i,0] = -x[self.height-2,i,0]
                x[0,i,1] = x[1,i,1]
                x[self.height-1,i,1] = x[self.height-2,i,1]
            for j in range(1, self.height-1):
                x[j,0,0] = x[j,self.width-2,0]
                x[j,self.width-1,0] = x[j,1,0]
                x[j,0,1] = x[j,self.width-2,1]
                x[j,self.width-1,1] = x[j,1,1]
        elif b == 0:
            for i in range(1, self.width-1):
                x[0,i] = x[1,i]
                x[self.height-1,i] = x[self.height-2,i]

            for j in range(1, self.height-1):
                x[j,0] = x[j,self.width-2]
                x[j,self.width-1] = x[j,1]

        x[0,0] = 0.5*(x[1,0]+x[0,1])
        x[0,self.width-1] = 0.5*(x[1,self.width-1]+x[0,self.width-2])
        x[self.height-1,0] = 0.5*(x[self.height-2,0]+x[self.height-1,1])
        x[self.height-1,self.width-1] = 0.5*(x[self.height-2,self.width-1]+\
                                        x[self.height-1,self.width-2])
