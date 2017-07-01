# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 08:04:35 2013

@author: Tom
"""
import cv2
import numpy as np
#import time
DEFAULTDEFF = .000001
#BASEFRICTION = .99
RGB = "RGB"
LAB = "LAB"
BGR = "BGR"

##Turns out color mixing is a horrible laborious process...
#def sRGB2linearRGB(srgb):
#    """Converts a standard 0-255 sRGB value to the linear sRGB in [0,1]"""
#    a = 0.055
#    rgbLin = []
#    for c in srgb:
#        c = c/255.0
#        if c<=0.04045:
#            rgbLin.append(c/12.92)
#        else:
#            rgbLin.append(((c+a)/(1+a))**2.4)
#    return tuple(rgbLin)
#
#def linearRGB2sRGB(rgbLin):
#    """Converts a linear sRGB in [0,1] to a standard 0-255 sRGB value"""
#    a = 0.055
#    srgb = []
#    for c in rgbLin:
#        if c<=0.0031308:
#            c = c*12.92
#        else:
#            c = (c**(1/2.4))*(1+a) - a
#        srgb.append(int(c*255))
#    return tuple(srgb)
#
#def RGB2XYZ(rgb):
#    """Converts a linear sRGB value to an XYZ value"""
#    return (0.4124*rgb[0]+0.3578*rgb[1]+0.1805*rgb[2]),\
#            (0.2126*rgb[0]+0.7152*rgb[1]+0.0722*rgb[2]),\
#            (0.0193*rgb[0]+0.1192*rgb[1]+0.9502*rgb[2])
#
#def XYZ2RGB(xyz):
#    """Convertsself.fMat = [[0, 0, 0]]*height*width an XYZ value to an linear sRGB value"""
#    return 3.2406*xyz[0] - 1.5372*xyz[1] - 0.4986*xyz[2],\
#            -0.9689*xyz[0] + 1.8758*xyz[1] + 0.0415*xyz[2],\
#            0.0557*xyz[0] - 0.2040*xyz[1] + 1.0570*xyz[2]
#
#XYZWhite = (0.9505, 1.0000, 1.0890)
#
#def _f(t):##helper function
#    if t>(6/29.0)**3:
#        return t**(1/3.0)
#    else:
#        return (1/3.0)*((29/6.0)**2)*t+4/29.0
#
#def _fInv(t):##helper function
#    if t>6/29.0:
#        return t**3
#    else:
#        return 3*((6/29.0)**2)*(t-4/29.0)
#
#def XYZ2LAB(xyz):
#    """Convert a XYZ value to LAB"""
#    return 166*(_f(xyz[1]/XYZWhite[1])) - 16,\
#            500*(_f(xyz[0]/XYZWhite[0])-_f(xyz[1]/XYZWhite[1])),\
#            200*(_f(xyz[1]/XYZWhite[1])-_f(xyz[2]/XYZWhite[2]))
#
#def LAB2XYZ(lab):
#    """Convert a LAB value to XYZ"""
#    return XYZWhite[0]*_fInv((lab[0]+16)/116.0+lab[1]/500.0),\
#            XYZWhite[1]*_fInv((lab[0]+16)/116.0),\
#            XYZWhite[2]*_fInv((lab[0]+16)/116.0-lab[2]/200.0)
#
#def sRGB2LAB(sRGB):
#    return XYZ2LAB(RGB2XYZ(sRGB2linearRGB(sRGB)))
#
#def LAB2sRGB(lab):
#    return linearRGB2sRGB(XYZ2RGB(LAB2XYZ(lab)))

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
#        self.fMat = cv2.cvtColor(inImg, cv2.COLOR_RGB2BGR)
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
#        self.makeInt()


#        self.densMat = [0]*height*width
#        self.getMI = lambda x, y: x+y*width
#        self.veltop = np.array([0,0])
#        self.velbot = np.array([0,0])
#        self.ftop = np.array([0, 0, 0])
#        self.fbot = np.array([0, 0, 0])
#        self.vftop = np.array([0, 0])
#        self.vfbot = np.array([0, 0])
#        self.stop = 1
#        self.sbot = 1
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
        self.fMat[location[0], location[1]] = np.array(fluid)
#
#    def removeFluid(self, location, radius):
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
#        for i in range(self.height/3, 2*self.height/3):
#            for j in range(self.width/3, 2*self.width/3):
#                newMat[i,j,1] = 1
        self.vel_step(newMat, dt)
        self.dens_step(np.zeros((self.height, self.width, 3)), dt)
#        self.fMat = self.fMat.astype(np.uint8)


    def makeInt(self):
        self.fMat = self.fMat.astype(int)

    def add_source(self,x, s, dt):
        return x+dt*s

    def deffuse(self, b, x, dt):

        a = dt*self.deff*self.height*self.width
#        filterMat = np.array(([[0, a, 0],[a, 1, a], [0, a, 0]]))
#        cv2.filter2D(x, -1, filterMat)
#        self.set_bnd(b, x)
#        return x
        newFM = x.copy()
        defInvMat = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])*a

        for k in xrange(20):
            newFM = cv2.filter2D(newFM,-1,defInvMat)
            newFM = (x + newFM)/(1.0+4*a)
            self.set_bnd(b, x)
        return newFM

#        a = dt*self.height*self.width*self.deff
#        newFM = x.copy()
#        for k in range(20):
#            for i in range(1, self.height-1):
#                for j in range(1, self.width-1):
#                    for c in range(3-b):
#                        newFM[i,j,c] = \
#                            (x[i,j,c]+\
#                            a*(newFM[i-1,j,c]+\
#                            newFM[i+1,j,c]+\
#                            newFM[i,j-1,c]+\
#                            newFM[i,j+1,c]))/(1+4*a)
#            self.set_bnd(b, newFM)
#        return newFM

    def advect(self, b, d, dt):
        newMat = d.copy()
        N = int((self.height*self.width)**.5)
        dt0 = dt*N
        jIndexArray = np.array([[[i] for i in range(self.width)]]*self.height)
#        jIndexArray.reshape(self.height, self.width, 1)
        iIndexArray = np.array([[[i]]*self.width for i in range(self.height)])
#        iIndexArray.reshape(self.height, self.width, 1)
#        print iIndexArray.shape, jIndexArray.shape
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
#        print s0.shape, t0.shape, d[i0, j0].shape, i0.shape, j0.shape
        newMat = s0*(t0*(d[i0,j0])\
                    +t1*(d[i0,j1]))\
                    +s1*(t0*(d[i1,j0])\
                    +t1*(d[i1,j1]))


#        for i in xrange(1, self.height-1):
#            for j in xrange(1, self.width-1):
#                x= i-dt0*self.velMat[i,j,0]/self.sMat[i, j]
#                y = j-dt0*self.velMat[i,j,1]
#                if x<0.5:
#                    x = 0.5
#                if x>self.height-2+0.5:
#                    x = self.height-2+0.5
#
#                i0 = int(x)
#
#                i1 = i0 + 1
#
#                if y<0.5:
#                    y = 0.5
#                if y>self.width-2+0.5:
#                    y = self.width-2+0.5
#                j0 = int(y)
#
#                j1 = (j0 + 1)
#
#                s1 = x-i0
#                s0 = 1-s1
#                t1 = (y-j0)%self.width
#                t0 = (1-t1)%self.width
#                newMat[i,j] = s0*(t0*d[i0,j0]\
#                        +t1*d[i0,j1])\
#                        +s1*(t0*d[i1,j0]\
#                        +t1*d[i1,j1])

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
#        self.velMat = self.velMat*BASEFRICTION
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
#        for i in range(1, self.height-1):
#            for j in range(1, self.width-1):
#                    div[i,j] = -0.5*h*(self.velMat[i+1,j,0]\
#                                    -self.velMat[i-1,j,0]\
#                                    +self.velMat[i,j+1,1]\
#                                    -self.velMat[i,j-1,1])*self.sMat[i, j]
        self.set_bnd(0, div)
        self.set_bnd(0, p)
        invMat = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        for k in range(20):
            p = (div + cv2.filter2D(p,-1,invMat))/4.0
            self.set_bnd(0, p)

#        horMat = np.array([[0, 0, 0],[1, 0, -1],[0, 0, 0]])
#        verMat = np.array([[0, 1, 0],[0, 0, 0],[0, -1, 0]])
        horVelMat = .5*cv2.filter2D(p, -1, horMat)/h
        verVelMat = .5*cv2.filter2D(p, -1, verMat)/h
        self.velMat -= np.dstack([verVelMat, horVelMat])
#        for i in range(1, self.height-1):
#            for j in range(1, self.width-1):
#                self.velMat[i,j,0] -= 0.5*(p[i+1,j] - p[i-1,j])/h
#                self.velMat[i,j,1] -= 0.5*(p[i,j+1] - p[i,j-1])/h
#        self.set_bnd(1, self.velMat)

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
