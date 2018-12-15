#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:33:21 2018

@author: virati
MayaVI tutorial

"""

import numpy
from mayavi.mlab import *

def test_points3d():
    t = np.linspace(0, 4 * np.pi, 20)

    x = np.sin(2 * t)
    y = np.cos(t)
    z = np.cos(2 * t)
    s = 2 + np.sin(t)

    return points3d(x, y, z, s, colormap="copper", scale_factor=.25)

test_points3d()


## Create the data.
#from numpy import pi, sin, cos, mgrid
#dphi, dtheta = pi/250.0, pi/250.0
#[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
#m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
#r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
#x = r*sin(phi)*cos(theta)
#y = r*cos(phi)
#z = r*sin(phi)*sin(theta)
#
## View it.
#from mayavi import mlab
#s = mlab.mesh(x, y, z)
#mlab.show()