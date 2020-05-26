#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:56:02 2020

@author: virati
quick PCA example
"""

import numpy as np
import matplotlib.pyplot as plt

M = np.array([1.0,-4.0]).reshape(-1,1).T

x = np.random.uniform(-10,10,(100,2))

y = np.dot(M,x.T).reshape(-1,1)

#plt.scatter(x[:,0].reshape(-1,1),y.T)

import sklearn.linear_model
from sklearn.decomposition import PCA

model = PCA()
model.fit(np.hstack((x,y)))
print(model.components_)