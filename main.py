# This script demonstrates Gauss-Seidel iteration applied to a 3 by 3 linear system of equations. The algorithm
# is iterated until the square root of mean square error between the actual solution and the result is less than $10^{-5}$

import numpy as np
import math

# Setup for output of arrays as float with 2 digits of accuracy
float_formatter = lambda x: '{: .2f}'.format(x)
np.set_printoptions(formatter={'float_kind':float_formatter})

# Sample linear system, 3 by 3
A = np.matrix([[2, -1, 1], [2, 2, 2], [-1, -1, 2]])
b = np.array([[-1, 4, -5]]).transpose()
x = np.zeros(3)

# Actual solution
y = np.array([[1, 2, -1]]).transpose()

# Gauss-Seidel iteration algorithm
error = 1 # Artificially high error for initialization
while error >= 1e-5:
  w = 0 # Holds error
  for i in range(3): # Loop over row space
    u = 0 # Holds part of estimated solution
    for j in range(3): # Loop over column space
      if i != j: # Skips diagonal terms of A
        u = u - A[i,j]*x[j]
    v = u + b[i]
    x[i] = v/A[i,i] # Estimated solution
    w = w + (x[i] - y[i])**2 # Mean square error
  error = math.sqrt(w)
  print(x, '{:.2e}'.format(error))
