# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:09:05 2018

@author: sami
"""

import numpy as np

arr = np.array([[1, 5, 3],
              [5, 3, 5],
              [3, 2, 0]])
print(arr[np.argsort(arr[:, 1])])