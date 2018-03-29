# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:55:18 2017

@author: Shabaka
"""

import numpy as np
import matplotlib.pyplot as plt

# empirical cummulative distribution function


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y