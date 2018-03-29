# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:36:24 2017

@author: Shabaka
"""
import csv

# open the file in universal line ending mode

with open('fixations.csv', 'rU') as infile:
    # read the file as a dictionary for each row ({header : value})
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]


def getColumn(filename, column):
    results = csv.reader(open('fixations.csv'), delimiter=",")
    return [result[column] for result in results]


import numpy as np
import matplotlib
# Import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
matplotlib.style.use('ggplot')


fixation = 'fixations.csv'
gaze_pos = 'gaze_postions.csv'

xfix = getColumn('fixations.csv', 2)
yfix = getColumn('fixations.csv', 8)

plt.scatter(xfix, yfix)
plt.show()

