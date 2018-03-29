# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:36:15 2017

@author: Shabaka
"""

import numpy as np
import itertools


def permutation_sample(data_1, data_2):
    """Generate a permutation sample from two data sets."""
    data_1 = itertools.permutations(data_1)
    data_2 = itertools.permutations(data_2)

    # Concatenate the data sets: full_data
    full_data = np.concatenate((data_1, data_2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(full_data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(full_data)]
    perm_sample_2 = permuted_data[len(full_data):]

    return perm_sample_1, perm_sample_2