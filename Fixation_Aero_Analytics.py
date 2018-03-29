# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:29:47 2017

@author: Shabaka
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt
from ecdf_func import ecdf
from draw_bootstrap_reps import draw_bs_reps
from LinReg_BS_Pairs_func import draw_bs_pairs_linreg
import sys
from Perm_Sample_Gen3 import permutation_sample

from Permutation_Replicates_func import draw_perm_reps

# Define the dataframe - fixation data csv

# C:\Users\Shabaka\ShabakaCodes\28-01-17-Export_0-14433 Test Scripts
# Assign the filename: file
# fixation = 'C:\\Users\\Shabaka\\ShabakaCodes\\\
# 28-01-17-Export_0-14433 Test Scripts\\fixations.csv'
# gaze_pos = 'C:\\Users\\Shabaka\\ShabakaCodes\\\
# 28-01-17-Export_0-14433 Test Scripts\\gaze_postions.csv'

# Read the file into a DataFrame: df
df_fixation = pd.read_csv('fixations.csv')   # sys.argv[1]
df_gazepos = pd.read_csv('gaze_positions.csv')     # sys.argv[2]
# df_fixation = pd.read_csv(sys.argv[1])

# df_gazepos = pd.read_csv(sys.argv[2])
# df_tac = pd.read_csv('C:\\Users\\Shabaka\\ recordings\\2017_01_03- Dev_Set\\002-Exploring Pupil_Video_Dev_Set\\exports_Dev_Set\\0-4028_Dev_Set\\aerodata.csv')
# View the head of the DataFrame

print(df_fixation.head())

# print(df_tac.head())

duration = df_fixation['duration']
avg_pupilsize = df_fixation['avg_pupil_size']
confidence = df_fixation['confidence']
dispersion = df_fixation['dispersion']

gazepos_confidence = df_gazepos['confidence']

df_rand_plot = plt.scatter(x=avg_pupilsize, y=duration)
plt.xlabel('Fixation Duration ms')
plt.ylabel('Avg. Pupil_Size Levels')
plt.margins(0.002)
plt.show()

# ''''''''' Works fine to this point '''''''#

fixation = genfromtxt('fixations.csv', delimiter=',', usecols=(2, 9),
                      unpack=True, dtype=int)

gazepos = genfromtxt('gaze_positions.csv', delimiter=',', usecols=(2, 9),
                     unpack=True, dtype=int)

# ############## CHECK NORMALITY OF DATA ########### #

# Compute mean and standard deviation: mu, sigma

print('Computation of Mean and CDF of fixation data')
mu = np.mean(fixation)
sigma = np.std(fixation)

mu2 = np.mean(avg_pupilsize)
sigma2 = np.std(avg_pupilsize)

# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, size=10000)

samples2 = np.random.normal(mu2, sigma2, size=10000)


# Get the CDF of the samples and of the data - Fixation
print('CDF of Fixation Data - Duration of Fix')

x_theor, y_theor = ecdf(samples)
x, y = ecdf(fixation)


# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('Fixation Duration (ms.)')
_ = plt.ylabel('CDF')
plt.show()

# Get the CDF of the samples2 and of the data - GazePosition
print('CDF of Fixation Data - Duration of Fix')

x_theor2, y_theor2 = ecdf(samples2)
x2, y2 = ecdf(avg_pupilsize)


# Plot the CDFs and show the plot
_ = plt.plot(x_theor2, y_theor2)
_ = plt.plot(x2, y2, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('Pupil Size (ms.)')
_ = plt.ylabel('CDF')
plt.show()

# Code works to this point - Plot doesn't make sense yet - why -ve values?


# Take a million samples out of the Normal distribution: samples

print('Calculating the Probability Dist. Function for Fixation <= 0.5')

samples = np.random.normal(mu, sigma, size=1000000)

# Compute the fraction of fixations that are longer than 0.5 seconds: prob
prob1 = np.sum(samples <= 0.5)/len(samples)

plt.hist(prob1)
plt.show()

# Print the result
print('PDF of full Fixation <= 0.5:', prob1)

print('Calculating the Probability Dist. Function for Fixation >= 0.5')

samples = np.random.normal(mu, sigma, size=1000000)
 
# Compute the fraction of fixations that are longer than 0.5 seconds: prob
prob2 = np.sum(samples >= 0.5)/len(samples)

plt.hist(prob2)
plt.show()

# Print the result
print('PDF of full Fixation >= 0.5:', prob2)

# Works to this point - Make sense of correlation # #

# Determine successive poisson relationship  - i.e. total time between
# two poisson processes


def successive_poisson(tau1, tau2, size=1):
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size)

    return t1 + t2


# Draw samples of fixation: fixating_times
fixating_times = successive_poisson(0.6, 0.45, size=10000)

# Make the histogram
_ = plt.hist(fixating_times, normed=True, histtype='step', bins=100)


# Label axes
plt.xlabel('fixation_cdf')
plt.ylabel('successive_poisson')


# Show the plot
plt.show()

# ##############   EDA ANALYSIS   #########     ######## #

df_fix_x = pd.read_csv('fixations.csv')
# fixation[:, 2]
df_fix_y = pd.read_csv('fixations.csv')
# fixation[:, 9]


# Plot the fixation duration rate versus average pupil isze
_ = plt.plot(duration, avg_pupilsize, marker='.', linestyle='none')

# Set the margins and label axes
plt.margins(0.02)
_ = plt.xlabel('average pupil size')
_ = plt.ylabel('fixation duration (ms)')

# Show the plot
plt.show()

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(duration, avg_pupilsize, 1)

# Make theoretical line to plot
x = np.array([0, 1])
y = a * x + b

# Add regression line to plot
_ = plt.plot(x, y)

plt.show()

# Print the results to the screen
print('slope =', a, '.......')
print('intercept =', b, 'c')

# Show the Pearson correlation coefficient - WRITE PEARSON CORR. func
# print(pearson_r(fix_x, fix_y))


# ############## DATA BOOTSTRAP ################ #

for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(duration, len(duration))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(duration)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('fixation duration(ms)')
_ = plt.xlabel('ECDF')

# Show the plot
plt.show()


# # COMPUTE MEAN & SEM OF BOOTSTRAP REPLICATES #### #

# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(duration, np.mean, size=100)

# Compute and print SEM
print(np.std(duration) / np.sqrt(len(duration)))

# Compute and print standard deviation of bootstrap replicates
print(np.std(bs_replicates))

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean fixation duration (ms)')
_ = plt.ylabel('PDF')

plt.show()

# ###  BOOTSTRAP CONFIDENCE INTERVALS ##### #

bs_percentiles = np.percentile(bs_replicates, [2.5, 97.5])

print('%tiles - Bootstrap Conf. Intervals =', bs_percentiles, '...')


# ########### BOOTSTRAP OF VARIANCE ######## #

# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates2 = draw_bs_reps(duration, np.var, size=10000)

# Put the variance in units of square centimeters
bs_replicates /= 100

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('variance of total fixation (ms')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# ###  BOOTSTRAP CONFIDENCE INTERVALS - VARIANCE ##### #

bs_percentiles2 = np.percentile(bs_replicates2, [2.5, 97.5])

print('95% - Bootstrap Conf. Inter. Var. =', bs_percentiles2, '...')


# #########  PLOTTING BOOTSTRAP REGRESSIONS ###### #

# Generate array of x-values for bootstrap lines: x
x = np.array([0, 10])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(draw_bs_pairs_linreg(duration, avg_pupilsize, size=10),
                 linewidth=0.5, alpha=0.2, color='red')

#     _ = plt.plot(x, bs_slope_reps[i] * x + bs_intercept_reps[i],
#                 linewidth=0.5, alpha=0.2, color='red')

# draw_bs_pairs_linreg
# Plot the data
_ = plt.plot(duration, avg_pupilsize, marker='.', linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('fix_duration')
_ = plt.ylabel('Average_Pupil_Size')
plt.margins(0.02)
plt.show()


# ########## STATISTICAL PARAMETER ESTIMATION ########## #

# Seed random number generator
np.random.seed(42)

# Compute mean no-hitter time: tau
tau = np.mean(duration)

# Draw out of an exponential distribution with
# parameter tau: inter_fix_time
inter_fixation_time = np.random.exponential(tau, 10000)

# Plot the PDF and label axes
_ = plt.hist(inter_fixation_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Sacc_density between fixations')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# ########## DOES THE DATA FOLLOW THE STORY  ########### #

# Create an ECDF from real data: x, y
x, y = ecdf(duration)

# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_fixation_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('between fixations')
plt.ylabel('CDF')

# Show the plot
plt.show()

# ########### SENSE CHECK! - is the parameteer optimal? ########

print('Is the Parameter Optimal?')
# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('between fixations')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2, 10000)

# Take samples with double tau: samples_double
samples_double = np.random.exponential(2*tau, 10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

# Plot these CDFs as lines
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)

# Show the plot
plt.show()

# ################## ####################### ################# #

# Plot the fix_duration rate versus average pupil size
_ = plt.plot(avg_pupilsize, duration, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('fixation den %')
_ = plt.ylabel('avg.pupil_size')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(avg_pupilsize, duration, 1)

# Print the results to the screen
print('slope =', a, 'saccade density / percent fixations')
print('intercept =', b, 'avg.pupil_size density')

# Make theoretical line to plot
x = np.array([0, 1])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()

"""
# ################  PLOTTING THE PERMUTATION SAMPLES
for i in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(duration, confidence)

    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(fixation)
x_2, y_2 = ecdf(avg_pupilsize)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('fixation time (ms)')
_ = plt.ylabel('ECDF')
plt.show()
"""