# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:29:47 2017

@author: Shabaka
"""

import glob
import os
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
# Read the file into a DataFrame: df
# df_fixation = pd.read_csv('fixations.csv')   # sys.argv[1]
# df_gazepos = pd.read_csv('gaze_pos.csv')     # sys.argv[2]

# ''''''Automate csv call '''''''''''''#

# df_fixation = pd.read_csv(sys.argv[1])
# df_gazepos = pd.read_csv(sys.argv[2])
# df_tac = pd.read_csv('C:\\Users\\Shabaka\\ recordings\\2017_01_03-

# path = r'C:\user\your\path\**'

path = r'C:\Users\Shabaka\recordings\2017_01_03- Dev_Set\002-Exploring Pupil_Video_Dev_Set\exports_Dev_Set\0-4028_Dev_Set'
all_rec = glob.iglob(os.path.join(path, "*.csv"), recursive=True)
dataframes = (pd.read_csv(f) for f in all_rec)
big_dataframe = pd.concat(dataframes, ignore_index=True)


print('\n DataFrame Overview \n')

# print(big_dataframe.head())
print(big_dataframe)

print(big_dataframe.info)


# ##########################################################################

#%%
# print(df_tac.head())

duration = big_dataframe['duration']
# confidence = big_dataframe['avg_pupil_size']
confidence = big_dataframe['confidence']
dispersion = big_dataframe['dispersion']

# gazepos_confidence = df_gazepos['confidence']

df_rand_plot = plt.scatter(x=confidence, y=duration)
plt.xlabel('Fixation Duration ms')
plt.ylabel('Confidence Levels')
plt.margins(0.002)
plt.show()

#%%

big_dataframe.plot(x=("duration"), y=("confidence"),
                   kind="hist")

plt.ylabel('Duration of Fixed Gaze (ms)')
plt.xlabel('Confidence Value:\
           Fixation Integrity >=0.9 is good\
           Pupil Labs Documentation')

plt.title('Relationship between Fixation and Confidence')
plt.show()


big_dataframe.plot(x=("duration"), y=("confidence"),
                   kind="scatter")

plt.ylabel('Duration of Fixed Gaze (ms)')
plt.xlabel('Confidence Value:\
           Fixation Integrity >=0.9 as previous plot')

plt.title('Relationship between Fixation and Confidence')
plt.show()

###############
#%%
\
print("####### Data normality check #######")
\
print("Compute mean and standard deviation: mu, sigma")
\
\
print('\n ######## Computation of Mean and CDF of fixation data #########\n')
\
\
mu = np.mean(duration)
sigma = np.std(duration)

print(mu, " = mean of fixation column from export")
\
print(sigma, " = std_dev of fixtion column from export")
\
mu2 = np.mean(confidence)

sigma2 = np.std(confidence)

print(mu2, " = mean of confidence column from export")
\
print(sigma2, " = std_dev of confidence column from export")
\

#%%
# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, size=10000)

samples2 = np.random.normal(mu2, sigma2, size=10000)
\
\
print(" Using a randomly generated normal distribution of 10k data spread\
      we can abstract a CDF for the mu and sigma values calculated as above")
\
\
# Get the CDF of the samples and of the data - Fixation
\
\
print('###### CDF of Fixation Data - Duration #######')

x_theor, y_theor = ecdf(samples)
x, y = ecdf(duration)


# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('Fixation Duration (ms.)')
_ = plt.ylabel('CDF')
plt.show()

# Get the CDF of the samples2 and of the data - GazePosition
\
\
print('####### CDF of Fixation Data - Average Pupil Size #######')

x_theor2, y_theor2 = ecdf(samples2)
x2, y2 = ecdf(duration)


# Plot the CDFs and show the plot
_ = plt.plot(x_theor2, y_theor2)
_ = plt.plot(x2, y2, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('Avg. Pupil Size')
_ = plt.ylabel('CDF')
plt.show()

#%%
# Take a million samples out of the Normal distribution: samples
\
\
print('\n Calculating the Probability Dist. Function for Fixation <= 0.5 \n')
\
\
samples = np.random.normal(mu, sigma, size=1000000)
\
# Compute the fraction of fixations that are less than 0.5 seconds: prob
prob1 = np.sum(samples <= 0.5)/len(samples)
plt.hist(prob1)
plt.title("PDF for fixations <= 0.5 against confidence")
plt.show()

#%%
# Print the result
print('\n PDF of full Fixation >= 0.5:', prob1)

print('\n Calculating the Probability Dist. Function for Fixation >= 0.5 \n')

samples = np.random.normal(mu, sigma, size=1000000)

# Compute the fraction of fixations that are more than 0.5 seconds: prob
prob2 = np.sum(samples >= 0.5)/len(samples)
plt.hist(prob2, bins=10)
plt.show()
\
\
# Print the result
print('\n PDF of full Fixation >= 0.5:', prob2)

#%%
# Determine successive poisson relationship  - i.e. total time between
# two poisson processes


def successive_poisson(tau1, tau2, size=1):
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size)

    return t1 + t2


# Call samples of fixation: fixating_times
fixating_times = successive_poisson(0.6, 0.45, size=10000)

# Make the histogram
_ = plt.hist(fixating_times, normed=True, histtype='step', bins=100)


# Label axes
plt.xlabel('fixation_cdf')
plt.ylabel('successive_poisson')
plt.title('Plot of fixation times succ.poisson plot')

# Show the plot
plt.show()

print('\n Plot of fixation times succ.poisson plot\n')

#%%
# ##############   EDA ANALYSIS   #########     ######## #

# Plot the fixation duration rate versus average pupil isze
_ = plt.plot(duration, confidence, marker='.', linestyle='none')

# Set the margins and label axes
plt.margins(0.02)
_ = plt.xlabel('average pupil size')
_ = plt.ylabel('fixation duration (ms)')

# Show the plot
plt.show()


#%%
# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(duration, confidence, 1)

# theoretical line to plot
x = np.array([0, 1])
y = a * x + b

# Add regression line to plot
_ = plt.plot(x, y)

plt.show()

# Print the results to the screen
print('slope =', a, '.......')
print('intercept =', b, 'c')

#########################################################################
#%%

"""
# Check the Pearson correlation coefficient
# between fixation dur & confidence
# Need to fix pearson correlation function. func

def pearson_r(x, y):
    # Compute Pearson correlation coefficient between two arrays
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0, 1]
    return corr_mat[0, 1]


print(big_dataframe.head())

# df_fix_x = pd.read_csv('fixations.csv')
df_fix_x = big_dataframe[2:]
# df_fix_y = pd.read_csv('fixations.csv')
df_fix_y = big_dataframe[8:]

print(pearson_r(df_fix_x, df_fix_y))

"""
#%%
#################################################
#################################################
# ############## DATA BOOTSTRAP ################ #
#%%
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
plt.title('Fixation Bootstrap - Empirical Cumulative Distribution Function')
plt.show()

########################################################################
#%%
# # COMPUTE MEAN & SEM OF BOOTSTRAP REPLICATES #### #

# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(duration, np.mean, size=100)

print("Compute and print the Statistical Error of Mean Value SEM.")
print("This shows the spread that the mean of the original data sample")
print("would have if multiple samples of the variable fix_dur are taken")
print("This is calculated using the bootstrapped data samples\n")


print(" This is given by np.std(duration) / np.sqrt(len(duration)\n")
print(np.std(duration) / np.sqrt(len(duration)))

# Compute and print SD of the bootstrapped replicates (Data)
print("\n")
print("Standard Deviation of the Bootstrapped Data\n")
print(np.std(bs_replicates))

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=100, normed=True)
_ = plt.xlabel('mean fixation duration (ms)')
_ = plt.ylabel('PDF')

plt.title('Probability Distribution Fxn of "Mean Fixation Durations" ')
plt.show()


#%%
# ###  BOOTSTRAP CONFIDENCE INTERVALS ##### #

bs_percentiles = np.percentile(bs_replicates, [2.5, 97.5])

print('Percentiles - Bootstrapped Data Conf. Intervals =', bs_percentiles)

###################################################################
#%%
# ########### BOOTSTRAP OF VARIANCE ######## #

# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates2 = draw_bs_reps(confidence, np.var, size=10000)

# Put the variance in units of cm^2
bs_replicates2 /= 100

# Make a histogram of the results
_ = plt.hist(bs_replicates2, bins=100, normed=True)
_ = plt.xlabel('variance of avg_pupil_size data')
_ = plt.ylabel('PDF')

# Show the plot
plt.title('Prob. Distr. Function (Variance of pupil size - Bootstrapped Data)')
plt.show()

#################################################################
#%%
# ###  BOOTSTRAP CONFIDENCE INTERVALS - VARIANCE ##### #

bs_percentiles2 = np.percentile(bs_replicates2, [2.5, 97.5])

print('95% - Bootstrap Conf. Inter. Var. =', bs_percentiles2, '...')

#%%
##################################################################
# ###  PLOTTING BOOTSTRAPPED DATA REGRESSIONS ###### #

# Generate array of x-values for bootstrap lines: x
x = np.array([0, 10])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(draw_bs_pairs_linreg(duration, confidence, size=10),
                 linewidth=0.5, alpha=0.2, color='red')

#     _ = plt.plot(x, bs_slope_reps[i] * x + bs_intercept_reps[i],
#                 linewidth=0.5, alpha=0.2, color='red')

# draw_bs_pairs_linreg
# Plot the data
_ = plt.plot(duration, confidence, marker='.', linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('duration')
_ = plt.ylabel('avg_pupil_size data')
plt.margins(0.02)
plt.title('Pupil Diameter vs Duration of Gaze - BootStrapped Data Regressions')
plt.show()
#%%
####################################################################
####################################################################
#%%
# ########## STATISTICAL PARAMETER ESTIMATION ########## #

# Seed random number generator
np.random.seed(1000)

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


#%%
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


#%%
# ########### SENSE CHECK! - is the parameteer optimal? ########
\
print('Is the Parameter Optimal?')
# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('between fixations')
plt.ylabel('Cumulative Distribution Function')

plt.title('Check Parameter is Optimal\nAre multivariate random behaviour optimal?')
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
#%%

print("A negative correlation coefficient means that")
print("for any two variables X and Y, an increase in X is associated")
print("with a decrease in Y.")
print("A negative correlation demonstrates a connection between two variables")
print("in the same way a positive correlation coefficient does,")
print("and the relative strengths are the same")
\
# Plot the fix_duration rate versus average pupil size
_ = plt.plot(confidence, duration, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('avg_pupil_size')
_ = plt.ylabel('duration')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(confidence, duration, 1)

# Print the results to the screen
print('slope =', a, 'saccade density / percent fixations')
print('intercept =', b, 'avg.pupil_size density')


#%%

# Make theoretical line to plot
x = np.array([0, 1])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

_ = plt.title('Regression Output')
# Draw the plot
plt.show()

#%%
"""
# ################  PLOTTING THE PERMUTATION SAMPLES
for i in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(duration, confidence)

    # Compute ECDFs
    x_1, y_1 = ecdf(duration)
    x_2, y_2 = ecdf(confidence)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)
plt.margins(0.02)
_ = plt.xlabel('fixation time (ms)')
_ = plt.ylabel('ECDF')
plt.show()
"""
#%%
# Create and plot ECDFs from original data
x_1, y_1 = ecdf(duration)
x_2, y_2 = ecdf(confidence)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('fixation time (ms)')
_ = plt.ylabel('ECDF')
plt.show()

#%%
