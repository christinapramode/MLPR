#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal

############################################# Parameters #############################################
# Priori Probabilites
PY = [0.7, 0.3]

# Gaussian PDF Parameters
mu_0 = np.array([-1, 1, -1, 1])
mu_1 = np.array([1, 1, 1, 1])

# For Part A
# sigma_0 = np.array([[2, -0.5, 0.3, 0],
#                     [-0.5, 1, -0.5, 0],
#                     [0.3, -0.5, 1, 0],
#                     [0, 0, 0, 2]])
# sigma_1 = np.array([[1, 0.3, -0.2, 0],
#                     [0.3, 2, 0.3, 0],
#                     [-0.2, 0.3, 1, 0],
#                     [0, 0, 0, 3]])

# For Part B
sigma_0 = np.array([[2, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 2]])
sigma_1 = np.array([[1, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 3]])

dimensions = 4
num_samples = 10000
num_samples_0 = int(PY[0] * num_samples)
num_samples_1 = int(PY[1] * num_samples)

############################################# Data Generation #############################################
true_labels = np.hstack((np.zeros(num_samples_0), np.ones(num_samples_1)))

class0_samples = np.random.multivariate_normal(mean=mu_0, cov=sigma_0, size=num_samples_0)
class1_samples = np.random.multivariate_normal(mean=mu_1, cov=sigma_1, size=num_samples_1)
samples = np.vstack((class0_samples, class1_samples))

# print(true_labels.shape)
# print(samples.shape)
# for i, label in enumerate(true_labels):
#     print("i=",i, "label=",label, "samples:",samples[i])

####################################### Classification and ROC Curve ######################################
# Calculate Likelihoods using Gaussian PDF
fX_given_Y0 = multivariate_normal.pdf(samples, mean=mu_0, cov=sigma_0)
fX_given_Y1 = multivariate_normal.pdf(samples, mean=mu_1, cov=sigma_1)

# Classification and Calculation of TP, FP, TN, FN, Perror
thresholds = np.log(np.logspace(-18,18,1000))
TP_list = []
FP_list = []
Perror_list = []

for threshold in thresholds:
    # Decision Rule
    decisions = (np.log(fX_given_Y1) - np.log(fX_given_Y0)) > threshold
    
    # Get true & false positives and negatives
    TP = np.sum((decisions == 1) & (true_labels == 1)) / num_samples_1
    FP = np.sum((decisions == 1) & (true_labels == 0)) / num_samples_0
    TN = np.sum((decisions == 0) & (true_labels == 0)) / num_samples_0
    FN = np.sum((decisions == 0) & (true_labels == 1)) / num_samples_1
    TP_list.append(TP)
    FP_list.append(FP)

    # Error Probability Calculation
    Perror = FP * PY[0] + FN * PY[1]
    Perror_list.append(Perror)

# Empirical Minimum Error Probability Calculation
min_error_index = np.argmin(Perror_list)
empirical_thresh = math.exp(thresholds[min_error_index])
empirical_min_Perror = Perror_list[min_error_index]
empirical_str = '(' +str(round(FP_list[min_error_index],4))+','+str(round(TP_list[min_error_index],4))+'), γ='+str(round(empirical_thresh,4))

# Theoretical Minimum Error Probability Calculation
theoretical_thresh = PY[0] / PY[1]
decisions = (fX_given_Y1 / fX_given_Y0) > theoretical_thresh
theoretical_TP = np.sum((decisions == 1) & (true_labels == 1)) / num_samples_1
theoretical_FP = np.sum((decisions == 1) & (true_labels == 0)) / num_samples_0
theoretical_TN = np.sum((decisions == 0) & (true_labels == 0)) / num_samples_0
theoretical_FN = np.sum((decisions == 0) & (true_labels == 1)) / num_samples_1
theoretical_min_Perror = theoretical_FP * PY[0] + theoretical_FN * PY[1]
theoretical_str = '('+str(round(theoretical_FP,4))+','+str(round(theoretical_TP,4))+'), γ='+str(round(theoretical_thresh,4))

# ROC Plot
fig = plt.figure(figsize=(8, 6))
plt.plot(FP_list, TP_list, linewidth=2, label='ROC Curve')
plt.scatter(FP_list[min_error_index], TP_list[min_error_index], color='red', label='Empirical Min P_error='+str(round(empirical_min_Perror,4))+','+empirical_str)
plt.scatter(theoretical_FP, theoretical_TP, color='green', label='Theoretical Min P_error='+str(round(theoretical_min_Perror,4))+','+theoretical_str)
plt.xlabel('False Positive Probability')
plt.ylabel('True Positive Probability')
plt.title('ROC Curve')
plt.xlim(-0.05,1)
plt.ylim(-0.05,1)
plt.grid()
plt.legend(loc='lower right')
plt.show()
