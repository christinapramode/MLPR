#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random

############################################# Parameters #############################################
# Priori Probabilites
PY = [0.3, 0.3, 0.4]

num_classes = 3
dimensions = 3

# Gaussian PDF Parameters
num_gaussians = 4

sigma = np.zeros((dimensions, dimensions, num_gaussians))
mean = np.zeros((dimensions, num_gaussians))

# Setting covariance matrix as diagonal random matrix
for i in range(num_gaussians):
    sigma[:, :, i] = [[random.random(), 0, 0],
                    [0, random.random(), 0],
                    [0, 0, random.random()]]
    
# Set distance between means as 2x standard deviation
avg_std_dev = np.mean([np.sqrt(sigma[d, d, i]) for i in range(num_gaussians) for d in range(dimensions)])
distance = 2 * avg_std_dev
mean[:, 0] = [0, 0, 0]
mean[:, 1] = [distance, 0, 0]
mean[:, 2] = [0, distance, 0]
mean[:, 3] = [distance, distance, 0]

# Printing the mean and covariance matrix values
for i in range(num_gaussians):
    print('Covariance Matrix ', i, ': ', sigma[:,:,i])
    print('Mean ', i, ': ', mean[:,i])

# Calculating number of samples in each class
num_samples = 10000
num_samples_pc = np.zeros(num_classes, dtype=int)
for i in range(num_classes):
    num_samples_pc[i] = int(PY[i] * num_samples)

############################################# Data Generation #############################################
true_labels = np.hstack((np.ones(num_samples_pc[0],dtype=int), 
                         2*np.ones(num_samples_pc[1],dtype=int), 
                         3*np.ones(num_samples_pc[2],dtype=int)))

class1_samples = np.random.multivariate_normal(mean=mean[:,0], cov=sigma[:,:,0], size=num_samples_pc[0])
class2_samples = np.random.multivariate_normal(mean=mean[:,1], cov=sigma[:,:,1], size=num_samples_pc[1])
class3_samples_1 = np.random.multivariate_normal(mean=mean[:,2], cov=sigma[:,:,2], size=num_samples_pc[2]//2)
class3_samples_2 = np.random.multivariate_normal(mean=mean[:,3], cov=sigma[:,:,3], size=num_samples_pc[2]//2)
class3_samples = np.vstack((class3_samples_1, class3_samples_2))
samples = np.vstack((class1_samples, class2_samples, class3_samples))

# Visualize True Label Distribution
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples[true_labels == 1, 0], samples[true_labels == 1, 1], samples[true_labels == 1, 2], marker='o', s=3, color='r', label='Class 1')
ax.scatter(samples[true_labels == 2, 0], samples[true_labels == 2, 1], samples[true_labels == 2, 2], marker='^', s=3, color='b', label='Class 2')
ax.scatter(samples[true_labels == 3, 0], samples[true_labels == 3, 1], samples[true_labels == 3, 2], marker='s', s=3, color='g', label='Class 3')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('True Label Distributions')
ax.legend()

############################################## Classification ##############################################
# Define PDFs
pdf1 = multivariate_normal(mean=mean[:,0], cov=sigma[:,:,0])
pdf2 = multivariate_normal(mean=mean[:,1], cov=sigma[:,:,1])
pdf3_part1 = multivariate_normal(mean=mean[:,2], cov=sigma[:,:,2])
pdf3_part2 = multivariate_normal(mean=mean[:,3], cov=sigma[:,:,3])

# Function to implement Classifier based on decision rule
def bayes_classifier(x, loss_matrix):
    pxgiven1 = pdf1.pdf(x)
    pxgiven2 = pdf2.pdf(x)
    pxgiven3 = 0.5 * pdf3_part1.pdf(x) + 0.5 * pdf3_part2.pdf(x)

    losses = np.zeros(num_classes)
    for i in range(num_classes):
        losses[i] = PY[0] * loss_matrix[i, 0] * pxgiven1 \
                  + PY[1] * loss_matrix[i, 1] * pxgiven2 \
                  + PY[2] * loss_matrix[i, 2] * pxgiven3
    
    return np.argmin(losses) + 1

# Classify samples
loss_matrix_01 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
decision_labels_01 = np.array([bayes_classifier(sample, loss_matrix_01) for sample in samples])
loss_matrix_10 = np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]])
decision_labels_10 = np.array([bayes_classifier(sample, loss_matrix_10) for sample in samples])
loss_matrix_100 = np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]])
decision_labels_100 = np.array([bayes_classifier(sample, loss_matrix_100) for sample in samples])

# Visualization
def visualize_distribution(decision_labels):
    # Compute confusion matrix: P(D=i│Y=j) ∀ i,j∈{1,2,3} 
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(1, num_classes+1):
        for j in range(1, num_classes+1):
            confusion_matrix[i-1, j-1] = np.sum((decision_labels == i) & (true_labels == j)) / num_samples_pc[j-1]
    print("Confusion Matrix:\n", confusion_matrix)
    column_sums = np.sum(confusion_matrix, axis=0)
    print("Column sums:", column_sums)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    markers = ['o', '^', 's']

    for i, label in enumerate([1, 2, 3]):        
        idx = (true_labels == label)
        correct = (true_labels == decision_labels)
        
        ax.scatter(samples[idx & correct, 0], samples[idx & correct, 1], samples[idx & correct, 2],
                marker=markers[label-1], color='green', label=f'Class {label} Correct', s=3)
        ax.scatter(samples[idx & ~correct, 0], samples[idx & ~correct, 1], samples[idx & ~correct, 2],
                marker=markers[label-1], color='red', label=f'Class {label} Incorrect', s=3)
        
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title('Correct vs Incorrect Classification')
    ax.legend()

def visualize_class_distributions(decision_labels):
    markers = ['o', '^', 's']
    for i, label in enumerate([1, 2, 3]):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        idx = (true_labels == label)
        correct = (true_labels == decision_labels)

        ax.scatter(samples[idx & correct, 0], samples[idx & correct, 1], samples[idx & correct, 2],
                marker=markers[label-1], color='green', label='Correct', s=3)
        ax.scatter(samples[idx & ~correct, 0], samples[idx & ~correct, 1], samples[idx & ~correct, 2],
                marker=markers[label-1], color='red', label='Incorrect', s=3)

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.set_title('Class'+str(label) + ': Correct vs Incorrect Classification')
        ax.legend()

visualize_distribution(decision_labels_01)
visualize_class_distributions(decision_labels_01)
# visualize_distribution(decision_labels_10)
# visualize_distribution(decision_labels_100)

plt.show()
