#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

############################## Dataset and Parameter Estimation ##############################
# Load the dataset
wine_data = pd.read_csv('winequality-white.csv', delimiter=';')

# Extract Features and Labels
X = wine_data.drop(columns=['quality'])
true_labels = wine_data['quality'].values

n_samples = X.shape[0]
n_dimensions = X.shape[1]
n_classes = 11
print('Num samples: ', n_samples, ', Num dimensions: ', n_dimensions, ', Num classes: ', n_classes)

# Calculate mean, covariance and priors for each class
class_means = []
class_covariances = []
class_priors = []
n_class_samples = []
for c in range(n_classes):
    X_c = X[true_labels == c]

    num_samples = X_c.shape[0]    
    if (num_samples > 0):
        mean = np.mean(X_c, axis=0)
        cov = np.cov(X_c, rowvar=False)
        prior = num_samples / n_samples
    else:
        mean = np.zeros(n_dimensions)
        cov = np.zeros((n_dimensions, n_dimensions))
        prior = 0.0

    class_means.append(mean)
    class_covariances.append(cov)
    class_priors.append(prior)
    n_class_samples.append(num_samples)

# Regularize covariance matrix
def regularize_covariance(cov, lambda_val):
    return cov + lambda_val * np.identity(cov.shape[0])
regularized_covariances = [regularize_covariance(cov, 0.01) for cov in class_covariances]

class_means = np.array(class_means)
class_covariances = np.array(regularized_covariances)
class_priors = np.array(class_priors)
n_class_samples = np.array(n_class_samples)

print('Priors: ', class_priors)
print('N Class Samples: ', n_class_samples)
################################ Classifier Implementation ################################
# Function to implement Classifier based on decision rule
def bayes_classifier(X, means, covariances, priors):
    likelihoods = np.zeros((n_samples, n_classes))
    for i in range(n_classes):
        mean = means[i]
        cov = covariances[i]
        prior = priors[i]
        mvn = multivariate_normal(mean=mean, cov=cov)
        likelihoods[:, i] = mvn.pdf(X) * prior
    return np.argmax(likelihoods, axis=1)

# Classify samples
decision_labels = bayes_classifier(X.values, class_means, class_covariances, class_priors)

# Compute confusion matrix: P(D=i│Y=j) ∀ i,j∈{0,1,....10} 
confusion_matrix = np.zeros((n_classes, n_classes))
for i in range(n_classes):
    for j in range(n_classes):
        if (n_class_samples[j] > 0):
            confusion_matrix[i, j] = np.sum((decision_labels == i) & (true_labels == j)) / n_class_samples[j]
        else:
            confusion_matrix[i, j] = 0.0
print("Confusion Matrix:\n", confusion_matrix)
column_sums = np.sum(confusion_matrix, axis=0)
print("Column sums:", column_sums)

# Compute error probability
sum_errors = np.sum(decision_labels != true_labels)
error_probability = sum_errors / n_samples
print(f'Error Probability: ', error_probability)

# Visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
colors = ['brown', 'grey', 'pink', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for c in range(n_classes):
    ax.scatter(X_pca[true_labels == c, 0], X_pca[true_labels == c, 1], X_pca[true_labels == c, 2], 
               label=f'Class {c}', alpha=0.5, color=colors[c])

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.legend()
ax.set_title('Wine Dataset PCA Projection')
plt.show()
