import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


np.random.seed(7)

def generateRandomSample(num_samples):
    n = np.zeros((num_samples, 2)).astype(np.float32)
    X1 = np.random.normal(loc=3, scale=3, size=100)
    X2 = 0.5*X1 + np.random.normal(loc=4, scale=2, size=100)

    n[:, 0] = X1
    n[:, 1] = X2

    return n


n = generateRandomSample(100)
print()

mean_sample = np.mean(n, axis=0)
print("-- Mean of Sample --")
print(mean_sample, end='\n\n')

cov_matrix = np.cov(n.T)
print("-- Covariance Matrix --")
print(cov_matrix, end='\n\n')

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("-- Eigenvector V1 with associated Lambda 1 --")
print("V1:", eigenvectors[:,0], "with eigenvalue", eigenvalues[0])
print()
print("-- Eigenvector V2 with associated Lambda 2 --")
print("V2:", eigenvectors[:,1], "with eigenvalue", eigenvalues[1])

# plt.scatter(n[:,0], n[:,1], color='darkblue', s=6)
# plt.arrow(*mean_sample, *(eigenvalues[1]*eigenvectors[:,1]), color='darkred', width=0.1)
# plt.arrow(*mean_sample, *(eigenvalues[0]*eigenvectors[:,0]), color='green', width=0.1)
# plt.title("Scatter of X1, X2 with eigenvectors scaled by eigenvalues")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.xlim([-15, 15])
# plt.ylim([-15, 15])
# plt.grid()
# plt.axes().set_aspect('equal')
# plt.show()
# plt.waitforbuttonpress()

maxEigenvector = np.argmax(eigenvalues)
v1 = eigenvectors[:,maxEigenvector].reshape(2,1)
v2 = eigenvectors[:,1 - maxEigenvector].reshape(2,1)

U = np.concatenate((v1,v2), axis=1)
centered_n = n - mean_sample
rotated_n = centered_n.dot(U)

plt.scatter(rotated_n[:,0], rotated_n[:,1], color='darkblue', s=6)
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.grid()
plt.title("Rotated and mean-centered X1 and X2")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axes().set_aspect('equal')
plt.show()
plt.waitforbuttonpress()

