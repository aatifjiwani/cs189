import matplotlib.pyplot as plt
import numpy as np

def multivariate_gaussian(X, mu, sigma):
    d = mu.shape[0]
    det = np.linalg.det(sigma)
    denominator = (2*np.pi)**(d/2) * np.sqrt(det)
    exponent = -0.5 * (X - mu).T.dot(np.linalg.inv(sigma)).dot((X - mu))
    return float( (1/denominator) * np.exp(exponent) )

def generateZ(mu, sigma):
    Z = np.empty((150,150))
    for i in range(0, 150):
        for j in range(0, 150):
            Z[i,j] = multivariate_gaussian(plot[i,j,:].reshape(2,1), mu, sigma)

    return Z

def plotGaussian(X, Y, Z):
    plt.contourf(X, Y, Z, 20, cmap="viridis")
    plt.colorbar()
    plt.axes().set_aspect('equal')
    plt.xlim([-7,7])
    plt.ylim([-7,7])
    plt.show()
    plt.waitforbuttonpress()
    ## Save Plot 

plot = np.empty((150,150,2))
X,Y = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))
plot[:,:,0] = X
plot[:,:,1] = Y

## 3.1
def q3_1():
    mu = np.array([1,1]).reshape(2,1)
    sigma = np.array([1,0,0,2]).reshape(2,2)
    Z = generateZ(mu, sigma)
    plotGaussian(X,Y,Z)

def q3_2():
    mu = np.array([-1,2]).reshape(2,1)
    sigma = np.array([2,1,1,4]).reshape(2,2)
    Z = generateZ(mu, sigma)
    plotGaussian(X,Y,Z)

def q3_3():
    mu1 = np.array([0,2]).reshape(2,1)
    mu2 = np.array([2,0]).reshape(2,1)

    sigma = np.array([2,1,1,1]).reshape(2,2)

    Z1 = generateZ(mu1, sigma)
    Z2 = generateZ(mu2, sigma)

    plotGaussian(X,Y,Z1 - Z2)

def q3_4():
    mu1 = np.array([0,2]).reshape(2,1)
    mu2 = np.array([2,0]).reshape(2,1)

    sigma1 = np.array([2,1,1,1]).reshape(2,2)
    sigma2 = np.array([2,1,1,4]).reshape(2,2)

    Z1 = generateZ(mu1, sigma1)
    Z2 = generateZ(mu2, sigma2)

    plotGaussian(X,Y,Z1 - Z2)


def q3_5():
    mu1 = np.array([1,1]).reshape(2,1)
    mu2 = np.array([-1,-1]).reshape(2,1)

    sigma1 = np.array([2,0,0,1]).reshape(2,2)
    sigma2 = np.array([2,1,1,2]).reshape(2,2)

    Z1 = generateZ(mu1, sigma1)
    Z2 = generateZ(mu2, sigma2)

    plotGaussian(X,Y,Z1 - Z2)

# q3_1()
# q3_2()
# q3_3()
# q3_4()
q3_5()
    