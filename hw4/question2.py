import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def makeDiag(s):
    shape = s.shape[0]
    eye = np.eye(shape)
    for i in range(shape):
        eye[i][i] = s[i]

    return eye

def computeSecondGradient(s, X):
    dsigmoid = s * (1 - s)
    diagS = makeDiag(dsigmoid)
    hessian = np.matmul( np.matmul(X.T, diagS), X )
    inv_hessian = np.linalg.inv(hessian)

    return inv_hessian

def computeFirstGradient(X, y, s):
    y_s = y - s
    return np.matmul(X.T, y_s)

x1 = np.array([0.2, 3.1, 1.0]).reshape(1, -1)
x2 = np.array([1.0, 3.0, 1.0]).reshape(1, -1)
x3 = np.array([-0.2, 1.2, 1.0]).reshape(1, -1)
x4 = np.array([1.0, 1.1, 1.0]).reshape(1, -1)

y = np.array([1, 1, 0, 0]).reshape(-1, 1) #(4,1)
X = np.concatenate((x1, x2, x3, x4), axis=0) #(4, 3)
w = np.array([-1.0, 1.0, 0]).reshape(-1, 1) #(3, 1)


## s(0)
s = sigmoid(np.matmul(X, w)) #(4,1)
print(s)
print()

## w(1)
grad = computeFirstGradient(X, y, s)
inv_hessian = computeSecondGradient(s, X)
w = w + np.matmul(inv_hessian, grad)
print(w)
print()

## s(1)
s = sigmoid(np.matmul(X, w))
print(s)
print()

## w(2)
grad = computeFirstGradient(X, y, s)
inv_hessian = computeSecondGradient(s, X)
w = w + np.matmul(inv_hessian, grad)
print(w)
print()

print(sigmoid(np.matmul(X, w)))
