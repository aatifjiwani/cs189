"""
Author: Sophia Sanborn
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas
"""

import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    """Abstract class defining the common interface for all activation methods."""

    def __call__(self, Z):
        return self.forward(Z)

    @abstractmethod
    def forward(self, Z):
        pass


def initialize_activation(name: str) -> Activation:
    """Factory method to return an Activation object of the specified type."""
    if name == "linear":
        return Linear()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return TanH()
    elif name == "relu":
        return ReLU()
    elif name == "softmax":
        return SoftMax()
    else:
        raise NotImplementedError("{} activation is not implemented".format(name))


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = z.

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return Z

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = z.

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        return dY


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for relu activation:
        f(z) = z if z >= 0
               0 otherwise

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return np.maximum(Z, 0)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for relu activation.

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        mask = (Z >= 0).astype(int)
        return np.multiply(dY, mask)


class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for softmax activation.
        Hint: The naive implementation might not be numerically stable.

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        m = np.max(Z, axis=1)
        s_minus_m = Z - m.reshape(-1, 1)
        
        exp_sm = np.exp(s_minus_m)
        
        return exp_sm / np.sum(exp_sm, axis=1).reshape(-1, 1)

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for softmax activation.

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        dLdZ = np.zeros(Z.shape)
        softmaxZ = self.forward(Z)
        
        for row in range(0, softmaxZ.shape[0]):
            curr_sample = softmaxZ[row, :][:, None] #k, 1
            curr_dY = dY[row, :][:, None] #k, 1
            jacobian = np.matmul(-curr_sample, curr_sample.T)
            
            np.fill_diagonal(jacobian, np.array([sig*(1-sig) for sig in curr_sample]))
            
            dLdZ[row, :][:, None] = np.matmul(jacobian, curr_dY)
        
        ### YOUR CODE HERE ###
        return dLdZ


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for sigmoid function:
        f(z) = 1 / (1 + exp(-z))

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### OPTIONAL: YOUR CODE HERE ###
        return ...

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for sigmoid.

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        ### OPTIONAL: YOUR CODE HERE ###
        return ...


class TanH(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = tanh(z).

        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### OPTIONAL: YOUR CODE HERE ###
        return ...

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = tanh(z).

        Parameters
        ----------
        Z   input to `forward` method
        dY  derivative of loss w.r.t. the output of this layer

        Returns
        -------
        derivative of loss w.r.t. input of this layer
        """
        ### OPTIONAL: YOUR CODE HERE ###
        return ...
