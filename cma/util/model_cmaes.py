#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
import sys
import numpy as np
import scipy.linalg

_EPS = 1e-8

# public symbols
__all__ = ['Model']


class Model(object):
    """
    Base class for models
    """

    @abstractmethod
    def sampling(self, lam):
        """
        Abstract method for sampling.
        :param int lam: sample size :math:`\\lambda`
        :return: samples
        """
        pass

    @abstractmethod
    def loglikelihood(self, X):
        """
        Abstract method for log likelihood.
        :param X: samples
        :return: log likelihoods
        """
        pass

    def terminate_condition(self):
        """
        Check terminate condition.
        :return bool: terminate condition is satisfied or not
        """
        return False

    def verbose_display(self):
        """
        Return verbose display string.
        :return str: string for verbose display
        """
        return ''

    def log_header(self):
        """
        Return model log header list.
        :return: header info list for model log
        :rtype string list:
        """
        return []

    def log(self):
        """
        Return model log string list.
        :return: model log string list
        :rtype string list:
        """
        return []
    
    def encoding(self, lam, X):
        """
        連続値問題なので、エンコードは不要。
        受け取った値をそのまま返す（恒等変換）。
        """
        return X


class Gaussian(Model):
    """
    Gaussian distribution parametrized by mean vector :math:`m` and (full) covariance matrix :math:`C`.

    :param int d: dimension
    :param m: mean vector :math:`m` (option, default is numpy.zeros(d))
    :param C: covariance matrix :math:`C` (option, default is numpy.identity(d))
    :param float minimal_eigenval: minimal eigenvalue for terminate condition
    :type m: array_like, shape(d), dtype=float
    :type C: array_like, shape(d, d), dtype=float
    """

    def __init__(self, d, m=None, C=None, minimal_eigenval=1e-30):
        self.d = d
        self.m = m if m is not None else np.zeros(self.d)
        self.C = C if C is not None else np.identity(self.d)
        self.min_eigenval = minimal_eigenval

        if len(self.m) != d or self.C.shape != (d, d):
            print("The size of parameters is invalid.")
            print("Dimension: %d, Mean vector: %d, Covariance matrix: %s" % (self.d, len(self.m), self.__C.shape))
            print("at " + self.__class__.__name__ + " class")
            sys.exit(1)

    def _get_C(self):
        return self.__C

    def _set_C(self, C):
        self.__C = C
        self.__eigen_decomposition()

    C = property(_get_C, _set_C)

    def sampling(self, lam):
        """
        Draw :math:`\\lambda` samples from the Gaussian distribution.

        :param int lam: sample size :math:`\\lambda`
        :return: sampled vectors from :math:`\\mathcal{N}(m, C)` Gaussian distribution
        :rtype: array_like, shape=(lam, d), dtype=float
        """
        return np.random.randn(lam, self.d).dot(self.sqrtC.T) + self.m

    def loglikelihood(self, X):
        """
        Calculate log likelihood.

        :param X: samples
        :type X: array_like, shape=(lam, d), dtype=float
        :return: log likelihoods
        :rtype: array_like, shape=(lam), dtype=float
        """
        Z = np.dot((X - self.m), self.invSqrtC.T)
        return - 0.5 * (self.d * np.log(2. * np.pi) + self.logDetC) - 0.5 * np.linalg.norm(Z, axis=1)**2

    def terminate_condition(self):
        return np.min(self.eigvals) < self.min_eigenval

    def verbose_display(self):
        return ' MinEigVal: %e' % (np.min(self.eigvals))

    def log_header(self):
        # mean, eigenvalues, eigenvectors (flattened row-major), log determinant
        eigvec_headers = [f'evec{i}_dim{j}' for i in range(self.d) for j in range(self.d)]
        return (
            ['m%d' % i for i in range(self.d)]
            + ['eigval%d' % i for i in range(self.d)]
            + eigvec_headers
            + ['logDetC']
        )

    def log(self):
        eigvec_flat = self.eigvectors.reshape(-1)
        return (
            ['%e' % i for i in self.m]
            + ['%e' % i for i in self.eigvals]
            + ['%e' % v for v in eigvec_flat]
            + ['%e' % self.logDetC]
        )

    # Private method
    def __eigen_decomposition(self):
        self.eigvals, self.eigvectors = scipy.linalg.eigh(self.C)
        B = self.eigvectors
        self.eigvals = np.where(self.eigvals < _EPS, _EPS, self.eigvals)

        if np.min(self.eigvals) > 0.:
            D = np.diag(np.sqrt(self.eigvals))
            self.sqrtC = np.dot(np.dot(B, D), B.T)
            # self.invC = np.dot(np.dot(B, np.diag(np.reciprocal(self.eigvals))), B.T)
            self.invSqrtC = np.dot(np.dot(B, np.diag(np.reciprocal(np.sqrt(self.eigvals)))), B.T)
            self.logDetC = np.log(self.eigvals).sum()
        else:
            print('The minimal eigenvalue becomes negative value!')


class GaussianSigmaC(Gaussian):
    def __init__(self, d, m=None, C=None, sigma=1., minimal_eigenval=1e-30):
        super(GaussianSigmaC, self).__init__(d, m=m, C=C, minimal_eigenval=minimal_eigenval)
        self.sigma = sigma
        self.lam = None  # set by optimizer to know population size for logging
        self.last_samples = None  # store latest sampled candidates for logging

    def sampling(self, lam):
        samples = self.sigma * np.random.randn(lam, self.d).dot(self.sqrtC.T) + self.m
        self.last_samples = samples
        return samples

    def loglikelihood(self, X):
        Z = np.dot((X - self.m), self.invSqrtC.T) / self.sigma
        return - 0.5 * (self.d * np.log(2. * np.pi) + self.logDetC) - np.log(self.sigma) - 0.5 * np.linalg.norm(Z, axis=1)**2

    def terminate_condition(self):
        return (self.sigma**2) * np.min(self.eigvals) < self.min_eigenval

    def verbose_display(self):
        return ' MinEigVal: %e' % ((self.sigma**2) * (np.min(self.eigvals)))

    def log_header(self):
        base = super(GaussianSigmaC, self).log_header() + ['sigma']
        if self.lam is None:
            return base
        sample_headers = [f'x{j}_dim{i}' for j in range(self.lam) for i in range(self.d)]
        return base + sample_headers

    def log(self):
        base = super(GaussianSigmaC, self).log() + ['%e' % self.sigma]
        if self.last_samples is None or self.lam is None:
            return base
        flat_samples = self.last_samples.reshape(-1)
        return base + ['%e' % v for v in flat_samples]


class Bernoulli(Model):
    """
    Bernoulli distribution for binary string parametrized by :math:`\\{ \\theta \\}_{i=1}^d`.

    :param int d: the number of bits
    :param theta: parameter vector for Bernoulli distribution :math:`\\theta`, the range is [0.0, 1.0] (option, default is 0.5 * numpy.ones(d))
    :type theta: array_like, shape(d), dtype=float
    """

    def __init__(self, d, theta=None):
        self.d = d
        self.theta = theta if theta is not None else 0.5 * np.ones(self.d)

        if len(self.theta) != d or (self.theta > 1).sum() > 0 or (self.theta < 0).sum() > 0:
            print("The initial parameters (theta) are invalid.")
            print("Dimension: %d, Dim of theta: %d" % (self.d, len(self.theta)))
            print("The range of parameters is [0.0, 1.0].")
            print("theta = " + self.theta)
            print("at " + self.__class__.__name__ + " class")
            sys.exit(1)

    def sampling(self, lam):
        """
        Draw :math:`\\lambda` samples from the Bernoulli distribution.
        :param int lam: sample size :math:`\\lambda`
        :return: sampled vectors from Bernoulli distribution
        :rtype: array_like, shape=(lam, d), dtype=bool
        """
        rand = np.random.rand(lam, self.d)
        return rand < self.theta

    def loglikelihood(self, X):
        """
        Calculate log likelihood.

        :param X: samples
        :type X: array_like, shape=(lam, d), dtype=bool
        :return: log likelihoods
        :rtype: array_like, shape=(lam), dtype=float
        """
        return (X * np.log(self.theta) + (1 - X) * np.log(1. - self.theta)).sum(axis=1)

    def verbose_display(self):
        return ''
        # return '\n' + str(self.theta)

    def log_header(self):
        return ['theta%d' % i for i in range(self.d)]

    def log(self):
        return ['%f' % i for i in self.theta]


class Categorical(Model):
    """
    Categorical distribution for categorical variables

    :param categories: the numbers of categories
    :type categories: array_like, shape(d), dtype=int
    :param theta: parameter vector for categorical distribution :math:`\\theta`, the range is [0.0, 1.0]
    :type theta: array_like, shape(d, Cmax), dtype=float
    """
    def __init__(self, categories, theta=None):
        self.N = np.sum(categories - 1)
        self.d = len(categories)
        self.C = categories
        self.Cmax = np.max(categories)
        if theta is not None:
            self.theta = theta
        else:
            self.theta = np.zeros((self.d, self.Cmax))
            # initialize theta by 1/C for each dimensions
            for i in range(self.d):
                self.theta[i, :self.C[i]] = 1. / self.C[i]

        # pad zeros to unused elements
        for i in range(self.d):
            self.theta[i, self.C[i]:] = 0.

        # valid dimension size
        self.valid_d = len(self.C[self.C > 1])

        if self.theta.shape != (self.d, self.Cmax) or (self.theta > 1).sum() > 0 or (self.theta < 0).sum() > 0:
            print("The initial parameters (theta) are invalid.")
            print("Dimension: %d, Max num. of categories: %d," % (self.d, self.Cmax))
            print("Shape of theta: " + str(self.theta.shape))
            print("The range of parameters is [0.0, 1.0].")
            print("theta = ")
            print(self.theta)
            print("at " + self.__class__.__name__ + " class")
            sys.exit(1)

    def sampling(self, lam):
        """
        Draw :math:`\\lambda` samples from the categorical distribution.
        :param int lam: sample size :math:`\\lambda`
        :return: sampled variables from the categorical distribution (one-hot representation)
        :rtype: array_like, shape=(lam, d, Cmax), dtype=int
        """
        rand = np.random.rand(lam, self.d, 1)    # range of random number is [0, 1)
        cum_theta = self.theta.cumsum(axis=1)    # (d, Cmax)
        X = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return X

    def loglikelihood(self, X):
        """
        Calculate log likelihood.

        :param X: samples (one-hot representation)
        :type X: array_like, shape=(lam, d, Cmax), dtype=bool
        :return: log-likelihoods
        :rtype: array_like, shape=(lam), dtype=float
        """
        return (X * np.log(self.theta)).sum(axis=2).sum(axis=1)

    def verbose_display(self):
        return ''
        # return '\n' + str(self.theta)

    def log_header(self):
        return ['theta%d_%d' % (i//self.Cmax, i % self.Cmax) for i in range(self.d * self.Cmax) if i % self.Cmax < self.C[i//self.Cmax]]

    def log(self):
        return ['%f' % theta for (i, theta) in enumerate(self.theta) if i % self.Cmax < self.C[i//self.Cmax]]