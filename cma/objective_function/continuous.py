#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
module for continuous optimization problems
"""

import numpy as np
from ..objective_function.base import *


# public symbols
__all__ = ['SharpRidge', 'RidgeWithCorner', 'HcAlphaFunction','Sphere']

class SharpRidge(ObjectiveFunction):
    """
    sharp ridge function 
    f(x) = x_1 + d_coeff * (Σ_{i=2 to dim} x_i^2)^alpha
    """
    minimization_problem = True

    def __init__(self, dim, d_coeff=100., alpha=0.5, target_eval=None, max_eval=1e7):
        """
        :param dim: 
        :param d_coeff: 
        :param alpha: 
        :param target_eval: 
        :param max_eval: 
        """
        self.dim = dim
        self.d_coeff = d_coeff
        self.alpha = alpha
        
        if target_eval is None:
            target_eval = -1e6
        super(SharpRidge, self).__init__(target_eval, max_eval, dim)

    def __call__(self, X):
        self.eval_count += len(X)
    
        if X.ndim == 1:
            term1 = X[0]
            sum_of_squares = np.sum(X[1:] ** 2)
        else:
            term1 = X[:, 0]
            sum_of_squares = np.sum(X[:, 1:] ** 2, axis=1)
        
        term2 = self.d_coeff * (sum_of_squares ** self.alpha)

        evals = term1 + term2
        self._update_best_eval(evals,X)
        
        return evals

class RidgeWithCorner(ObjectiveFunction):
    """
    Ridge with Corner
    f(X,a,d) = x_1 + d * ((x_2 - |x_1|)^2 + Σ_{i=3 to N}x_i^2)^a
    """
    minimization_problem = True 

    def __init__(self, dim, d_coeff=100., alpha=0.5, target_eval=None, max_eval=1e7):
        if dim < 2:
            raise ValueError("RidgeWithCorner function requires at least 2 dimensions.")
        self.dim = dim
        self.d_coeff = d_coeff
        self.alpha = alpha

        if target_eval is None:
            target_eval = -1e6
        super(RidgeWithCorner, self).__init__(target_eval, max_eval, dim)

    def __call__(self, X):
        self.eval_count += len(X) if X.ndim > 1 else 1

        if X.ndim == 1:
            x1 = X[0]
            x2 = X[1]
            
            term1 = x1
            term_corner = (x2 - np.abs(x1)) ** 2
            
            sum_of_squares_rest = 0
            if self.dim > 2:
                sum_of_squares_rest = np.sum(X[2:] ** 2)
            
            inside_parenthesis = term_corner + sum_of_squares_rest
            term2 = self.d_coeff * (inside_parenthesis ** self.alpha)

        else:
            x1 = X[:, 0]
            x2 = X[:, 1]

            term1 = x1
            term_corner = (x2 - np.abs(x1)) ** 2
            
            sum_of_squares_rest = 0
            if self.dim > 2:
                sum_of_squares_rest = np.sum(X[:, 2:] ** 2, axis=1)

            inside_parenthesis = term_corner + sum_of_squares_rest
            term2 = self.d_coeff * (inside_parenthesis ** self.alpha)
        
        evals = term1 + term2
        self._update_best_eval(evals, X)

        return evals
    

class HcAlphaFunction(ObjectiveFunction):
    """
    HappyCat Function revised
    f(x) = [ (Q - 2S)^2 ]^alpha + Q / (2N)
    """
    minimization_problem = True

    def __init__(self, dim, alpha=0.5, target_eval=None, max_eval=1e7):
        self.dim = dim
        self.alpha = alpha
        self.bestX = np.ones(dim) * np.inf
        
        if target_eval is None:
            target_eval = 1e-8
        super(HcAlphaFunction, self).__init__(target_eval, max_eval, dim)
    
    def verbose_display(self):
        return ' EvalCount: %d' % self.eval_count + ' BestEval: %e' % self.best_eval

    def __call__(self, X):
        self.eval_count += len(X) if X.ndim > 1 else 1
        
        N = self.dim
        if N == 0:
            return np.nan

        if X.ndim == 1:
            # Q = ||x||^2
            Q = np.dot(X, X)
            # S = sum(x_i)
            S = np.sum(X)
            
        else:
            Q = np.sum(X ** 2, axis=1)
            S = np.sum(X, axis=1)

        base_term1 = (Q - 2 * S) ** 2
        term1 = np.power(base_term1, self.alpha)
        term2 = Q / (2 * N)
        evals = term1 + term2
        
        np.set_printoptions(suppress=True, precision=17)

        self._update_best_eval(evals, X)
        
        return evals
    
class Sphere(ObjectiveFunction):
    minimization_problem = True

    def __init__(self, dim, target_eval=None, max_eval=1e7):
        self.dim = dim
        
        if target_eval is None:
            target_eval = 1e-8
        super(Sphere, self).__init__(target_eval, max_eval, dim)

    def __call__(self, X):
        self.eval_count += len(X)
        
        if X.ndim == 1:
            evals = np.sum(X ** 2)
        else:
            evals = np.sum(X ** 2, axis=1)
        
        self._update_best_eval(evals,X)
        
        return evals