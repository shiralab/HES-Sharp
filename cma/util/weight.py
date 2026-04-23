#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.stats import rankdata


__all__ = ['CMANonIncFunc', 'SelectionNonIncFunc', 'QuantileBasedWeight', 'CMAWeight']


class CMANonIncFunc(object):
    def __init__(self):
        pass

    @staticmethod
    def calc_weight(q):
        q[q < 1e-300] = 1e-300
        w = - 2. * np.log(2. * q)
        w[q > 0.5] = 0.
        return w

    @staticmethod
    def tie_component(q):
        w = np.zeros_like(q)
        mask = (q != 0)
        w[mask] = np.minimum(q[mask], 0.5) * (CMANonIncFunc.calc_weight(q[mask]) + 2.)
        return w

    def __call__(self, q_plus, q_minus=None):
        if q_minus is None:
            return self.calc_weight(q_plus)

        q_diff = q_plus - q_minus
        idx = (q_diff != 0)

        weights = np.zeros_like(q_plus)
        weights[~idx] = CMANonIncFunc.calc_weight(q_plus[~idx])
        weights[idx] = (CMANonIncFunc.tie_component(q_plus[idx]) - CMANonIncFunc.tie_component(q_minus[idx])) / q_diff[idx]
        return weights


class CMAWeight(object):
    def __init__(self, lam, min_problem=True):
        self.lam = lam
        self.min_problem = min_problem
        self.w = np.maximum(np.log((self.lam + 1.)/2.) - np.log(np.arange(self.lam)+1.), np.zeros(self.lam))
        self.w = self.w / self.w.sum() if self.w.sum() != 0 else self.w
        self.weights = np.zeros_like(self.w)

    def __call__(self, evals):
        evals = evals if self.min_problem else -evals
        index = np.argsort(evals)
        self.weights[index] = self.w

        # tie case check
        unique_val, count = np.unique(evals, return_counts=True)
        if len(evals) == len(unique_val):
            return self.weights

        # tie case: averaging
        for u_val in unique_val[count > 1]:
            duplicate_index = np.where(evals == u_val)
            self.weights[duplicate_index] = self.weights[duplicate_index].mean()
        return self.weights