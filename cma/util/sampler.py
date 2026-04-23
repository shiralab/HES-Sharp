#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

# public symbols
__all__ = ['Sampler']


class Sampler(object):
    def __init__(self, f, lam):
        self.f = f
        self.lam = lam

    def __call__(self, model):
        X = model.sampling(self.lam)
        X_enc = model.encoding(self.lam, X)
        evals = self.f(X_enc)
        return X, evals

    def verbose_display(self):
        return ''

    def log_header(self):
        return []

    def log(self):
        return []
    
class LST_sampler(Sampler):
    def __call__(self, model,s):
        X = model.sampling(self.lam)
        a = np.eye(X.size)
        b = np.eye(X.size) * -1
        c = np.zeros([1, X.size])
        Delta = np.concatenate([a, b, c], axis=0)
        F_lst = np.array([self.f(X + s * delta) for delta in Delta])
        return X, np.array([F_lst.max()])
    
    def __call__(self,x,s):
        a = np.eye(x.size)
        b = np.eye(x.size) * -1
        c = np.zeros([1, x.size])
        Delta = np.concatenate([a, b, c], axis=0)
        F_lst = np.array([self.f(x + s * delta) for delta in Delta])
        return x, np.array([F_lst.max()])