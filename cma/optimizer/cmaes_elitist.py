#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import gmean
import sys

from ..optimizer.base_optimizer import BaseOptimizer
from ..util.model import Gaussian

# public symbols
__all__ = ['CMAES_elitist']

class CMAES_elitist(BaseOptimizer):
    """修正版：モデルとオプティマイザの責務を分離"""
    def __init__(self, d, sampler, m=None, C=None, sigma=1., minimal_eigenval=1e-30,
                 c_cov=None, c_p=None, damping=None, normalize='None', 
                 min_problem=True, tie_success=True,seed = None):
        
        # ★sigmaをGaussianモデルの初期化時に渡す
        self.model = Gaussian(d, m=m, C=C, sigma=sigma, minimal_eigenval=minimal_eigenval, normalize=normalize,seed=seed)
        self.sampler = sampler
        self.d = d
        # ... (他のパラメータは変更なし) ...
        self.is_better = (lambda x, y: x < y) if min_problem else (lambda x, y: x > y)
        self.best_X, self.best_eval = m, sampler.f(np.array([m]))
        self.damping = 1. + d / 2. if damping is None else damping
        self.c_p = 1. / 12. if c_p is None else c_p
        self.p_target = 2. / 11. 
        self.p_succ = self.p_target
        self.c_cov = 2. / (d ** 2 + 6.) if c_cov is None else c_cov
        self.c_c = 2. / (2. + d)
        self.p_thresh = 0.44
        self.pc = np.zeros(d)
        self.gen_count = 0
        self.tie_success = tie_success
        self.model.lam = 1
    
    def log_header(self):
        return self.model.log_header()

    def log(self):
        return self.model.log()
    
    def terminate_condition(self):
        return self.model.terminate_condition()

    def sampling_model(self):
        # ★samplerが期待する「モデルオブジェクト」を返すように修正
        return self.model
    
    def update_step_size(self):
        # ★sigmaの更新はモデルの属性を直接変更する
        self.model.sigma *= np.exp((self.p_succ - self.p_target) / (1. - self.p_target) / self.damping)

    def update(self, X, evals):
        self.gen_count += 1
        if self.best_X is None:
            self.best_X, self.best_eval, self.model.m = X[0], evals[0], X[0]
            return
        
        lam_succ = self.is_better(evals[0], self.best_eval)
        eq_lam_succ = (evals[0] == self.best_eval)
        lam_succ = lam_succ or (eq_lam_succ and self.tie_success)
        self.p_succ = (1. - self.c_p) * self.p_succ + self.c_p * lam_succ
        
        if lam_succ:
            # ★yの計算ではモデルのsigmaを使用
            y = (X[0] - self.model.m) / self.model.sigma
            self.update_cov(y)
            self.best_X, self.best_eval, self.model.m = X[0], evals[0], X[0]

        self.update_step_size()

    def update_cov(self, y):
        # ... (このメソッドの中身は変更なし) ...
        if self.p_succ < self.p_thresh:
            self.pc = (1. - self.c_c) * self.pc + np.sqrt(self.c_c * (2. - self.c_c)) * y
            self.model.C = (1. - self.c_cov) * self.model.C + self.c_cov * np.outer(self.pc, self.pc)
        else:
            self.pc = (1. - self.c_c) * self.pc
            self.model.C = (1. - self.c_cov) * self.model.C + self.c_cov * (np.outer(self.pc, self.pc) + self.c_c * (2. - self.c_c) * self.model.C)
    
    def run(self, sampler, logger=None, verbose=False, max_eval = None):
        f = sampler.f
        if logger is not None:
            logger.write_csv(['Ite'] + f.info_header() + self.log_header() + sampler.log_header())

        ite = 0
        while not sampler.f.terminate_condition() and not self.terminate_condition():
            ite += 1

            # sampling and evaluation
            X, evals = sampler(self.sampling_model())

            # display and save log
            if verbose:
                print(str(ite) + f.verbose_display() + self.verbose_display() + sampler.verbose_display())
            if logger is not None:
                logger.write_csv([str(ite)] + f.info_list() + self.log() + sampler.log())

            # parameter update
            self.update(X, evals)
            if max_eval is not None and ite > max_eval:
                break
        return [f.eval_count, self.best_eval, f.is_success(), self.model.m, ite]