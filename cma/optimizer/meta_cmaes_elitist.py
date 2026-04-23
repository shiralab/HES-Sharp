#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import gmean
import sys

from ..optimizer.base_optimizer import BaseOptimizer
from ..optimizer.cmaes_elitist import CMAES_elitist
from ..util.model import Gaussian
import cma.objective_function.continuous as cont

# public symbols
__all__ = ['Meta_CMAES_elitist']

class Meta_CMAES_elitist(BaseOptimizer):
    """修正版：モデルとオプティマイザの責務を分離"""
    def __init__(self, d, meta_sampler,inner_sampler, m=None, C=None, sigma=1., minimal_eigenval=1e-30,
                 c_cov=None, c_p=None, damping=None, normalize='None', 
                 min_problem=True, tie_success=True):
        
        # ★sigmaをGaussianモデルの初期化時に渡す
        self.model = Gaussian(d, m=m, C=C, sigma=sigma, minimal_eigenval=minimal_eigenval, normalize=normalize)
        self.meta_sampler = meta_sampler
        self.inner_sampler = inner_sampler
        self.d = d
        # ... (他のパラメータは変更なし) ...
        self.is_better = (lambda x, y: x < y) if min_problem else (lambda x, y: x > y)
        self.best_X, self.best_eval = None, None
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
    
    def log_header(self):
        pass
        # Cの対角成分を全てログに保存
        # diag_headers = [f"c{i}{i}" for i in range(self.d)]
        # return ["sigma", "best_X", "optimized_bestx", "best_eval", "init_mean", "optimized_mean", "optimized_evals"] + diag_headers

    def log(self,init_mean, optimized_mean,optimized_evals,optimized_bestx):
        pass
        # # Cの対角成分を全て取得
        # diag_values = [self.model.C[i][i] for i in range(self.d)]
        # return [self.model.sigma, self.best_X, optimized_bestx, self.best_eval, init_mean, optimized_mean, optimized_evals] + diag_values
    
    def terminate_condition(self):
        return self.model.terminate_condition()

    def sampling_model(self):
        # ★samplerが期待する「モデルオブジェクト」を返すように修正
        return self.model
    
    def update_step_size(self):
        # ★sigmaの更新はモデルの属性を直接変更する
        self.model.sigma *= np.exp((self.p_succ - self.p_target) / (1. - self.p_target) / self.damping)

    def verbose_display(self):
        return  ' bestX: %s' % self.best_X

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
    
    def run(self, logger=None, verbose=True):
        meta_f = self.meta_sampler.f
        inner_f = self.inner_sampler.f
        if logger is not None:
            logger.write_csv(['Ite'] + meta_f.info_header()+inner_f.info_header())
    
        ite = 0
        self.best_X = self.model.m

        while not meta_f.terminate_condition():
            ite += 1
            seed = None
            re_eval_count = 1
            optimized_bestx = np.zeros(self.d)
            self.best_eval = 0

            for _ in range(re_eval_count):
                # optimize best_X 
                inner_f.clear()
                opt = CMAES_elitist(
                    d = self.d,
                    sampler = self.inner_sampler,
                    m = self.best_X,
                    sigma = 1,
                    min_problem = inner_f.minimization_problem,
                    seed = seed,
                    minimal_eigenval=1e-30
                )
                results = opt.run(self.inner_sampler)
                optimized_bestx += np.array([results[3]])[0] / re_eval_count
                self.best_eval += np.array([results[1]])[0] / re_eval_count
                meta_f.eval_count += results[0]
                meta_f._update_best_eval(np.array([self.best_eval][0]), np.array([optimized_bestx]))
            
            self.best_eval, original_best_eval = meta_f.LST_transformation(optimized_bestx, self.model.sigma)
            
            init_mean, _ = self.meta_sampler(self.sampling_model())
            init_sigma = 1
            optimized_mean = np.zeros(self.d)
            evals = 0

            for _ in range(re_eval_count):
                # optimize mean
                inner_f.clear()
                opt = CMAES_elitist(
                    d = self.d,
                    sampler = self.inner_sampler,
                    m = init_mean[0],
                    sigma = init_sigma,
                    min_problem = inner_f.minimization_problem,
                    seed=seed,
                    minimal_eigenval=1e-30
                )
                results = opt.run(self.inner_sampler)
                # print(results)
                optimized_mean += np.array([results[3]])[0] / re_eval_count
                evals += np.array([results[1]]) / re_eval_count
                meta_f.eval_count += results[0]
                meta_f._update_best_eval(np.array([evals][0]), np.array([optimized_mean]))

            evals, original_eval = meta_f.LST_transformation(optimized_mean, self.model.sigma)

            self.best_X = optimized_bestx
            self.model.m = optimized_bestx

            # display and save log
            if verbose:
                print(str(ite) + meta_f.verbose_display() + self.verbose_display() + self.meta_sampler.verbose_display())

            if logger is not None:
                logger.write_csv(['Ite'] + meta_f.info_list()+inner_f.info_list())

            # parameter update
            self.update(np.array([optimized_mean]), np.array([evals]))

            if self.best_eval == evals:
                self.best_eval = original_eval
            else:
                self.best_eval = original_best_eval
            inner_f.clear()
        return [meta_f.eval_count, meta_f.best_eval, meta_f.is_success(), meta_f.bestX]