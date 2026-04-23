#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
import sys
import numpy as np
import pandas as pd
import scipy.linalg

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

class Gaussian(Model):
    """sigmaを内蔵した正規分布モデル"""
    def __init__(self, d, m=None, C=None, sigma=1.0, minimal_eigenval=1e-30, normalize='None',seed = None):
        self.d = d
        self.m = m if m is not None else np.zeros(self.d)
        self.sigma = sigma # ★sigmaを属性として保持
        self.C = C if C is not None else np.identity(self.d)
        self.min_eigenval = minimal_eigenval

        self.rng = np.random.default_rng(seed)
        self.prev_sample = None  # 候補解を保存
        self.lam = None  # サンプルサイズを保存
    
    def sampling(self, lam):
        """
        ★sigmaを考慮したサンプリングを実行
        平均m、共分散行列 (sigma^2 * C) の正規分布からサンプリングする
        """
        # self.sqrtCはCの平方根行列
        # return self.sigma * (np.random.randn(lam, self.d) @ self.sqrtC.T) + self.m
        self.lam = lam  # サンプルサイズを保存
        samples = self.sigma * (self.rng.standard_normal((lam, self.d)) @ self.sqrtC.T) + self.m
        self.prev_sample = samples  # サンプリング結果を保存
        return samples
    
    def encoding(self, lam, X):
        """
        連続値問題なので、エンコードは不要。
        受け取った値をそのまま返す（恒等変換）。
        """
        return X

    def _get_C(self):
        return self.__C
    
    def _set_C(self, C):
        self.__C = (C + C.T) / 2
        self.__eigen_decomposition()
    
    C = property(_get_C, _set_C)

    def __eigen_decomposition(self):
        try:
            self.eigvals, self.eigvectors = np.linalg.eigh(self.C)
            if np.min(self.eigvals) > 0:
                self.sqrtC = self.eigvectors @ np.diag(np.sqrt(self.eigvals)) @ self.eigvectors.T
            else:
                self.sqrtC = np.diag(np.sqrt(np.maximum(np.diag(self.C), 1e-20)))
        except np.linalg.LinAlgError:
            self.sqrtC = np.identity(self.d)

    def terminate_condition(self):
        return (self.sigma**2 * np.min(self.eigvals)) < self.min_eigenval

    def verbose_display(self):
        return f' MinEigVal: {self.sigma**2 * np.min(self.eigvals):.2e}'
    
    def log(self):
        eigvec_flat = self.eigvectors.reshape(-1)
        log_data = (
            ['%e' % i for i in self.m]
            + ['%e' % i for i in self.eigvals]
            + ['%e' % v for v in eigvec_flat]
            + ['%e' % self.sigma]
        )
        
        # 候補解をログに追加
        if self.prev_sample is not None:
            for i, sample in enumerate(self.prev_sample):
                for j, val in enumerate(sample):
                    log_data.append('%e' % val)
        
        return log_data
    
    def log_header(self):
        # mean, eigenvalues, eigenvectors (flattened row-major), log determinant
        eigvec_headers = [f'evec{i}_dim{j}' for i in range(self.d) for j in range(self.d)]
        headers = (
            ['m%d' % i for i in range(self.d)]
            + ['eigval%d' % i for i in range(self.d)]
            + eigvec_headers
            + ['sigma']
        )
        
        # 候補解のヘッダーを追加（lamが保存されていれば）
        if self.lam is not None:
            for i in range(self.lam):
                for j in range(self.d):
                    headers.append(f'x{i}_dim{j}')
        
        return headers
    


class GaussianSigmaACA(Gaussian):
    def __init__(self, d, z_space, m=None, C=None, sigma=1., minimal_eigenval=1e-30, normalize='None'):
        super().__init__(d, m=m, C=C, minimal_eigenval=minimal_eigenval, normalize=normalize)
        self.sigma = sigma
        # discrete variables
        self.zd = len(z_space)
        # parameter for Affine Map (std)
        self.A = np.full(d, 1.)
        # boder
        lim = (z_space[:,1:] + z_space[:,:-1])/2
        # nan to maxima
        df_a = pd.DataFrame(z_space.T)
        df_li = pd.DataFrame(lim.T)
        self.z_space = df_a.fillna(df_a.max()).values.T
        self.z_lim = df_li.fillna(df_li.max()).values.T
        self.z_lim_low = np.concatenate([self.z_lim.min(axis=1).reshape([self.zd,1]), self.z_lim], 1)
        self.z_lim_up = np.concatenate([self.z_lim, self.z_lim.max(axis=1).reshape([self.zd,1])], 1)
        m_z = m[self.d - self.zd:].reshape(([self.zd, 1]))
        # m_z_lim_low ->|  mean vector    |<- m_z_lim_up
        self.m_z_lim_low = (self.z_lim_low * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1))==m_z, 1, 0)).sum(axis=1)
        self.m_z_lim_up = (self.z_lim_up * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1))==m_z, 1, 0)).sum(axis=1)

        self.prev_sample = None

    def sampling(self, lam):
        samples = self.sigma * np.random.randn(lam, self.d).dot(self.sqrtC.T) + self.m
        self.prev_sample = samples
        return samples
    
    def encoding(self, lam, X):
        """
        X.shape = (lam, N_continuous + N_integer) 
        """
        # Affine Mapped Samples
        X = (X - self.m) * self.A + self.m
        num_cont = self.d - self.zd # = N_continuous
        # get variables for discrete
        X_z = X[:,num_cont:]
        # reshape variables for discrete
        X_z_c = X_z.reshape(([lam, self.zd, 1]))
        # encoding
        X_z_enc = (self.z_space * np.where(np.sort(np.concatenate([np.tile(self.z_lim, (lam,1,1)), X_z_c], 2))==X_z_c, 1, 0)).sum(axis=2)
        return np.hstack((X[:,:num_cont], X_z_enc))

    def loglikelihood(self, X):
        Z = np.dot((X - self.m), self.invSqrtC.T) / self.sigma
        return - 0.5 * (self.d * np.log(2. * np.pi) + self.logDetC) - np.log(self.sigma) - 0.5 * np.linalg.norm(Z, axis=1)**2

    def terminate_condition(self):
        return np.logical_or(np.logical_or((self.sigma**2) * np.min(self.eigvals) < self.min_eigenval, (0 < (np.isinf(self.m).sum() + np.isinf(self.C).sum() + np.isinf(self.sigma).sum()))), (0 < (np.isnan(self.m).sum() + np.isnan(self.C).sum() + np.isnan(self.sigma).sum())))

    def verbose_display(self):
        return ' MinEigVal: %e' % ((self.sigma**2) * (np.min(self.eigvals))) \
            + ' Cond: %e' % ((np.max(self.eigvals)) / (np.min(self.eigvals))) \
            + ' sigma: %e' % (self.sigma**2)

    def log_header(self):
        return super().log_header()

    def log(self):
        return super().log()


