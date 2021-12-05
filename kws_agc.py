import numpy

import attr
import numpy as np
import math

__all__ = ["KwsAgc", "KwsAgcParams"]

"""
Python implementation of google paper:
"AUTOMATIC GAIN CONTROL AND MULTI-STYLE TRAINING FOR ROBUST
SMALL-FOOTPRINT KEYWORD SPOTTING WITH DEEP NEURAL NETWORKS"
https://static.googleusercontent.com/media/research.google.com/ru//pubs/archive/43289.pdf

"""


@attr.s
class KwsAgcParams:
    delta = attr.ib(type=numpy.float32, default=16)  # scale parameter
    theta = attr.ib(type=numpy.float32, default=0.5)  # voice gain level
    theta_s = attr.ib(type=numpy.float32, default=0.1)  # noise gain level
    sigma_s = attr.ib(type=numpy.float32, default=0.003)  # start variance for speech
    mu_s = attr.ib(type=numpy.float32, default=0.0025)  # start expectation for speech
    sigma_b = attr.ib(type=numpy.float32, default=0.003)  # start variance for noise
    mu_b = attr.ib(type=numpy.float32, default=0.00)  # start mu for noise
    win_size = attr.ib(type=int, default=100)  # size of filter window
    k_mu = attr.ib(type=numpy.float32, default=0.4)  # forgetting coef for expectation
    k_sigma = attr.ib(type=numpy.float32, default=0.33)  # forgetting coef for expectation


class KwsAgc:

    def __init__(self, params: KwsAgcParams = None):
        self.params: KwsAgcParams = params if params else KwsAgcParams()
        self.int16 = np.iinfo(np.int16)

    def __call__(self, data: np.ndarray, rate: int = 16000) -> np.ndarray:
        """
        Implements automatic gain control
        :param data: - float data normalized to [-1:1] range
        :param rate: - sample rate of data? default value is 16000
        :return:
        """

        result = data.copy()
        is_int16 = isinstance(result[0], np.int16)
        if is_int16:
            # convert to float
            result = (result / self.int16.max).astype('float32')

        params: KwsAgcParams = self.params
        win_size: int = int(rate / 1000 * self.params.win_size)
        sigma_s: numpy.float32 = params.sigma_s
        mu_s: numpy.float32 = params.mu_s
        sigma_b: params.sigma_s = params.sigma_b
        mu_b: params.sigma_s = params.mu_b
        for r in range(win_size, int(result.shape[0]), win_size):
            # for r in range(win_size, win_size + 2, win_size):
            s, e = r - win_size, r
            w = result[s:e]
            l = numpy.float32(np.abs(w).max())

            mu_s_n = numpy.float32(params.k_mu * l + (1 - params.k_mu) * mu_s)
            mu_b_n = numpy.float32(params.k_mu * l + (1 - params.k_mu) * mu_b)

            sigma_s_n = numpy.float32(params.k_sigma * (l - mu_s_n) * (l - mu_s_n) + \
                              (1 - params.k_sigma) * sigma_s * sigma_s)
            sigma_b_n = numpy.float32(params.k_sigma * (l - mu_b_n) * (l - mu_b_n) + \
                              (1 - params.k_sigma) * sigma_b * sigma_b)
            z_b = numpy.float32((l - mu_b_n) / math.sqrt(sigma_b_n))
            z_s = numpy.float32((l - mu_s_n) / math.sqrt(sigma_s_n))

            speech = True if z_s * z_s < z_b * z_b else False
            tau = numpy.float32(0.5 * l)
            if speech:
                mu_s = mu_s_n
                if sigma_s_n < tau * tau:
                    sigma_s_n = numpy.float32(sigma_s_n + (sigma_s_n + sigma_b_n) / (2 * params.delta))
                sigma_s = math.sqrt(sigma_s_n)
                gain = numpy.float32(params.theta / (mu_s + sigma_s) if (mu_s - mu_b) > \
                                                          (sigma_s + sigma_b) else \
                    params.theta_s / min((mu_s + sigma_s),
                                         (mu_b + sigma_b)))
            else:
                mu_b = mu_b_n
                if sigma_b_n < tau * tau:
                    sigma_b_n = numpy.float32(sigma_b_n + (sigma_b_n + sigma_s_n) / (2 * self.params.delta))
                sigma_b = math.sqrt(sigma_b_n)
                gain = numpy.float32(1)
            result[s:e] *= gain
            # k = result[s:e]
            # k[result[s:e] < -1] = -1
            # k[result[s:e] > 1] = 1
        return (result * self.int16.max).astype(np.int16) if is_int16 else result
