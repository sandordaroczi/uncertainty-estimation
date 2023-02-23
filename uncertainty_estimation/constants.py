from enum import Enum

import numpy as np

EPSILON = np.finfo(np.float64).eps
MAX_NLL_VALUE = 500


class PredEnum(Enum):
    POINT_ESTIMATES = 'point_estimates'
    SAMPLES = 'samples'
    QUANTILES = 'quantiles'
    DISTRIBUTION_PARAMS = 'distribution_params'


valid_prediction_types = [PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES, PredEnum.SAMPLES, PredEnum.DISTRIBUTION_PARAMS]


class DistEnum(Enum):
    NORMAL = 'normal'
    STUDENT_T = 'student_t'
    LAPLACE = 'laplace'
    LOGISTIC = 'logistic'
    LOG_NORMAL = 'lognormal'
    LOG_LOG_NORMAL = 'log_log_normal'
    GAMMA = 'gamma'
    GUMBEL = 'gumbel'
    WEIBULL = 'weibull'
    POISSON = 'poisson'
    EXPONENTIAL = 'exponential'
    NEGATIVE_BINOMIAL = 'negative_binomial'


nll_valid_distributions = [DistEnum.NORMAL, DistEnum.STUDENT_T, DistEnum.LAPLACE, DistEnum.LOG_NORMAL, DistEnum.GAMMA,
                           DistEnum.GUMBEL, DistEnum.EXPONENTIAL, DistEnum.LOGISTIC, DistEnum.NEGATIVE_BINOMIAL,
                           DistEnum.LOG_LOG_NORMAL, DistEnum.WEIBULL]

# Distributions which support exact calculation of NLL
pgbm_distributions_for_optimization_id_scale = ['normal', 'studentt', 'laplace', 'lognormal', 'gamma', 'gumbel',
                                                'negativebinomial']

# Distributions which support exact calculation of NLL for their exponentiated distribution
# The exponentiated distribution for Normal is LogNormal, for LogNormal it is LogLogNormal
pgbm_distributions_for_optimization_log_scale = ['normal', 'lognormal']

pgbm_dist_to_enum = {
    'normal': DistEnum.NORMAL,
    'studentt': DistEnum.STUDENT_T,
    'laplace': DistEnum.LAPLACE,
    'logistic': DistEnum.LOGISTIC,
    'lognormal': DistEnum.LOG_NORMAL,
    'gamma': DistEnum.GAMMA,
    'gumbel': DistEnum.GUMBEL,
    'weibull': DistEnum.WEIBULL,
    'negativebinomial': DistEnum.NEGATIVE_BINOMIAL,
    'poisson': DistEnum.POISSON
}

pgbm_valid_distributions = list(pgbm_dist_to_enum.values())
ngboost_valid_distributions = [DistEnum.NORMAL, DistEnum.EXPONENTIAL, DistEnum.LOG_NORMAL]
