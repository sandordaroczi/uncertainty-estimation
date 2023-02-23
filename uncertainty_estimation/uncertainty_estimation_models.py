import copy
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

import lightgbm
import numpy as np
import pandas as pd
import pgbm as PGBM_base
import properscoring as ps
import scipy.stats as stats
import torch
import torch.distributions as distributions
import xgboost
from gluonts.model.rotbaum._model import LSF as LSF_base
from joblib import Parallel, delayed
from mapie.quantile_regression import MapieQuantileRegressor
from ngboost import NGBRegressor, distns
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_lightning import Trainer as Lightning_Trainer
from scipy.linalg import LinAlgError
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import FeatureUnion

from .constants import PredEnum, DistEnum, EPSILON, MAX_NLL_VALUE, nll_valid_distributions, \
    pgbm_distributions_for_optimization_log_scale, pgbm_dist_to_enum, ngboost_valid_distributions, \
    pgbm_distributions_for_optimization_id_scale, pgbm_valid_distributions, valid_prediction_types

target_transformer_map = {
    "id": (lambda x: x),
    "log": np.log,
    "log1p": np.log1p
}

inverse_target_transformer_map = {
    "id": (lambda x: x),
    "log": np.exp,
    "log1p": np.expm1
}

valid_target_transformers = list(target_transformer_map.values())


class Model(ABC):

    @abstractmethod
    def __init__(self, vectorizer, target_transformer: str = "log1p"):
        self.target_transformer_string = target_transformer
        try:
            self.target_transformer = target_transformer_map[target_transformer]
            self.inverse_target_transformer = inverse_target_transformer_map[target_transformer]
        except KeyError as e:
            raise KeyError(f"Target transformer should be one of {list(target_transformer_map.keys())}") from e

        self.vectorizer = vectorizer
        self.quantiles = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, target: str, verbose: bool, **kwargs):
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame, verbose: bool, **kwargs: object) -> Dict:
        pass

    @abstractmethod
    def metrics(self, y_test, predictions, prediction_types: Optional[List],
                verbose: bool):
        """

        Args:
            y_test: array-like of shape (n_samples, 1)
                array containing the test target values
            predictions: Dict
                a dictionary with values containing point and uncertainty predictions
            prediction_types: List
                a list of prediction types
            verbose: bool
                currently not used

        Returns:
            metrics values as a dict
        """
        if prediction_types is None:
            prediction_types = list(predictions.keys())
        prediction_types = list(set(prediction_types) & set(valid_prediction_types))

        metrics = {}
        if PredEnum.POINT_ESTIMATES in prediction_types:
            point_predictions = predictions[PredEnum.POINT_ESTIMATES]
            metrics["mse"] = self.mse(y_test, point_predictions)
            metrics["mae"] = self.mae(y_test, point_predictions)
            metrics["rmse"] = self.rmse(y_test, point_predictions)
            metrics["mape"] = self.mape(y_test, point_predictions)
            metrics["rmspe"] = self.rmspe(y_test, point_predictions)

        if PredEnum.QUANTILES in prediction_types:
            quantiles = predictions[PredEnum.QUANTILES]
            if quantiles.shape[1] != 2:
                raise ValueError(f"Exactly 2 quantiles have to be supplied"
                                 f"to calculate avg_interval_length (sharpness) and coverage")
            metrics["avg_interval_length"] = self.avg_interval_length(quantiles)
            metrics["sharpness"] = metrics["avg_interval_length"]
            metrics["coverage"] = self.coverage(y_test, quantiles)

        if PredEnum.SAMPLES in prediction_types:
            metrics["crps"] = self.crps(y_test, predictions[PredEnum.SAMPLES])
            try:
                metrics["nll_from_samples"] = self.neg_log_likelihood_with_kde(y_test, predictions[PredEnum.SAMPLES],
                                                                               parallel=True, verbose=verbose)
            except LinAlgError as e:
                print(f"Exception encountered while trying to calculate NLL from samples using KDE: {e}")
                print("Setting NLL value as infinity.")
                metrics["nll_from_samples"] = np.inf

        if PredEnum.DISTRIBUTION_PARAMS in prediction_types:
            pass
        if len(metrics.keys()) == 0:
            raise ValueError(f"Cannot compute metrics."
                             f"Please provide via the prediction_types parameter which metrics to compute."
                             f"Valid options: {valid_prediction_types}")

        return metrics

    @staticmethod
    def get_predictions_for_ci_quantiles(predictions, confidence_interval_quantiles, quantiles):
        if confidence_interval_quantiles is None:
            confidence_interval_quantiles = [min(quantiles), max(quantiles)]
        if not all(quantile in quantiles for quantile in confidence_interval_quantiles):
            raise ValueError('Please specify confidence_interval_quantiles which are computed by the model')

        if len(confidence_interval_quantiles) != 2:
            raise ValueError('Please specify exactly 2 quantiles for confidence_interval_quantiles')

        if confidence_interval_quantiles[0] > confidence_interval_quantiles[1]:
            raise ValueError('Lower quantile has to be a smaller than upper quantile')

        lower_limit = np.reshape(predictions[PredEnum.QUANTILES][confidence_interval_quantiles[0]],
                                 newshape=(-1, 1))
        upper_limit = np.reshape(predictions[PredEnum.QUANTILES][confidence_interval_quantiles[1]],
                                 newshape=(-1, 1))

        return np.concatenate((lower_limit, upper_limit), axis=1)

    @staticmethod
    def train_val_split(X, y, prob, group_after: Optional[str] = None):
        if group_after is None:
            bit = np.random.choice([0, 1], size=(X.shape[0],), p=[prob, 1 - prob]).astype(bool)
            X_train = X[bit]
            X_calib = X[np.logical_not(bit)]
            y_train = y[bit]
            y_calib = y[np.logical_not(bit)]
        else:
            groups = X[group_after].unique()
            bit = np.random.choice([0, 1], size=(groups.shape[0],), p=[prob, 1 - prob]).astype(bool)
            chosen_groups = groups[bit]
            X_train = X[X[group_after].isin(chosen_groups)]
            X_calib = X[~X[group_after].isin(chosen_groups)]
            y_train = y[X[group_after].isin(chosen_groups)]
            y_calib = y[~X[group_after].isin(chosen_groups)]

        return X_train, y_train, X_calib, y_calib

    @staticmethod
    def mse(y_test, predictions):
        """mean squared error

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,1)
                point estimates for the real values
        """
        return mean_squared_error(y_test, predictions)

    @staticmethod
    def mae(y_test, predictions):
        """mean absolute error

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,1)
                point estimates for the real values
        """
        return mean_absolute_error(y_test, predictions)

    @staticmethod
    def rmse(y_test, predictions):
        """residual mean squared error

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,1)
                point estimates for the real values
        """
        return np.sqrt(mean_squared_error(y_test, predictions))

    @staticmethod
    def mape(y_test, predictions):
        """mean absolute percentage error

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,1)
                point estimates for the real values
        """
        return mean_absolute_percentage_error(y_test, predictions)

    @staticmethod
    def rmspe(y_test, predictions):
        """residual mean squared percentage error

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,1)
                point estimates for the real values
        """
        return np.sqrt(np.mean(
            np.square((y_test - predictions) / (y_test + EPSILON))))  # Avoid numerical issues of division by zero

    @staticmethod
    def mase(y_test, predictions, y_train, n_timeseries, seasonal_periodicity=1):
        """mean absolut scaled error: measure of the accuracy of forecasts

        Args:
            y_test: array-like of shape (n_timeseries, forecast_horizon, 1)
                observed value
            predictions: array-like of shape (n_timeseries, forecast_horizon, 1)
            y_train: array-like of shape (n_timeseries, lookback+forecast_horizon, 1)
                target variables which have been used during model training
            n_timeseries: int describing the total number of time series
            seasonal_periodicity: Seasonal periodicity of training data.
        """
        mase = []
        for ts_index in range(n_timeseries):
            y_pred_naive = y_train[ts_index, :-seasonal_periodicity]
            mae_naive = mean_absolute_error(y_train[ts_index, seasonal_periodicity:], y_pred_naive)
            mae_pred = mean_absolute_error(y_test, predictions)
            single_mase = mae_pred / np.maximum(mae_naive, EPSILON)
            mase.append(single_mase)

        return np.mean(mase)

    @staticmethod
    def avg_interval_length(predictions):
        """ average interval length

        Args:
            predictions:
                lower and upperbounds of confidence interval for each datapoint in a np.array(N,2)
        """
        return np.mean(predictions[:, 1] - predictions[:, 0])

    @staticmethod
    def coverage(y_test, predictions):
        """coverage

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,2)
                point estimates for the real values
        """
        return np.mean((y_test >= predictions[:, 0]) & (y_test <= predictions[:, 1]))

    @staticmethod
    def crps(y_test, predictions):
        """continuous ranked probability score

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            predictions: array-like of shape (n_samples,2)
                point estimates for the real values
        """
        return ps.crps_ensemble(observations=y_test, forecasts=predictions).mean()

    @staticmethod
    def neg_log_likelihood(y_test, distribution: DistEnum, **kwargs):
        """Negative Log-Likelihood

        Args:
            y_test: array-like of shape (n_samples,1)
                observed value
            distribution: DistEnum
                assumed supported distribution
            **kwargs:
                parameters of distribution

        Returns:
            float: mean NLL value
        """
        # Convert to tensor if data is numpy array
        if distribution not in nll_valid_distributions:
            raise ValueError(f"Computation of NLL is not supported for {distribution} distribution."
                             f"Supported distributions: {nll_valid_distributions}.")
        if isinstance(y_test, np.ndarray):
            y_test = torch.from_numpy(y_test)

        try:
            if distribution == DistEnum.LOG_LOG_NORMAL:
                loc = kwargs.get("loc")
                scale = kwargs.get("scale")
                log_y = torch.log(y_test)
                log_log_y = torch.log(torch.log(y_test))
                sqrt_2pi = np.sqrt(np.pi * 2)
                nll_temp = torch.empty(size=(len(loc), 1))
                for i in range(len(loc)):
                    nll_temp[i] = - torch.log(1 / (sqrt_2pi * scale[i] * log_y[i] * y_test[i])
                                              * torch.exp(-0.5 * (log_log_y[i] - loc[i]) ** 2 / scale[i] ** 2))
                nll = torch.mean(nll_temp)
                return nll.item()

            # Compute NLL for considered distribution
            elif distribution == DistEnum.NORMAL:
                loc = kwargs.get("loc")
                scale = kwargs.get("scale")
                dist = {i: distributions.normal.Normal(loc[i], scale[i]) for i in range(len(loc))}
            elif distribution == DistEnum.LOG_NORMAL:
                loc = kwargs.get("loc")
                scale = kwargs.get("scale")
                dist = {i: distributions.log_normal.LogNormal(loc[i], scale[i]) for i in range(len(loc))}
            elif distribution == DistEnum.EXPONENTIAL:
                rate = kwargs.get("scale")
                dist = {i: distributions.exponential.Exponential(rate[i]) for i in range(len(rate))}
            elif distribution == DistEnum.STUDENT_T:
                df = kwargs.get("df")
                loc = kwargs.get("loc")
                scale = kwargs.get("scale")
                dist = {i: distributions.studentT.StudentT(df, loc[i], scale[i]) for i in range(len(loc))}
            elif distribution == DistEnum.LAPLACE:
                loc = kwargs.get("loc")
                scale = kwargs.get("scale")
                dist = {i: distributions.laplace.Laplace(loc[i], scale[i]) for i in range(len(loc))}
            elif distribution == DistEnum.GAMMA:
                shape = kwargs.get("shape")
                rate = kwargs.get("rate")
                dist = {i: distributions.gamma.Gamma(shape[i], rate[i]) for i in range(len(shape))}
            elif distribution == DistEnum.GUMBEL:
                loc = kwargs.get("loc")
                scale = kwargs.get("scale")
                dist = {i: distributions.gumbel.Gumbel(loc[i], scale[i]) for i in range(len(loc))}
            elif distribution == DistEnum.NEGATIVE_BINOMIAL:
                probs = kwargs.get("probs")
                counts = kwargs.get("counts")
                dist = {i: distributions.negative_binomial.NegativeBinomial(counts[i], probs[i]) for i in
                        range(len(probs))}
            else:
                print(f"Exact calculation of NLL is not (yet) supported for {distribution} distribution.")
                return np.inf

            nll_temp = torch.tensor([-dist[i].log_prob(torch.tensor(y_test[i])) for i in range(len(dist))])
            nll = torch.mean(nll_temp)
            return nll.item()

        except ValueError as e:
            print(f"ValueError encountered when calculating NLL: {e}")
            return np.inf

    @staticmethod
    def neg_log_likelihood_with_kde(y_test, distribution_samples,
                                    bw_method="scott", parallel=True, verbose=False):
        """Negative Log-Likelihood with KDE
            Calculates NLL using Kernel Density Estimation from a list of samples

                Args:
                    y_test: array-like of shape (n_datapoints,1)
                        observed values
                    distribution_samples: array-like of shape (n_datapoints, n_samples)
                        samples taken from predicted distribution for each test datapoint
                    bw_method: string or float
                        bandwidth estimation method for the underlying scipy.stats.gaussian_kde() function
                        possible values: "scott", "silverman", or a float number denoting the bandwidth value
                    parallel: bool
                        set to true to compute NLL with KDE using joblib.Parallel() or to false for using a for-loop
                    verbose: bool
                        to print runtime of algorithm

                Returns:
                    float: mean NLL value
                """
        start_time = time.perf_counter()
        if parallel:
            def kde_for_a_single_point(y_test_point, samples):
                kde_samples = np.array(samples)[:, np.newaxis]
                kde_instance = stats.gaussian_kde(kde_samples.T, bw_method=bw_method)
                kde_value = kde_instance(y_test_point)
                kde_value_clipped = np.clip(kde_value, a_min=np.exp(-MAX_NLL_VALUE), a_max=None)
                return -np.log(kde_value_clipped)

            nll_values = Parallel(n_jobs=-1)(
                delayed(kde_for_a_single_point)(y_test_point, samples) for y_test_point, samples in
                zip(y_test, distribution_samples)
            )

        else:
            nll_values = []
            for y_test_point, samples in zip(y_test, distribution_samples):
                samples_reshaped = np.array(samples)[:, np.newaxis]
                kde = stats.gaussian_kde(samples_reshaped.T, bw_method=bw_method)
                kde_evaluated = kde(y_test_point)
                kde_evaluated_clipped = np.clip(kde_evaluated, a_min=np.exp(-MAX_NLL_VALUE), a_max=None)
                nll = -np.log(kde_evaluated_clipped)
                nll_values.append(nll)

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for calculating NLL with KDE: {np.round(end_time - start_time, 2)} s")

        nll_values = np.array(nll_values)
        return nll_values.mean()


class LightGBM(Model):
    """
    Implementation of LightGBM for use and benchmarking for confidence intervals
    """

    def __init__(self, vectorizer: FeatureUnion, target_transformer: str = "log1p"):
        super().__init__(vectorizer, target_transformer)
        self.model = None
        self.best_iteration = None

    def fit(self, X: pd.DataFrame, target: str, params: Optional[Dict[str, Any]] = None,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            verbose: bool = False, **kwargs):
        """

        Args:
            X:  pd.DataFrame
                raw data
            target: str
                value to estimate, target value
            params: Optional[Dict[str, Any]]
                parameters for lightgbm.train()
            X_val: Optional[np.ndarray]
                Validation data
            y_val: Optional[np.ndarray]
                validation values
            verbose: bool
                to print training time
            **kwargs: dict
                additional arguments for lightgbm.train()

        Returns:
            trained lightgbm model
        """
        start_time = time.perf_counter()
        if params is None:
            params = {}

        y = self.target_transformer(X[target].values)

        if X_val is not None and y_val is not None:
            X_train_vectorized = self.vectorizer.fit_transform(X)
            X_val_vectorized = self.vectorizer.transform(X_val)
            y_train = y
            y_val = self.target_transformer(y_val)

            self.model = lightgbm.LGBMRegressor(**params)
            self.model.fit(X_train_vectorized, y_train, eval_set=[(X_val_vectorized, y_val)], **kwargs)
            self.best_iteration = self.model.best_iteration_
        else:
            X_vectorized = self.vectorizer.fit_transform(X)
            self.model = lightgbm.LGBMRegressor(**params)
            self.model.fit(X_vectorized, y, **kwargs)

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for fitting {self.__class__.__name__} model: {np.round(end_time - start_time, 2)} s")

        return self.model

    def predict(self, X_test: pd.DataFrame, verbose: bool = False, **kwargs) -> dict:
        """

        Args:
            X_test: pd.Dataframe of size (n_samples, n_attributes + 1)
                test data for prediction
            **kwargs: dict
                additional parameters for lightgbm.predict()
            verbose: bool
                to print elapsed time for prediction

        Returns: dict
            dictionary with point estimates

        """
        start_time = time.perf_counter()
        test_vectorized = self.vectorizer.transform(X_test)
        if self.best_iteration is None:
            y_pred = self.inverse_target_transformer(self.model.predict(test_vectorized, **kwargs))
        else:
            y_pred = self.inverse_target_transformer(self.model.predict(test_vectorized,
                                                                        num_iteration=self.best_iteration, **kwargs))

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for predicting with {self.__class__.__name__} model:"
                  f"{np.round(end_time - start_time, 2)} s")

        return {PredEnum.POINT_ESTIMATES: y_pred}

    def metrics(self, y_test: np.ndarray, predictions: dict, prediction_types: Optional[List] = None,
                verbose: bool = False) -> dict:
        """

            Args:
                y_test: array-like of shape (n_samples,1)
                    observed values
                predictions: dict
                    results from LightGBM.predict
                prediction_types: Optional[list]
                    irrelevant parameter as only point predictions can be provided
                verbose: bool
                    currently not used

            Returns: dict
                values of mean squared error, residual mean squarred error, mean absolute percentage error and residual
                 mean squared percentage error for the predicted values

            """

        return super().metrics(y_test, predictions, [PredEnum.POINT_ESTIMATES], verbose=verbose)


class LightGBMQuantileRegressor(Model):
    """
    Implementation of LightGBM for use and benchmarking for confidence intervals
    """

    def __init__(self, vectorizer: FeatureUnion, target_transformer: str = "log1p"):
        super().__init__(vectorizer, target_transformer)
        self.models = {}
        self.best_iterations = {}

    def fit(self, X: pd.DataFrame, target: str, params: Optional[Dict[str, Any]] = None,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            quantiles: Optional[List[float]] = None,
            verbose: bool = False, **kwargs):
        """

        Args:
            X:  pd.DataFrame
                raw data
            target: str
                value to estimate, target value
            params: Optional[Dict[str, Any]]
                parameters for lightgbm.train()
            X_val: Optional[np.ndarray]
                Validation data
            y_val: Optional[np.ndarray]
                validation values
            quantiles: Optional[List[float]]
                list of quantiles to fit the regressor on. Default values: [0.05, 0.5, 0.95]
            verbose: bool
                to print training time
            **kwargs: dict
                additional arguments for lightgbm.train()

        Returns:
            trained lightgbm model
        """
        start_time = time.perf_counter()
        if params is None:
            params = {}

        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]

        self.quantiles = quantiles
        params['objective'] = 'quantile'
        params['metric'] = 'quantile'

        y = self.target_transformer(X[target].values)

        if X_val is not None and y_val is not None:
            X_train_vectorized = self.vectorizer.fit_transform(X)
            X_val_vectorized = self.vectorizer.transform(X_val)
            y_train = y
            y_val = self.target_transformer(y_val)

            for quantile in self.quantiles:
                params['alpha'] = quantile
                self.models[quantile] = lightgbm.LGBMRegressor(**params)
                self.models[quantile].fit(X_train_vectorized, y_train, eval_set=[(X_val_vectorized, y_val)], **kwargs)
                self.best_iterations[quantile] = self.models[quantile].best_iteration_
        else:
            X_vectorized = self.vectorizer.fit_transform(X)
            for quantile in self.quantiles:
                params['alpha'] = quantile
                self.models[quantile] = lightgbm.LGBMRegressor(**params)
                self.models[quantile].fit(X_vectorized, y, **kwargs)

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for fitting {self.__class__.__name__} model: {np.round(end_time - start_time, 2)} s")

        return self.models

    def predict(self, X_test: pd.DataFrame, verbose: bool = False, **kwargs) -> dict:
        """

        Args:
            X_test: pd.Dataframe of size (n_samples, n_attributes + 1)
                test data for prediction
            verbose: bool
                to print elapsed time for prediction
            **kwargs: dict
                additional parameters for lightgbm.predict()

        Returns: dict
            dictionary with point estimates

        """
        start_time = time.perf_counter()
        test_vectorized = self.vectorizer.transform(X_test)
        quantile_pred = {}
        for quantile in self.quantiles:
            quantile_model = self.models[quantile]
            best_iter = self.best_iterations[quantile]
            if best_iter is None:
                single_quantile_pred = quantile_model.predict(test_vectorized, **kwargs)
            else:
                single_quantile_pred = quantile_model.predict(test_vectorized, num_iteration=best_iter, **kwargs)
            quantile_pred[quantile] = self.inverse_target_transformer(single_quantile_pred)

        predictions = {PredEnum.POINT_ESTIMATES: quantile_pred[0.5],
                       PredEnum.QUANTILES: quantile_pred}

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for predicting with {self.__class__.__name__} model:"
                  f"{np.round(end_time - start_time, 2)} s")

        return predictions

    def metrics(self, y_test: np.ndarray, predictions: dict, prediction_types: Optional[List] = None,
                verbose: bool = False, confidence_interval_quantiles: Optional[List[float]] = None) -> dict:
        """

            Args:
                y_test: array-like of shape (n_samples,1)
                    observed values
                predictions: dict
                    results from LightGBM.predict
                prediction_types:
                    irrelevant parameter as only point predictions can be provided
                verbose: bool
                    currently not used


            Returns: dict
                values of mean squared error, residual mean squarred error, mean absolute percentage error and residual
                 mean squared percentage error for the predicted values

            """
        if not prediction_types:
            prediction_types = [PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES]

        prediction_types = list(set(prediction_types) & {PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES})
        predictions_reshaped = copy.deepcopy(predictions)
        if PredEnum.QUANTILES in prediction_types:
            predictions_reshaped[PredEnum.QUANTILES] = self.get_predictions_for_ci_quantiles(
                predictions,
                confidence_interval_quantiles,
                self.quantiles
            )

        return super().metrics(y_test, predictions_reshaped, prediction_types, verbose=verbose)


class XGBoost(Model):
    """
    Implementation of XGBoost for use and benchmarking with different vectorizers for confidence intervals
    """

    def __init__(self, vectorizer: FeatureUnion, params: Optional[Dict[str, Any]] = None,
                 target_transformer: str = "log1p"):
        super().__init__(vectorizer, target_transformer)
        if params is None:
            params = {}
        self.model = xgboost.XGBRegressor(**params)
        self.best_iteration = None

    def fit(self, X: pd.DataFrame, target: str,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            verbose: bool = False, **kwargs):
        """

        Args:
            X:  pd.DataFrame
                raw data
            target: str
                value to estimate, target value
            X_val: Optional[np.ndarray]
                Validation data
            y_val: Optional[np.ndarray]
                validation values
            verbose: bool
                to print training time
            **kwargs: dict
                additional arguments for xgboost.train()

        Returns:
            trained xgboost model
        """
        start_time = time.perf_counter()
        y = self.target_transformer(X[target].values)

        if X_val is not None and y_val is not None:
            X_train_vectorized = self.vectorizer.fit_transform(X)
            X_val_vectorized = self.vectorizer.transform(X_val)
            y_train = y
            y_val = self.target_transformer(y_val)

            self.model.fit(X_train_vectorized, y_train, eval_set=[X_val_vectorized, y_val], **kwargs)
            self.best_iteration = self.model.best_iteration_
        else:
            X_vectorized = self.vectorizer.fit_transform(X)
            self.model.fit(X_vectorized, y, **kwargs)

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for fitting {self.__class__.__name__} model: {np.round(end_time - start_time, 2)} s")

        return self.model

    def predict(self, X_test: pd.DataFrame, verbose: bool = False, **kwargs) -> dict:
        """

        Args:
            X_test: pd.Dataframe of size (n_samples, n_attributes + 1)
                test data for prediction
            verbose: bool
                to print elapsed time for prediction
            **kwargs: dict
                additional parameters for lightgbm.predict()

        Returns: dict
            dictionary with point estimates

        """
        start_time = time.perf_counter()
        test_vectorized = self.vectorizer.transform(X_test)

        if self.best_iteration is None:
            y_pred = self.inverse_target_transformer(self.model.predict(test_vectorized, **kwargs))
        else:
            y_pred = self.inverse_target_transformer(
                self.model.predict(test_vectorized, iteration_range=(0, self.best_iteration + 1), **kwargs))

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for predicting with {self.__class__.__name__} model:"
                  f"{np.round(end_time - start_time, 2)} s")

        return {PredEnum.POINT_ESTIMATES: y_pred}

    def metrics(self, y_test: np.ndarray, predictions: dict, prediction_types: Optional[List] = None,
                verbose: bool = False) -> dict:
        """

        Args:
            y_test: array-like of shape (n_samples,1)
                observed values
            predictions: dict
                results from XGBoost.predict
            prediction_types:
                irrelevant parameter as only point estimates can be provided
            verbose: bool
                currently not used

        Returns: dict
            values of mean squared error, residual mean squared error, mean average percentage error and residual mean
            squared percentage error for the predicted values

        """
        return super().metrics(y_test, predictions, [PredEnum.POINT_ESTIMATES], verbose=verbose)


class NGBoost(Model):
    """
    Implementation of NGBoost for use and benchmarking with different vectorizers for confidence intervals
    """

    def __init__(self, vectorizer: FeatureUnion, target_transformer: str = "log1p",
                 distribution: DistEnum = DistEnum.NORMAL,
                 **kwargs):
        super().__init__(vectorizer, target_transformer)
        self.model = NGBRegressor(Dist=NGBoost.conversion_from_dist_enum_to_distns(distribution), **kwargs)
        self.distribution = distribution
        self.best_iteration = None

    def fit(self, X: pd.DataFrame, target: str, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, verbose: bool = False, **kwargs):
        """

        Args:
            X:  pd.DataFrame
                raw data
            target: str
                value to estimate, target value
            X_val: Optional[np.ndarray]
                Validation data
            y_val: Optional[np.ndarray]
                validation values
            verbose: bool
                to print training time
            **kwargs: dict
                additional arguments for xgboost.train()

        Returns:
            trained ngboost model
        """
        start_time = time.perf_counter()
        if self.distribution == DistEnum.LOG_NORMAL and np.any(X[target].values <= 1):
            raise ValueError("Log_Normal Distribution does not work as target contains values <=1")

        y = self.target_transformer(X[target].values)

        if X_val is not None and y_val is not None:
            X_train_vectorized = self.vectorizer.fit_transform(X)
            X_val_vectorized = self.vectorizer.transform(X_val)
            y_train = y
            y_val = self.target_transformer(y_val)

            self.model.fit(X_train_vectorized, y_train, X_val=X_val_vectorized, Y_val=y_val, **kwargs)
            self.best_iteration = self.model.best_val_loss_itr
        else:
            X_vectorized = self.vectorizer.fit_transform(X)
            self.model.fit(X_vectorized, y, **kwargs)

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for fitting {self.__class__.__name__} model: {np.round(end_time - start_time, 2)} s")

        return self.model

    def predict(self, X_test, quantiles: Optional[List[float]] = None,
                prediction_types: Optional[List] = None, sample_size: int = 200,
                quantile_sample_size: int = 1000, verbose: bool = False, **kwargs):
        """

        Args:
            X_test: pd.Dataframe of size (n_samples, n_attributes + 1)
                test data for prediction
            quantiles: Optional[List]
                list of quantiles to estimate
            prediction_types: list
                predictions to return. Possible predictions are point estimates (PredEnum.POINT_ESTIMATES), quantiles
                (PredEnum.QUANTILES), samples (PredEnum.SAMPLES) and distribution (PredEnum.DISTRIBUTION_PARAMS). If
                None is set point estimates, quantiles and samples are set
            sample_size: int
                returned sample size when chosen samples
            quantile_sample_size: int
                number of samples the quantile computation is based of
            verbose: bool
                to print elapsed time for prediction
            **kwargs:
                additional parameters for pgbm.predict or pgbm.pred_dist

        Returns:
            dictionary with predictions as specified in prediction_types is returned
        """

        start_time = time.perf_counter()
        if prediction_types is None:
            prediction_types = [PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES, PredEnum.SAMPLES,
                                PredEnum.DISTRIBUTION_PARAMS]
        self.quantiles = quantiles

        test_vectorized = self.vectorizer.transform(X_test)
        predictions = {}

        if PredEnum.POINT_ESTIMATES in prediction_types:
            if self.best_iteration is None:
                predict = self.model.predict(test_vectorized, **kwargs)
            else:
                predict = self.model.predict(test_vectorized, max_iter=self.best_iteration)
            predictions[PredEnum.POINT_ESTIMATES] = self.inverse_target_transformer(predict)

        if PredEnum.QUANTILES in prediction_types:
            if quantiles is None:
                raise ValueError('Please specify quantiles you want to predict')
            if self.distribution == DistEnum.NORMAL:
                if self.best_iteration is None:
                    samples_quant = self.model.pred_dist(test_vectorized, **kwargs).sample(1000)
                else:
                    samples_quant = self.model.pred_dist(test_vectorized,
                                                         max_iter=self.best_iteration).sample(1000)
            elif self.distribution == DistEnum.LOG_NORMAL:
                if self.best_iteration is None:
                    samples_quant_temp = self.model.pred_dist(test_vectorized, **kwargs)
                else:
                    samples_quant_temp = self.model.pred_dist(test_vectorized,
                                                              max_iter=self.best_iteration)
                samples_quant = np.zeros((1000, X_test.shape[0]))
                for i in range(X_test.shape[0]):
                    samples_quant_params = stats.lognorm.rvs(s=samples_quant_temp.params['s'][i],
                                                             scale=samples_quant_temp.params['scale'][i], size=1000)
                    samples_quant[:, i] = samples_quant_params
            else:
                if self.best_iteration is None:
                    samples_quant_temp = self.model.pred_dist(test_vectorized, **kwargs)
                else:
                    samples_quant_temp = self.model.pred_dist(test_vectorized,
                                                              max_iter=self.best_iteration)
                samples_quant = np.zeros((1000, X_test.shape[0]))
                for i in range(X_test.shape[0]):
                    samples_quant_params = stats.expon.rvs(scale=samples_quant_temp.params['scale'][i], size=1000)
                    samples_quant[:, i] = samples_quant_params

            quantile_fc = {}
            for quantile in quantiles:
                single_quantile_fc = np.empty((X_test.shape[0], 1))
                for i in range(X_test.shape[0]):
                    single_quantile_fc[i, :] = np.quantile(samples_quant[:, i], quantile)
                quantile_fc[quantile] = self.inverse_target_transformer(single_quantile_fc)
            predictions[PredEnum.QUANTILES] = quantile_fc

        if PredEnum.SAMPLES in prediction_types:
            if self.distribution == DistEnum.NORMAL:
                if self.best_iteration is None:
                    samples_temp = self.model.pred_dist(test_vectorized, **kwargs).sample(sample_size)
                else:
                    samples_temp = self.model.pred_dist(test_vectorized, max_iter=self.best_iteration).sample(
                        sample_size)
                predictions[PredEnum.SAMPLES] = self.inverse_target_transformer(samples_temp)
            elif self.distribution == DistEnum.LOG_NORMAL:
                if self.best_iteration is None:
                    samples_temp = self.model.pred_dist(test_vectorized, **kwargs)
                else:
                    samples_temp = self.model.pred_dist(test_vectorized, max_iter=self.best_iteration)
                filler = np.zeros((sample_size, X_test.shape[0]))
                for i in range(X_test.shape[0]):
                    samples_rvs = stats.lognorm.rvs(s=samples_temp.params['s'][i],
                                                    scale=samples_temp.params['scale'][i], size=sample_size)
                    filler[:, i] = samples_rvs
                predictions[PredEnum.SAMPLES] = self.inverse_target_transformer(filler)
            else:
                if self.best_iteration is None:
                    samples_temp = self.model.pred_dist(test_vectorized, **kwargs)
                else:
                    samples_temp = self.model.pred_dist(test_vectorized, max_iter=self.best_iteration)
                filler = np.zeros((sample_size, X_test.shape[0]))
                for i in range(X_test.shape[0]):
                    samples_rvs = stats.expon.rvs(scale=samples_temp.params['scale'][i], size=sample_size)
                    filler[:, i] = samples_rvs
                predictions[PredEnum.SAMPLES] = self.inverse_target_transformer(filler)

        if PredEnum.DISTRIBUTION_PARAMS in prediction_types:
            if self.best_iteration is None:
                predictions[PredEnum.DISTRIBUTION_PARAMS] = self.model.pred_dist(test_vectorized, **kwargs).params
            else:
                predictions[PredEnum.DISTRIBUTION_PARAMS] = self.model.pred_dist(test_vectorized,
                                                                                 max_iter=self.best_iteration
                                                                                 ).params

        predictions[PredEnum.SAMPLES] = np.transpose(predictions[PredEnum.SAMPLES])

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for predicting with {self.__class__.__name__} model:"
                  f"{np.round(end_time - start_time, 2)} s")

        return predictions

    def metrics(self, y_test, predictions, prediction_types: Optional[List] = None, verbose: bool = False,
                confidence_interval_quantiles: Optional[List[float]] = None):
        """

            Args:
                y_test: array-like of shape (n_samples,1)
                    observed values
                predictions: dict
                    results from PGBM.predict
                prediction_types:
                    prediction_types should be a subset of the keys of predictions. If prediction_types=None then the
                    keys of predictions are taken
                verbose: bool
                    currently not used
                confidence_interval_quantiles: Optional[List[float]]
                    quantiles for which confidence intervals shall be computed. Must be contained in computed quantiles.

            Returns: dict
                values of mean squared error, residual mean squarred error, mean absolute percentage error and residual
                mean squared percentage error if prediction_types contains PredEnum.POINT_ESTIMATE, average interval
                length and coverage if prediction_types contains PredEnuM.QUANTILES, continuous ranked probability
                score if prediction_types contains PredEnum.SAMPLES and negative log likelihood if prediction_type
                contains PredEnum.DISTRIBUTION_PARAMS for the predicted values.

            """
        if prediction_types is None:
            prediction_types = list(predictions.keys())

        prediction_types_super = list(
            set(prediction_types) & set(valid_prediction_types).difference({PredEnum.DISTRIBUTION_PARAMS}))

        predictions_reshaped = copy.deepcopy(predictions)
        if PredEnum.QUANTILES in prediction_types_super:
            predictions_reshaped[PredEnum.QUANTILES] = self.get_predictions_for_ci_quantiles(
                predictions,
                confidence_interval_quantiles,
                self.quantiles
            )

        metrics = super().metrics(y_test, predictions_reshaped, prediction_types_super, verbose=verbose)

        if PredEnum.DISTRIBUTION_PARAMS in prediction_types:
            if self.distribution not in ngboost_valid_distributions:
                raise ValueError("Distribution must be one of %r." % ngboost_valid_distributions)

            distribution_params = predictions[PredEnum.DISTRIBUTION_PARAMS]
            if self.distribution == DistEnum.NORMAL:
                loc = distribution_params['loc']
                scale = distribution_params['scale']
                if self.target_transformer_string == 'id':
                    metrics["nll"] = super().neg_log_likelihood(y_test, self.distribution, loc=loc, scale=scale)
                elif self.target_transformer_string == 'log1p':
                    metrics["nll"] = super().neg_log_likelihood(y_test + 1, DistEnum.LOG_NORMAL, loc=loc, scale=scale)
                else:
                    metrics["nll"] = super().neg_log_likelihood(y_test, DistEnum.LOG_NORMAL, loc=loc, scale=scale)

            elif self.distribution == DistEnum.LOG_NORMAL:
                s = distribution_params['s']
                scale = distribution_params['scale']
                loc = np.log(scale)
                scale = s
                if self.target_transformer_string == 'id':
                    metrics["nll"] = super().neg_log_likelihood(y_test, self.distribution, loc=loc, scale=scale)
                elif self.target_transformer_string == 'log1p':
                    metrics["nll"] = super().neg_log_likelihood(y_test + 1, DistEnum.LOG_LOG_NORMAL, loc=loc,
                                                                scale=scale)
                else:
                    metrics["nll"] = super().neg_log_likelihood(y_test, DistEnum.LOG_LOG_NORMAL, loc=loc, scale=scale)

            elif self.distribution == DistEnum.EXPONENTIAL:
                scale = distribution_params['scale']
                if self.target_transformer_string == 'id':
                    metrics["nll"] = super().neg_log_likelihood(y_test, self.distribution, scale=scale)
                else:
                    print(
                        f"Exact calculation of NLL not (yet) supported for the exp of {self.distribution} distribution."
                    )
                    metrics["nll"] = np.inf

        return metrics

    @staticmethod
    def conversion_from_dist_enum_to_distns(distribution: DistEnum):
        if distribution == DistEnum.NORMAL:
            return distns.normal.Normal
        elif distribution == DistEnum.LOG_NORMAL:
            return distns.lognormal.LogNormal
        elif distribution == DistEnum.EXPONENTIAL:
            return distns.exponential.Exponential
        else:
            raise ValueError(f"{distribution} distribution not supported by NGBoost")


class PGBM(Model):
    """
    Implementation of PGBM for use and benchmarking with different vectorizers for confidence intervals
    """

    def __init__(self, vectorizer: FeatureUnion, target_transformer: str = "log1p"):
        super().__init__(vectorizer, target_transformer)
        self.model = PGBM_base.PGBM()
        self.best_iteration = None

    def fit(self, X: pd.DataFrame, target: str,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            params: Optional[Dict] = None, apply_optimize_distribution: bool = False,
            restrict_range_of_distributions: bool = True,
            verbose: bool = False, **kwargs):
        """

        Args:
            X: pd.Dataframe
                raw/prepared data
            target: str
                target value, value to estimate
            X_val: None or arraylike
                If provided together with y_val this split is used when performing validation
            y_val: None or arraylike
                If provided together with X_val this split is used when performing validation
            params: dict
                Parameters for PGBM
            apply_optimize_distribution: bool
                set to true if you want to find the possible best distribution for the validation dataset
            restrict_range_of_distributions: bool
                set to true to restrict range of distributions to those for which exact NLL values
                can be calculated
            verbose: bool
                to print training time
            **kwargs:
                additional arguments for PGBM.fit()

        Returns:
            trained pgbm model
        """
        start_time = time.perf_counter()
        if params is None:
            params = {}
        y = self.target_transformer(X[target].values)

        if X_val is not None and y_val is not None:
            X_train_vectorized = self.vectorizer.fit_transform(X)
            X_val_vectorized = self.vectorizer.transform(X_val)
            y_train = y
            y_val = self.target_transformer(y_val)

            train_vectorized_y_train = (X_train_vectorized, y_train)
            val_vectorized_y_val = (X_val_vectorized, y_val)
            self.model.train(
                train_set=train_vectorized_y_train,
                objective=PGBM.mseloss_objective,
                metric=PGBM.rmseloss_metric,
                valid_set=val_vectorized_y_val,
                params=params,
                **kwargs
            )
            self.best_iteration = self.model.best_iteration

            if apply_optimize_distribution:
                if not restrict_range_of_distributions:
                    best_dist, best_tree_corr = self.model.optimize_distribution(
                        X_val_vectorized,
                        y_val
                    )
                elif self.target_transformer_string == 'id':
                    best_dist, best_tree_corr = self.model.optimize_distribution(
                        X_val_vectorized,
                        y_val,
                        distributions=pgbm_distributions_for_optimization_id_scale
                    )
                else:
                    best_dist, best_tree_corr = self.model.optimize_distribution(
                        X_val_vectorized,
                        y_val,
                        distributions=pgbm_distributions_for_optimization_log_scale
                    )
                self.model.distribution = best_dist
                self.model.tree_correlation = best_tree_corr

        else:
            X_vectorized = self.vectorizer.fit_transform(X)
            train_vectorized_y_train = (X_vectorized, y)
            self.model.train(
                train_vectorized_y_train,
                objective=PGBM.mseloss_objective,
                metric=PGBM.rmseloss_metric,
                params=params,
                **kwargs
            )

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for fitting {self.__class__.__name__} model: {np.round(end_time - start_time, 2)} s")

        return self.model

    def predict(self, X_test, quantiles: Optional[List[float]] = None,
                prediction_types: Optional[List] = None,
                distribution: Optional[str] = None, sample_size: int = 200, verbose: bool = False, **kwargs):
        """

        Args:
            X_test: pd.Dataframe of size (n_samples, n_attributes + 1)
                test data for prediction
            quantiles: list
                list of quantiles to estimate
            prediction_types: list
                predictions to return. Possible predictions are point estimates (PredEnum.POINT_ESTIMATES), quantiles
                (PredEnum.QUANTILES), samples (PredEnum.SAMPLES) and distribution (PredEnum.DISTRIBUTION_PARAMS). If
                None is set point estimates, quantiles and samples are set
            distribution: str
                distribution of model when predictions other than 'pointestimates' are wanted
            sample_size: int
                returned sample size when chosen samples
            verbose: bool
                to print elapsed time for prediction
            **kwargs:
                additional parameters for pgbm.predict() or pgbm.pred_dist()

        Returns:
            dictionary with predictions as specified in prediction_types is returned
        """

        start_time = time.perf_counter()
        if prediction_types is None:
            prediction_types = [PredEnum.POINT_ESTIMATES, PredEnum.SAMPLES, PredEnum.QUANTILES,
                                PredEnum.DISTRIBUTION_PARAMS]
        self.quantiles = quantiles

        test_vectorized = self.vectorizer.transform(X_test)
        predictions = {}

        if PredEnum.POINT_ESTIMATES in prediction_types:
            pred_point_estimates = self.model.predict(test_vectorized).cpu().detach().numpy()
            predictions[PredEnum.POINT_ESTIMATES] = self.inverse_target_transformer(pred_point_estimates)

        if PredEnum.QUANTILES in prediction_types:
            if quantiles is None:
                raise ValueError('Please specify quantiles you want to predict')
            if distribution is not None:
                if pgbm_dist_to_enum[distribution] not in pgbm_valid_distributions:
                    raise ValueError("results: status must be one of %r." % pgbm_valid_distributions)
                self.model.distribution = distribution
            samples_quant = self.model.predict_dist(test_vectorized, n_forecasts=sample_size, **kwargs) \
                .cpu().detach().numpy()
            quantile_fc = {}
            for quantile in quantiles:
                single_quantile_fc = np.empty((X_test.shape[0], 1))
                for i in range(X_test.shape[0]):
                    single_quantile_fc[i, :] = np.quantile(samples_quant[:, i], quantile)
                quantile_fc[quantile] = self.inverse_target_transformer(single_quantile_fc)
            predictions[PredEnum.QUANTILES] = quantile_fc

        if PredEnum.SAMPLES in prediction_types:
            if distribution is not None:
                if pgbm_dist_to_enum[distribution] not in pgbm_valid_distributions:
                    raise ValueError(f"{distribution} distribution is not supported by PGBM.")
                self.model.distribution = distribution
            pred_samples = self.model.predict_dist(test_vectorized, n_forecasts=sample_size, **kwargs) \
                .cpu().detach().numpy()
            predictions[PredEnum.SAMPLES] = self.inverse_target_transformer(np.transpose(pred_samples))

        if PredEnum.DISTRIBUTION_PARAMS in prediction_types:
            if distribution is not None:
                if pgbm_dist_to_enum[distribution] not in pgbm_valid_distributions:
                    raise ValueError("results: status must be one of %r." % pgbm_valid_distributions)
                self.model.distribution = distribution
            sample_and_params = self.model.predict_dist(test_vectorized, n_forecasts=0, output_sample_statistics=True,
                                                        **kwargs)
            predictions[PredEnum.DISTRIBUTION_PARAMS] = (sample_and_params[1].cpu().detach().numpy(),
                                                         sample_and_params[2].cpu().detach().numpy())

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for predicting with {self.__class__.__name__} model:"
                  f"{np.round(end_time - start_time, 2)} s")

        return predictions

    def metrics(self, y_test, predictions, prediction_types: Optional[List] = None, verbose: bool = False,
                confidence_interval_quantiles: Optional[List[float]] = None):
        """

            Args:
                y_test: array-like of shape (n_samples,1)
                    observed values
                predictions: dict
                    results from PGBM.predict
                prediction_types:
                    prediction_types should be a subset of the keys of predictions. If prediction_types=None then the
                    keys of predictions are taken
                verbose: bool
                    currently not used
                confidence_interval_quantiles: Optional[List[float]]
                    quantiles for which confidence intervals shall be computed. Must be contained in computed quantiles.

            Returns: dict
                values of mean squared error, residual mean squarred error, mean absolute percentage error and residual
                mean squared percentage error if prediction_types contains PredEnum.POINT_ESTIMATE, average interval
                length and coverage if prediction_types contains PredEnuM.QUANTILES, continuous ranked probability
                score if prediction_types contains PredEnum.SAMPLES and negative log likelihood if prediction_type
                contains PredEnum.DISTRIBUTION_PARAMS for the predicted values.

            """
        if prediction_types is None:
            prediction_types = list(predictions.keys())
        prediction_types_super = list(
            set(prediction_types) & set(valid_prediction_types).difference({PredEnum.DISTRIBUTION_PARAMS}))

        predictions_reshaped = copy.deepcopy(predictions)
        if PredEnum.QUANTILES in prediction_types_super:
            predictions_reshaped[PredEnum.QUANTILES] = self.get_predictions_for_ci_quantiles(
                predictions,
                confidence_interval_quantiles,
                self.quantiles
            )

        metrics = super().metrics(y_test, predictions_reshaped, prediction_types_super, verbose=verbose)

        if PredEnum.DISTRIBUTION_PARAMS in prediction_types:
            distribution_params = predictions[PredEnum.DISTRIBUTION_PARAMS]
            mu = distribution_params[0]
            variance = distribution_params[1]
            metrics["nll"] = self.calculate_neg_log_likelihood(y_test, pgbm_dist_to_enum[self.model.distribution], mu,
                                                               variance)

        return metrics

    def calculate_neg_log_likelihood(self, target, distribution, mu, variance):
        if distribution not in pgbm_valid_distributions:
            raise ValueError(f"Distribution must be one of {pgbm_valid_distributions}.")

        if self.target_transformer_string != 'id' and distribution not in [DistEnum.NORMAL, DistEnum.LOG_NORMAL]:
            print(f"Exact calculation of NLL is not (yet) supported for the exp of a {distribution} distribution.")
            return np.inf

        if distribution == DistEnum.NORMAL:
            loc = mu
            scale = np.nan_to_num(np.max([np.sqrt(variance), np.ones(len(variance)) * 1e-9], axis=0), nan=1e-9)
            if self.target_transformer_string == 'id':
                return super().neg_log_likelihood(target, distribution, loc=loc, scale=scale)
            elif self.target_transformer_string == 'log1p':
                return super().neg_log_likelihood(target + 1, DistEnum.LOG_NORMAL, loc=loc, scale=scale)

            return super().neg_log_likelihood(target, DistEnum.LOG_NORMAL, loc=loc, scale=scale)

        elif distribution == DistEnum.LOG_NORMAL:
            mu_adj = np.clip(mu, a_min=1e-9, a_max=None)
            variance = np.max([np.nan_to_num(variance, nan=1e-9), np.ones(len(variance)) * 1e-9], axis=0)
            loc = np.log(mu_adj ** 2 / np.sqrt(variance + mu_adj ** 2))
            scale = np.clip(np.log(1 + variance / mu_adj ** 2), a_min=1e-9, a_max=None)
            if self.target_transformer_string == 'id':
                return super().neg_log_likelihood(target, distribution, loc=loc, scale=scale)
            elif self.target_transformer_string == 'log1p':
                return super().neg_log_likelihood(target + 1, DistEnum.LOG_LOG_NORMAL, loc=loc, scale=scale)

            return super().neg_log_likelihood(target, DistEnum.LOG_LOG_NORMAL, loc=loc, scale=scale)

        elif distribution == DistEnum.STUDENT_T:
            df = 3
            loc = mu
            factor = df / (df - 2)
            scale = np.nan_to_num(np.sqrt((variance / factor)), nan=1e-9)
            return super().neg_log_likelihood(target, distribution, df=df, loc=loc, scale=scale)

        elif distribution == DistEnum.LAPLACE:
            loc = mu
            scale = np.nan_to_num(np.sqrt((0.5 * variance)), nan=1e-9)
            return super().neg_log_likelihood(target, distribution, loc=loc, scale=scale)

        elif distribution == DistEnum.GAMMA:
            variance = np.nan_to_num(variance, nan=1e-9)
            mu_adj = np.nan_to_num(mu, nan=1e-9)
            rate = (mu_adj.clip(1e-9)) / (variance.clip(1e-9))
            shape = mu_adj.clip(1e-9) * rate
            return super().neg_log_likelihood(target, distribution, rate=rate, shape=shape)

        elif distribution == DistEnum.GUMBEL:
            variance = np.nan_to_num(variance, nan=1e-9)
            scale = (np.sqrt(6 * variance / np.pi ** 2)).clip(1e-9)
            loc = mu - scale * np.euler_gamma
            return super().neg_log_likelihood(target, distribution, loc=loc, scale=scale)

        elif distribution == DistEnum.NEGATIVE_BINOMIAL:
            loc = np.clip(mu, a_min=1e-9, a_max=None)
            eps = 1e-9
            variance = np.nan_to_num(variance, nan=1e-9)
            scale = np.maximum(loc + eps, variance).clip(1e-9)
            probs = (1 - (loc / scale)).clip(0, 0.99999)
            counts = (-loc ** 2 / (loc - scale)).clip(eps)
            return super().neg_log_likelihood(target, distribution, probs=probs, counts=counts)

        elif distribution in [DistEnum.LOGISTIC, DistEnum.WEIBULL, DistEnum.POISSON]:
            print(f"Exact calculation of NLL is not (yet) supported for {distribution} distribution.")

        return np.inf

    @staticmethod
    def mseloss_objective(yhat, y, sample_weight):
        """
        computation of gradient and hessian of mse_loss
        """
        gradient = (yhat - y)
        hessian = torch.ones_like(yhat)
        return gradient, hessian

    @staticmethod
    def rmseloss_metric(yhat, y, sample_weight):
        """
        computation of gradient and hessian of rmse_loss
        """
        loss = torch.sqrt(torch.mean(torch.square(yhat - y)))
        return loss


class LSF(Model):
    """
    Implementation of LSF for use and benchmarking with for confidence intervals
    """

    def __init__(self, vectorizer: FeatureUnion, target_transformer: str = "log1p",
                 base_model=lightgbm.LGBMRegressor(),
                 model_trained: bool = False, min_bin_size: Optional[int] = None, **kwargs):
        """

        Args:
            vectorizer:
            target_transformer:
            base_model:
            model_trained:
            min_bin_size:
                recommendation: log^2(# samples)
            **kwargs:
        """
        super().__init__(vectorizer, target_transformer)
        self.model = LSF_base(model=base_model, **kwargs)
        self.model_trained = model_trained
        self.min_bin_size = min_bin_size

    def fit(self, X: pd.DataFrame, target: str, verbose: bool = False, **kwargs):
        """

        Args:
            X:  pd.DataFrame
                raw data
            target: str
                value to estimate, target value
            verbose: bool
                to print training time
            **kwargs: dict
                additional arguments for lightgbm.train()

        Returns:
            trained lsf model
        """
        start_time = time.perf_counter()
        y = self.target_transformer(X[target].values)
        X_vectorized = self.vectorizer.fit_transform(X)

        if self.min_bin_size is None:
            self.min_bin_size = np.log(len(X.index)) ** 2

        self.model.fit(X_vectorized, y, model_is_already_trained=self.model_trained, min_bin_size=self.min_bin_size,
                       **kwargs)

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for fitting {self.__class__.__name__} model: {np.round(end_time - start_time, 2)} s")

        return self.model

    def predict(self, X_test, quantiles: Optional[List[float]] = None,
                prediction_types: Optional[List] = None, verbose: bool = False, **kwargs) -> dict:
        """

        Args:
            X_test: pd.Dataframe of size (n_samples, n_attributes + 1)
                test data for prediction
            quantiles: Optional[List[float]]
                a list of quantiles to estimate
            verbose: bool
                to print elapsed time for prediction
            prediction_types: Optional[List]
                predictions to return. Possible predictions are point estimates (PredEnum.POINT_ESTIMATES), quantiles
                (PredEnum.QUANTILES) and samples (PredEnum.SAMPLES). If None is is set point estimates, quantiles and
                samples are set


        Returns:
            dictionary with predictions as specified in prediction_types is returned

        """

        start_time = time.perf_counter()
        if not prediction_types:
            prediction_types = [PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES, PredEnum.SAMPLES]
        self.quantiles = quantiles

        test_vectorized = self.vectorizer.transform(X_test)

        predictions = {}
        if PredEnum.POINT_ESTIMATES in prediction_types:
            predictions[PredEnum.POINT_ESTIMATES] = self.inverse_target_transformer(
                self.model.model.predict(test_vectorized))

        if PredEnum.QUANTILES in prediction_types:
            if quantiles is None:
                raise ValueError('Please specify quantiles you want to predict')
            quantile_predictions = {}
            for quantile in quantiles:
                quantile_predictions[quantile] = np.transpose(self.inverse_target_transformer(
                    self.model.predict(test_vectorized, quantile)))
            predictions[PredEnum.QUANTILES] = quantile_predictions

        if PredEnum.SAMPLES in prediction_types:
            samples = self.model.estimate_dist(test_vectorized)
            samples = np.array([np.array(samples_for_one_input) for samples_for_one_input in samples])
            for i in range(len(samples)):
                samples[i] = self.inverse_target_transformer(samples[i])
            predictions[PredEnum.SAMPLES] = samples

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for predicting with {self.__class__.__name__} model:"
                  f"{np.round(end_time - start_time, 2)} s")

        return predictions

    def metrics(self, y_test, predictions, prediction_types: Optional[List] = None, verbose: bool = False,
                confidence_interval_quantiles: Optional[List[float]] = None) -> dict:
        """

            Args:
                y_test: array-like of shape (n_samples,1)
                    observed values
                predictions: dict
                    results from LSF.predict
                prediction_types:
                    prediction_types should be a subset of the keys of predictions. If prediction_types=None then the
                    keys of predictions are taken
                verbose: bool
                    currently not used
                confidence_interval_quantiles: Optional[List[float]]
                    quantiles for which confidence intervals shall be computed. Must be contained in computed quantiles.

            Returns: dict
                values of mean squared error, residual mean squarred error, mean absolute percentage error and residual
                mean squared percentage error if prediction_types contains PredEnum.POINT_ESTIMATE, average interval
                length and coverage if prediction_types contains PredEnuM.QUANTILES and continuous ranked probability
                score if prediction_types contains PredEnum.SAMPLES for the predicted values.

            """
        if not prediction_types:
            prediction_types = list(predictions.keys())

        prediction_types_super = list(set(prediction_types) & {PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES})

        predictions_reshaped = copy.deepcopy(predictions)
        if PredEnum.QUANTILES in prediction_types_super:
            predictions_reshaped[PredEnum.QUANTILES] = self.get_predictions_for_ci_quantiles(predictions,
                                                                                             confidence_interval_quantiles,
                                                                                             self.quantiles)

        metrics = super().metrics(y_test, predictions_reshaped, prediction_types_super, verbose=verbose)

        if PredEnum.SAMPLES in prediction_types:
            y_pred_dist_bins = predictions[PredEnum.SAMPLES]
            # Loop required as number of generated samples deviates for every observation
            crps = np.empty(len(y_test))
            for i in range(len(y_test)):
                crps[i] = super().crps(y_test[i], y_pred_dist_bins[i])
            metrics["crps"] = crps.mean()
            metrics["nll_from_samples"] = self.neg_log_likelihood_with_kde(y_test, y_pred_dist_bins,
                                                                           parallel=True, verbose=verbose)

        return metrics


class CQR(Model):
    """
    Implementation of CQR for use and benchmarking with different vectorizers for confidence intervals
    """

    def __init__(self, vectorizer: FeatureUnion, target_transformer: str = "log1p"):
        super().__init__(vectorizer, target_transformer)
        self.model = None
        self.alpha = None

    def fit(self, X: pd.DataFrame, target: str,
            X_calib: Optional[pd.DataFrame] = None, y_calib: Optional[pd.DataFrame] = None,
            list_of_estimators: Optional[List] = None, train_cal_split: float = 0.2, group_after: Optional[str] = None,
            params: Optional[Dict] = None, alpha: float = 0.2, verbose: bool = False,
            **kwargs):
        """

        Args:
            X: pd.Dataframe
                raw data
            target: str
                value to estimate, target value
            X_calib: pd.Dataframe
                calibration dataset to fit the conformal model on
            y_calib: pd.Dataframe
                target values in the calibration dataset
            list_of_estimators: List
                list of size 3 containing point estimators for quantiles alpha/2, 1-alpha/2, 0.5
                Default estimators are LGBM Quantile Regressors
            train_cal_split: value between 0 and 1 including
                proportion of the raw data to use for training
            group_after: Optional[str]
                if random split, option for splitting after feature
            params: Dict
                hyperparameters for the underlying LGBM models (if list_of_estimators is not specified)
            alpha: value between 0 and 1 including
                alpha value corresponding to an (1-alpha) * 100% confidence interval
            verbose: bool
                to print training time

            **kwargs: dict
                additional arguments for MapieQuantileRegressor.fit()

        Returns:
            trained MapieQuantileRegressor
        """

        start_time = time.perf_counter()
        if params is None:
            params = {}
            params['objective'] = 'quantile'

        self.quantiles = [alpha / 2, 1 - (alpha / 2)]
        if list_of_estimators is None:
            list_of_estimators = [lightgbm.LGBMRegressor(alpha=alpha_value, **params)
                                  for alpha_value in self.quantiles + [0.5]]

        y = self.target_transformer(X[target].values)
        if X_calib is None or y_calib is None:
            # Split dataset into training and calibration sets
            X_train, y_train, X_calib, y_calib = self.train_val_split(X, y, train_cal_split, group_after)
        else:
            X_train = X
            y_train = y

        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_calib_vectorized = self.vectorizer.transform(X_calib)

        for estimator in list_of_estimators:
            estimator.fit(X_train_vectorized, y_train)

        # Calibrate uncertainties on calibration set
        self.model = MapieQuantileRegressor(list_of_estimators, cv="prefit")
        self.model.fit(X_calib_vectorized, y_calib)

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for fitting {self.__class__.__name__} model: {np.round(end_time - start_time, 2)} s")

        return self.model

    def predict(self, X_test, verbose: bool = False, **kwargs):
        """computation of confidence intervals

        args:
            X_test: pd.Dataframe of size (n_samples, n_attributes + 1)
                test data for prediction
            verbose: bool
                to print elapsed time for prediction
            kwargs:
                additional arguments for MapieQuantileRegressor.predict()

        :return: dict
            contains confidence intervals as ndarray of size (n_samples, 2)
        """

        start_time = time.perf_counter()
        test_vectorized = self.vectorizer.transform(X_test)
        y_pred, y_quantiles_pred = self.model.predict(test_vectorized)
        y_pred = self.inverse_target_transformer(y_pred)
        y_quantiles_pred = self.inverse_target_transformer(y_quantiles_pred)

        predictions = {PredEnum.POINT_ESTIMATES: y_pred,
                       PredEnum.QUANTILES: {self.quantiles[0]: y_quantiles_pred[:, 0, :],
                                            0.5: y_pred,
                                            self.quantiles[1]: y_quantiles_pred[:, 1, :]}}

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for predicting with {self.__class__.__name__} model:"
                  f"{np.round(end_time - start_time, 2)} s")

        return predictions

    def metrics(self, y_test, predictions, prediction_types: Optional[List] = None, verbose: bool = False):
        """

            Args:
                y_test: array-like of shape (n_samples,1)
                    observed values
                predictions: dict
                    results from XGBoost.predict
                prediction_types:
                    irrelevant parameter as only quantiles can be provided
                verbose: bool
                    currently not used

            Returns: dict
                values of average interval length and coverage for the predicted values

            """
        if not prediction_types:
            prediction_types = [PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES]
        prediction_types = list(set(prediction_types) & {PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES})

        predictions_reshaped = copy.deepcopy(predictions)
        if PredEnum.QUANTILES in prediction_types:
            predictions_reshaped[PredEnum.QUANTILES] = self.get_predictions_for_ci_quantiles(predictions,
                                                                                             self.quantiles,
                                                                                             self.quantiles)

        return super().metrics(y_test, predictions_reshaped, prediction_types, verbose=verbose)


class TFTPytorchFC(Model):
    """
    Implementation of TFT for use and benchmarking for confidence intervals
    """

    def __init__(self, vectorizer=None, target_transformer: str = "log1p", lookback: int = 1,
                 forecast_horizon: int = 1, time_idx: str = None, group_ids: Optional[List[str]] = None,
                 static_categoricals: Optional[List[str]] = None, static_reals: Optional[List[str]] = None,
                 time_varying_known_categoricals: Optional[List[str]] = None,
                 time_varying_known_reals: Optional[List[str]] = None,
                 time_varying_unknown_categoricals: Optional[List[str]] = None,
                 time_varying_unknown_reals: Optional[List[str]] = None,
                 quantiles: Optional[List[float]] = None):

        """
        Args:
            vectorizer: not used in this model
            target_transformer: specify if model should be trained on log scale --> metrics are evaluated on real scale
            lookback: int: Number of time units that condition the predictions
            forecast_horizon: int: Length of the prediction
            time_idx: str: name of the column with time index (running int number for every timeseries from 1...T)
            group_ids: List[str]: Name of the columns that, when grouped by, gives the different time series.
            static_categoricals: List[str]: List of column names that contain static categorical features.
            static_reals: List[str]: List of column names that contain static real features.
            time_varying_known_categoricals: List[str]: List of column names that contain known dynamic real features.
                (future covariates)
            time_varying_known_reals: List[str]: List of column names that contain known dynamic categorical features.
                (future covariates)
            time_varying_unknown_categoricals: List[str]: List of column names that contain
                                                          unknown dynamic real features.
                (past covariates)
            time_varying_unknown_reals: List[str]: List of column names that contain
                                                   unknown dynamic categorical features.
                (past covariates)
            quantiles: Optional[List[float]]: List of quantiles you want to perform Quantile Regression on.
                                              Must contain 0.5 quantiles
            and be at least of length 3
        """

        if group_ids is None:
            raise ValueError(
                'group_ids must be specified, otherwise TFT for multiple time series does not make sense')

        if time_idx is None:
            raise ValueError('Please provide a time_idx. If there is no time_idx in your dataset use method'
                             ' TFTPytorchFC.add_time_idx_to_df, which adds a column called "time_index_tft"')

        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

        if static_categoricals is None:
            static_categoricals = []
        if static_reals is None:
            static_reals = []
        if time_varying_known_categoricals is None:
            time_varying_known_categoricals = []
        if time_varying_known_reals is None:
            time_varying_known_reals = []
        if time_varying_unknown_categoricals is None:
            time_varying_unknown_categoricals = []
        if time_varying_unknown_reals is None:
            time_varying_unknown_reals = []

        if 0.5 not in quantiles:
            raise ValueError('Median quantile is required to get meaningful point predictions')

        if len(quantiles) < 3:
            raise ValueError('Specify at least one median and two upper and lower bound quantiles')

        super().__init__(vectorizer, target_transformer)
        self.model = None

        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.time_idx = time_idx
        self.group_ids = group_ids
        self.static_categoricals = static_categoricals
        self.static_reals = static_reals
        self.time_varying_known_categoricals = time_varying_known_categoricals
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_unknown_categoricals = time_varying_unknown_categoricals
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.quantiles = quantiles

        # Initialize values to None as they result out of fit but will be needed for fit and predict
        self.target_normalizer = None
        self.categorical_encoders = None
        self.scalers = None
        self.target = None
        self.params_dataset_creation = None

    def fit(self, X: pd.DataFrame, target: str, lightning_trainer: Lightning_Trainer = Lightning_Trainer(),
            params_dataset_creation: Optional[Dict] = None, params_tft: Optional[Dict] = None,
            params_dataloader: Optional[Dict] = None, verbose: bool = False):

        """
        fit method trains TFT model with possibility for early stopping (when callback early stopping is added)
                train data: consists of all data of dataframe X from least recent data up until the beginning of the
                last forecast_horizon time steps
                validation data: consists of most recent lookback+forecast_horizon timesteps

        Args:
            X: pd.Dataframe
                dataframe with features --> Given up until start of holdout set
            target: str
                value to estimate, target value
            lightning_trainer: dict
                additional arguments for pytorch lightning method Trainer() (e.g. Early Stopping callbacks, logger,
                learning rate, ...)
            params_dataset_creation: dict
                additional arguments for method TimeSeriesDataSet() in PyTorchFC
            params_tft: dict
                additional arguments for TFTModel.fit()
            params_dataloader: dict
                additional arguments for PyTorchFC method TimeSeriesDataSet.to_dataloader()
            verbose: bool
                to print training time

        Returns: self.model which is the best model according to validation loss
        """
        if params_dataset_creation is None:
            params_dataset_creation = {}
        if params_tft is None:
            params_tft = {}
        if params_dataloader is None:
            params_dataloader = {}

        start_time = time.perf_counter()
        self.target = target
        X_copy = X.copy()
        X_copy[self.target] = self.target_transformer(X_copy[self.target])
        most_recent_index = X_copy[self.time_idx].max()
        end_train_index = most_recent_index - self.forecast_horizon

        training = TimeSeriesDataSet(
            X_copy[X_copy[self.time_idx] <= end_train_index], time_idx=self.time_idx, target=self.target,
            group_ids=self.group_ids,
            max_encoder_length=self.lookback, max_prediction_length=self.forecast_horizon,
            static_categoricals=self.static_categoricals, static_reals=self.static_reals,
            time_varying_known_categoricals=self.time_varying_known_categoricals,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_categoricals=self.time_varying_unknown_categoricals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            add_relative_time_idx=True,  # otherwise we could not train without time_varying_known_reals
            **params_dataset_creation)

        self.params_dataset_creation = params_dataset_creation

        validation = TimeSeriesDataSet.from_dataset(training, X_copy, min_prediction_idx=training.index.time.max() + 1,
                                                    stop_randomization=True)

        train_dataloader = training.to_dataloader(train=True, **params_dataloader)
        val_dataloader = validation.to_dataloader(train=False, **params_dataloader)

        self.model = TemporalFusionTransformer.from_dataset(training, loss=QuantileLoss(self.quantiles),
                                                            output_size=len(self.quantiles), **params_tft)

        lightning_trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        best_model_path = lightning_trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        self.model = best_tft

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for fitting {self.__class__.__name__} model: {np.round(end_time - start_time, 2)} s")

        return self.model

    def predict(self, X_test, prediction_types: Optional[List] = None,
                verbose: bool = False, **kwargs):

        """
        predict method computes confidence intervals and points predictions with TFTPytorchFC model
        prediction is done for last forecast_horizon timesteps of input X_test (until fct = Forecast Creation Time)
        Args:
            X_test: pd.Dataframe
                dataframe with features
            prediction_types: list
                specifies which prediction types to return. Possible values: POINT_ESTIMATES, QUANTILES
            verbose: bool
                to print elapsed time for prediction
            **kwargs: additional arguments for model.predict()

        Returns:dict
            under key PredEnum.POINT_ESTIMATES is np.array of shape (#timeseries, #forecast_horizon, 1)
            under key PredEnum.QUANTILES is dictionary, which contains under every quantile key predictions of shape
            (#timeseries, #forecast_horizon)
        """

        start_time = time.perf_counter()
        if prediction_types is None:
            prediction_types = [PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES]

        X_test_copy = X_test.copy()
        X_test_copy[self.target] = self.target_transformer(X_test_copy[self.target])
        most_recent_index = X_test_copy[self.time_idx].max()
        end_train_index = most_recent_index - self.forecast_horizon

        training = TimeSeriesDataSet(
            X_test_copy[X_test_copy[self.time_idx] <= end_train_index], time_idx=self.time_idx, target=self.target,
            group_ids=self.group_ids, max_encoder_length=self.lookback, max_prediction_length=self.forecast_horizon,
            static_categoricals=self.static_categoricals, static_reals=self.static_reals,
            time_varying_known_categoricals=self.time_varying_known_categoricals,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_categoricals=self.time_varying_unknown_categoricals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            add_relative_time_idx=True,  # otherwise we could not train without time_varying_known_reals
            **self.params_dataset_creation)

        validation = TimeSeriesDataSet.from_dataset(training, X_test_copy,
                                                    min_prediction_idx=training.index.time.max() + 1,
                                                    stop_randomization=True)

        quantile_predictions = self.model.predict(
            validation.filter(lambda x: (x.time_idx_first_prediction == end_train_index + 1)), mode="quantiles")

        predictions = {}
        if PredEnum.POINT_ESTIMATES in prediction_types:
            point_pred = quantile_predictions[:, :, self.quantiles.index(0.5)]
            point_pred_np = point_pred.cpu().detach().numpy()
            predictions[PredEnum.POINT_ESTIMATES] = self.inverse_target_transformer(point_pred_np)

        if PredEnum.QUANTILES in prediction_types:
            quantile_fc = {}
            for i in range(len(self.quantiles)):
                single_quantile_fc = quantile_predictions[:, :, i]
                single_quantile_fc_np = single_quantile_fc.cpu().detach().numpy()
                quantile_fc[self.quantiles[i]] = self.inverse_target_transformer(single_quantile_fc_np)
            predictions[PredEnum.QUANTILES] = quantile_fc

        end_time = time.perf_counter()
        if verbose:
            print(f"Elapsed time for predicting with {self.__class__.__name__} model:"
                  f"{np.round(end_time - start_time, 2)} s")

        return predictions

    def metrics(self, y_test, predictions, prediction_types: Optional[List] = None, verbose: bool = False,
                confidence_interval_quantiles: Optional[List[float]] = None):

        """
                metrics method computes all possible metrics(defined in Abstract Model class)
                given the predictions obtained in predict method
                Args:
                    y_test: np.array
                        contains ground truth of shape (#timeseries, #forecast_horizon, 1)
                    predictions: dict
                        every key contains predictions
                        POINT_ESIMATES: shape (#timeseries, #forecast_horizon, 1)
                        QUANTILES: dictionary where each key is a quantile of shape (#timeseries, #forecast_horizon, 1)
                    prediction_types: list
                        specify for which prediction types we want to calculate metrics
                    verbose: bool
                        currently not used
                    confidence_interval_quantiles: List[float]
                        quantiles for which confidence intervals shall be computed.
                        Must be contained in computed quantiles.

                Returns: dict
                        contains metrics
                """

        if prediction_types is None:
            prediction_types = list(predictions.keys())
        prediction_types_super = list(set(prediction_types) & {PredEnum.POINT_ESTIMATES, PredEnum.QUANTILES})

        predictions_reshaped = copy.deepcopy(predictions)
        if PredEnum.POINT_ESTIMATES in prediction_types_super:
            predictions_reshaped[PredEnum.POINT_ESTIMATES] = np.reshape(predictions[PredEnum.POINT_ESTIMATES],
                                                                        newshape=(-1, 1))

        if PredEnum.QUANTILES in prediction_types_super:
            predictions_reshaped[PredEnum.QUANTILES] = self.get_predictions_for_ci_quantiles(predictions,
                                                                                             confidence_interval_quantiles,
                                                                                             self.quantiles)

        y_test_reshaped = np.reshape(y_test, newshape=(-1, 1))

        return super().metrics(y_test_reshaped, predictions_reshaped, prediction_types_super, verbose=verbose)

    @staticmethod
    def add_time_idx_to_df(X: pd.DataFrame, group_ids: List[str]):
        """
        helper method to obtain a time index column called "time_index_tft"
        Args:
            X: pd.Dataframe
                dataframe with features
            group_ids: str
                Name of the column that, when grouped by, gives the different time series.

        Returns: X_with_time_index: pd.Dataframe
                dataframe X with additional column named "time_index_tft"
        """
        list_of_dataframes = []
        for _, slice_object in X.groupby(by=group_ids, observed=True):
            slice_object = slice_object.assign(time_index_tft=range(len(slice_object)))
            list_of_dataframes.append(slice_object)
        X_with_time_index = pd.concat(list_of_dataframes)
        print('Time index called "time_index_tft" added to provided dataframe')
        return X_with_time_index

    @staticmethod
    def obtain_y_test_out_of_X_test(X_test, forecast_horizon, time_idx, target, group_ids):
        """
        Helper method to obtain y_test for metrics method directly out of X_test. Input has to be the same dataframe
        as for predict method
        Args:
            X_test: pd.Dataframe
                dataframe with features
            forecast_horizon: int
                Length of the prediction
            time_idx: str
                column with time information in form of running index (no data column)
            target: str
                Name of the target column
            group_ids: str
                Name of the column that, when grouped by, gives the different time series.

        Returns:
            y_test: np.array
                array with ground truth for forecast horizon for every time series
                shape (number of timeseries, forecast_horizon)
        """
        y_test = []
        for _, sliced in X_test.groupby(group_ids, observed=True):
            split_index = X_test[time_idx].max() - forecast_horizon
            single_y = sliced[target][sliced[time_idx] > split_index].to_numpy()
            y_test.append(single_y)

        return np.array(y_test)
