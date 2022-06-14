import collections
import inspect
import math
from typing import Type

import pandas as pd
import torch
from river import base, anomaly, stats
from torch import nn

from DeepRiver.utils import get_loss_fn, get_optimizer_fn, dict2tensor, SequentialLSTM


class DeepEstimator(base.Estimator):
    """
    A PyTorch to River base class that aims to provide basic supervised functionalities.
    ----------
    build_fn
    loss_fn
    optimizer_fn
    learning_rate
    device
    net_params
    """
    def __init__(
        self,
        build_fn,
        loss_fn: str,
        optimizer_fn: Type[torch.optim.Optimizer],
        learning_rate=1e-3,
        device="cpu",
        seed=42,
        **net_params
    ):
        self.build_fn = build_fn
        self.loss_fn = loss_fn
        self.loss = get_loss_fn(loss_fn=loss_fn)
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        self.device = device
        self.net_params = net_params
        self.seed = seed
        torch.manual_seed(seed)
        self.net = None

    @classmethod
    def _unit_test_params(cls):
        def build_torch_linear_regressor(n_features):
            net = torch.nn.Sequential(
                torch.nn.Linear(n_features, 1), torch.nn.Sigmoid()
            )
            return net

        yield {
            "build_fn": build_torch_linear_regressor,
            "loss_fn": "mae",
            "optimizer_fn": torch.optim.SGD,
        }

    @classmethod
    def _unit_test_skips(self):
        """Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.
        """
        return {
            "check_pickling",
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    def _learn_one(self, x: torch.Tensor, y: torch.Tensor):
        self.net.train()
        self.net.zero_grad()
        y_pred = self.net(x)
        # depending on loss function
        try:
            loss = self.loss(y_pred, y)
        except:
            loss = self.loss(y_pred, torch.argmax(y, 1))  # TODO CHECK loss
        loss.backward()
        self.optimizer.step()

    def learn_one(self, x, y):
        if self.net is None:
            self._init_net(n_features=len(list(x.values())))

        x = torch.Tensor([list(x.values())])
        x = x.to(self.device)
        y = torch.Tensor([[y]])
        y = y.to(self.device)  # todo check if this works
        self._learn_one(x=x, y=y)
        return self

    def _filter_torch_params(self, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `torch_params`

        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.net_params.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res

    def _init_net(self, n_features):
        self.net = self.build_fn(
            n_features=n_features, **self._filter_torch_params(self.build_fn)
        )
        self.net.to(self.device)
        self.optimizer = self.optimizer_fn(self.net.parameters(), lr=self.learning_rate)


class RollingDeepEstimator(base.Estimator):
    def __init__(
        self,
        build_fn,
        loss_fn: str,
        optimizer_fn: Type[torch.optim.Optimizer],
        learning_rate=1e-3,
        window_size=1,
        seed=42,
        device="cpu",
        append_predict=True,
        **net_params
    ):
        self.build_fn = build_fn
        self.loss_fn = loss_fn
        self.loss = get_loss_fn(loss_fn=loss_fn)
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        self.device = device
        self.window_size = window_size
        self.net_params = net_params
        self.seed = seed
        self.append_predict = append_predict
        torch.manual_seed(seed)

        self._x_window = collections.deque(maxlen=window_size)
        self._batch_i = 0
        self.net = None

    @classmethod
    def _unit_test_params(cls):
        def build_torch_lstm_regressor(n_features, hidden_size):
            net = nn.Sequential(
                SequentialLSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1),
                nn.Linear(hidden_size, 10),
                nn.Linear(10, 1)
            )
            return net

        yield {
            "loss_fn": "mse",
            "build_fn": build_torch_lstm_regressor,
            "optimizer_fn": torch.optim.SGD,
            "hidden_size": 1,
            "window_size" : 2,
        }
    @classmethod
    def _unit_test_skips(self):
        """Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.
        """
        return {
            "check_pickling",
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    def _learn_batch(self, x: torch.Tensor, y: torch.Tensor):
        y_pred = self.net(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self

    def learn_one(self, x, y):
        self._x_window.append(list(x.values()))
        if self.net is None:
            self._init_net(n_features=len(list(x.values())))

        if len(self._x_window) == self.window_size:
            x = torch.Tensor([self._x_window])
            x = x.to(self.device)
            y = torch.Tensor([[y]])
            y = y.to(self.device)
            self._learn_batch(x=x, y=y)
        return self

    def _filter_torch_params(self, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `torch_params`

        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.net_params.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res

    def _init_net(self, n_features):
        self.net = self.build_fn(
            n_features=n_features, **self._filter_torch_params(self.build_fn)
        )
        self.net.to(self.device)
        self.optimizer = self.optimizer_fn(self.net.parameters(), self.learning_rate)


class AutoencodedAnomalyDetector(anomaly.AnomalyDetector, nn.Module):
    def __init__(
        self,
        encoder_fn,
        decoder_fn,
        loss_fn="smooth_mae",
        optimizer_fn: Type[torch.optim.Optimizer] = "sgd",

        device="cpu",
        scale_scores=True,
        window_size=250,
        **net_params
    ):
        super().__init__()
        self.encoder_fn = encoder_fn
        self.decoder_fn = decoder_fn
        self.encoder = None
        self.decoder = None
        self.loss_fn = get_loss_fn(loss_fn)
        self.optimizer_fn = get_optimizer_fn(optimizer_fn)
        self.net_params = net_params
        self.device = device
        self.mean_meter = stats.RollingMean(window_size) if scale_scores else None
        self.scale_scores = scale_scores
        self.window_size = window_size

    @classmethod
    def _unit_test_params(cls):

        def encoder_fn(n_features):
            return nn.Sequential(nn.Linear(n_features, math.ceil(n_features / 2)), nn.LeakyReLU())

        def decoder_fn(n_features):
            return nn.Sequential(nn.Linear(math.ceil(n_features/2), n_features))

        yield {
            "encoder_fn": encoder_fn,
            "decoder_fn": decoder_fn,
            "loss_fn": torch.nn.MSELoss,
            "optimizer_fn": torch.optim.SGD,
        }

    @classmethod
    def _unit_test_skips(self):
        """Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.
        """
        return {
            "check_pickling",
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    def learn_one(self, x):
        return self._learn(x)


    def _learn(self, x):
        x = dict2tensor(x, device=self.device)

        if self.encoder is None or self.decoder is None:
            self._init_net(n_features=x.shape[1])

        self.train()
        x_pred = self(x)
        loss = self.loss_fn(x_pred, x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scale_scores:
            self.mean_meter.update(loss.item())
        return self

    def score_one(self, x: dict):
        x = dict2tensor(x, device=self.device)

        if self.to_init:
            self._init_net(n_features=x.shape[1])

        self.eval()
        with torch.inference_mode():
            x_rec = self(x)
        loss = self.loss_fn(x_rec, x).item()
        if self.scale_scores and self.mean_meter.get() != 0:
            loss /= self.mean_meter.get()
        return loss

    def learn_many(self, x: pd.DataFrame):
        return self._learn(x)

    def score_many(self, x: pd.DataFrame) -> float:

        x = dict2tensor(x, device=self.device)

        if self.to_init:
            self._init_net(n_features=x.shape[1])

        self.eval()
        with torch.inference_mode():
            x_rec = self(x)
        loss = torch.mean(
            self.loss_fn(x_rec, x, reduction="none"),
            dim=list(range(1, x.dim())),
        )
        score = loss.cpu().detach().numpy()
        if self.scale_scores and self.mean_meter.get() != 0:
            score /= self.mean_meter.get()
        return score

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def _init_net(self, n_features):
        self.encoder = self.encoder_fn(
            n_features=n_features, **self._filter_args(self.build_fn)
        )
        self.decoder = self.decoder_fn(
            n_features=n_features, **self._filter_args(self.build_fn)
        )

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.optimizer = self.configure_optimizers()
        self.to_init = False

    def _filter_args(self, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `torch_params`

        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.net_params.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(
            nn.ModuleList([self.encoder, self.decoder]).parameters(),
            **self._filter_args(self.optimizer_fn),
        )
        return optimizer

    def _filter_args(self, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `torch_params`

        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.net_params.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res
