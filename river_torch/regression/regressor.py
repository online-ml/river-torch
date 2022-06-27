import copy
from typing import Type

import torch
from DeepRiver.base import DeepEstimator, RollingDeepEstimator
from river import base


class Regressor(DeepEstimator, base.Regressor):
    """Compatibility layer from PyTorch to River for regression.

    Parameters
    ----------
    build_fn
    loss_fn
    optimizer_fn
    batch_size

    Examples
    --------

    >>> from river import compat
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import preprocessing
    >>> import torch
    >>> from torch import nn
    >>> from torch import optim

    >>> _ = torch.manual_seed(0)

    >>> dataset = datasets.TrumpApproval()

    >>> n_features = 6
    >>> net = nn.Sequential(
    ...     nn.Linear(n_features, 3),
    ...     nn.Linear(3, 1)
    ... )

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     compat.Regressor(
    ...         net=net,
    ...         loss_fn=nn.MSELoss(),
    ...         optimizer_fn=optim.SGD(net.parameters(), lr=1e-3),
    ...         batch_size=2
    ...     )
    ... )
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric).get()
    2.78258

    """

    def __init__(
        self,
        build_fn,
        optimizer_fn: Type[torch.optim.Optimizer],
        loss_fn="mse",
        device="cpu",
        learning_rate=1e-3,
        **net_params
    ):
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            device=device,
            optimizer_fn=optimizer_fn,
            learning_rate=learning_rate,
            **net_params
        )

    def predict_one(self, x):
        if self.net is None:
            self._init_net(len(x))
        x = torch.Tensor(list(x.values()))
        return self.net(x).item()


class RollingRegressor(RollingDeepEstimator, base.Regressor):
    """
    A Rolling Window PyTorch to River Regressor
    Parameters
    ----------
    build_fn
    loss_fn
    optimizer_fn
    window_size
    learning_rate
    net_params
    """

    def predict_one(self, x: dict):
        if self.net is None:
            self._init_net(len(list(x.values())))
        if len(self._x_window) == self.window_size:

            if self.append_predict:
                self._x_window.append(list(x.values()))
                x = torch.Tensor([self._x_window])
            else:
                l = copy.deepcopy(self._x_window)
                l.append(list(x.values()))
                x = torch.Tensor([l])
            return self.net(x).item()
        else:
            return 0.0
