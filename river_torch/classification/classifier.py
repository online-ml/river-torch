import copy
import math
from typing import Callable, Dict, Union

import torch
from river import base
from river.base.typing import ClfTarget
from torch.nn import init, parameter

from river_torch.base import DeepEstimator, RollingDeepEstimator
from river_torch.utils.layers import find_output_layer
from river_torch.utils.river_compat import (
    dict2tensor,
    list2tensor,
    output2proba,
    target2onehot,
)


class Classifier(DeepEstimator, base.Classifier):
    """
    Wrapper for PyTorch classification models that automatically handles increases in the number of classes by adding output neurons in case the number of observed classes exceeds the current number of output neurons.

    Parameters
    ----------
    build_fn
        Function that builds the PyTorch classifier to be wrapped. The function should accept parameter `n_features` so that the returned model's input shape can be determined based on the number of features in the initial training example. For the dynamic adaptation of the number of possible classes, the returned network should be a torch.nn.Sequential model with a Linear layer as the last module.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss function provided by `torch.nn.functional` or one of the following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy', 'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model. Can be an optimizer class provided by `torch.optim` or one of the following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **net_params
        Parameters to be passed to the `build_fn` function aside from `n_features`.

    Examples
    --------

    >>> from river.datasets import Phishing
    >>> from river import evaluate, metrics, preprocessing
    >>> from torch import nn, optim, manual_seed
    >>> from river_torch.classification import Classifier
    >>> _ = manual_seed(0)
    >>> def build_torch_mlp_classifier(n_features):
    ...     net = nn.Sequential(
    ...         nn.Linear(n_features, 5),
    ...         nn.ReLU(),
    ...         nn.Linear(5, 2),
    ...         nn.Softmax(dim=-1),
    ...     )
    ...     return net
    ...
    >>> model = Classifier(build_fn=build_torch_mlp_classifier, loss_fn='binary_cross_entropy',optimizer_fn=optim.Adam, lr=1e-3)
    >>> dataset = Phishing()
    >>> metric = metrics.Accuracy()
    >>> evaluate.progressive_val_score(dataset=dataset, model=model, metric=metric)
    Accuracy: 79.82%
    """

    def __init__(
        self,
        build_fn: Callable,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **net_params,
    ):
        self.observed_classes = []
        self.output_layer = None
        super().__init__(
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            build_fn=build_fn,
            device=device,
            lr=lr,
            seed=seed,
            **net_params,
        )

    def learn_one(self, x: dict, y: ClfTarget, **kwargs) -> "Classifier":
        """
        Performs one step of training with a single example.

        Parameters
        ----------
        x
            Input example.
        y
            Target value.

        Returns
        -------
        Classifier
            The classifier itself.
        """
        # check if model is initialized
        if self.net is None:
            self._init_net(len(x))
        x = dict2tensor(x, device=self.device)

        # check last layer
        if y not in self.observed_classes:
            self.observed_classes.append(y)

        if self.output_layer is None:
            self.output_layer = find_output_layer(self.net)

        out_features_target = (
            len(self.observed_classes) if len(self.observed_classes) > 2 else 1
        )
        n_classes_to_add = out_features_target - self.output_layer.out_features
        if n_classes_to_add > 0:
            self._add_output_dims(n_classes_to_add)

        y = target2onehot(
            y,
            self.observed_classes,
            self.output_layer.out_features,
            device=self.device,
        )
        self.net.train()
        return self._learn_one(x=x, y=y)

    def _add_output_dims(self, n_classes_to_add: int) -> None:
        """
        Adds output dimensions to the model by adding new rows of weights to the existing weights of the last layer.

        Parameters
        ----------
        n_classes_to_add
            Number of output dimensions to add.
        """
        new_weights = torch.empty(n_classes_to_add, self.output_layer.in_features)
        init.kaiming_uniform_(new_weights, a=math.sqrt(5))
        self.output_layer.weight = parameter.Parameter(
            torch.cat([self.output_layer.weight, new_weights], axis=0)
        )

        if self.output_layer.bias is not None:
            new_bias = torch.empty(n_classes_to_add)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.output_layer.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(new_bias, -bound, bound)
            self.output_layer.bias = parameter.Parameter(
                torch.cat([self.output_layer.bias, new_bias], axis=0)
            )
        self.output_layer.out_features += n_classes_to_add
        self.optimizer = self.optimizer_fn(self.net.parameters(), lr=self.lr)

    def _learn_one(self, x: torch.TensorType, y: torch.TensorType):
        self.optimizer.zero_grad()
        y_pred = self.net(x)[0]
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return self

    def predict_proba_one(self, x: dict) -> Dict[ClfTarget, float]:
        """
        Predict the probability of each label given the input.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        Dict[ClfTarget, float]
            Dictionary of probabilities for each label.
        """
        if self.net is None:
            self._init_net(len(x))
        x = dict2tensor(x, device=self.device)
        self.net.eval()
        y_pred = self.net(x)
        return output2proba(y_pred, self.observed_classes)


class RollingClassifier(RollingDeepEstimator, base.Classifier):
    """
    Wrapper that feeds a sliding window of the most recent examples to the wrapped PyTorch classification model. The class also automatically handles increases in the number of classes by adding output neurons in case the number of observed classes exceeds the current number of output neurons.

    Parameters
    ----------
    build_fn
        Function that builds the PyTorch classifier to be wrapped. The function should accept parameter `n_features` so that the returned model's input shape can be determined based on the number of features in the initial training example. For the dynamic adaptation of the number of possible classes, the returned network should be a torch.nn.Sequential model with a Linear layer as the last module.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss function provided by `torch.nn.functional` or one of the following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy', 'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model. Can be an optimizer class provided by `torch.optim` or one of the following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    window_size
        Number of recent examples to be fed to the wrapped model at each step.
    append_predict
        Whether to append inputs passed for prediction to the rolling window.
    **net_params
        Parameters to be passed to the `build_fn` function aside from `n_features`.
    """

    def __init__(
        self,
        build_fn: Callable,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 10,
        append_predict: bool = False,
        **net_params,
    ):
        self.observed_classes = []
        self.output_layer = None
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            window_size=window_size,
            append_predict=append_predict,
            device=device,
            lr=lr,
            seed=seed,
            **net_params,
        )

    def learn_one(self, x: dict, y: ClfTarget, **kwargs) -> "Classifier":
        """
        Performs one step of training with the most recent training examples stored in the sliding window.

        Parameters
        ----------
        x
            Input example.
        y
            Target value.

        Returns
        -------
        Classifier
            The classifier itself.
        """
        # check if model is initialized
        if self.net is None:
            self._init_net(len(x))

        self._x_window.append(list(x.values()))

        # check last layer
        if y not in self.observed_classes:
            self.observed_classes.append(y)

        if self.output_layer is None:
            self.output_layer = find_output_layer(self.net)

        out_features_target = (
            len(self.observed_classes) if len(self.observed_classes) > 2 else 1
        )
        n_classes_to_add = out_features_target - self.output_layer.out_features
        if n_classes_to_add > 0:
            self._add_output_dims(n_classes_to_add)

        # training process
        if len(self._x_window) == self.window_size:
            self.net.train()
            x = list2tensor(self._x_window, device=self.device)
            y = target2onehot(
                y,
                self.observed_classes,
                self.output_layer.out_features,
                device=self.device,
            )
            self._learn_window(x=x, y=y)
        return self

    def _add_output_dims(self, n_classes_to_add: int) -> None:
        """
        Adds output dimensions to the model by adding new rows of weights to the existing weights of the last layer.

        Parameters
        ----------
        n_classes_to_add
            Number of output dimensions to add.
        """
        new_weights = torch.empty(n_classes_to_add, self.output_layer.in_features)
        init.kaiming_uniform_(new_weights, a=math.sqrt(5))
        self.output_layer.weight = parameter.Parameter(
            torch.cat([self.output_layer.weight, new_weights], axis=0)
        )

        if self.output_layer.bias is not None:
            new_bias = torch.empty(n_classes_to_add)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.output_layer.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(new_bias, -bound, bound)
            self.output_layer.bias = parameter.Parameter(
                torch.cat([self.output_layer.bias, new_bias], axis=0)
            )
        self.output_layer.out_features += n_classes_to_add
        self.optimizer = self.optimizer_fn(self.net.parameters(), lr=self.lr)

    def _learn_window(self, x: torch.TensorType, y: torch.TensorType):
        self.optimizer.zero_grad()
        y_pred = self.net(x)[0]
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

    def predict_proba_one(self, x: dict) -> Dict[ClfTarget, float]:
        """
        Predict the probability of each label given the most recent examples stored in the sliding window.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        Dict[ClfTarget, float]
            Dictionary of probabilities for each label.
        """
        if self.net is None:
            self._init_net(len(x))

        if self.append_predict:
            self._x_window.append(list(x.values()))
            x = self._x_window
        else:
            x = copy.deepcopy(self._x_window)
            x.append(list(x.values()))

        if len(self._x_window) == self.window_size:
            x = list2tensor(x, device=self.device)
            self.net.eval()
            y_pred = self.net(x)
            proba = output2proba(y_pred, self.observed_classes)
        else:
            mean_proba = 1 / len(self.observed_classes)
            proba = {c: mean_proba for c in self.observed_classes}
        return proba
