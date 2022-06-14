![GitHub last commit](https://img.shields.io/github/last-commit/kulbachcedric/DeepRiver)
[![unit-tests](https://github.com/kulbachcedric/DeepRiver/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/kulbachcedric/DeepRiver/actions/workflows/unit-tests.yml)
![Codecov](https://img.shields.io/codecov/c/github/kulbachcedric/DeepRiver)
[![docs](https://github.com/kulbachcedric/IncrementalTorch/actions/workflows/mkdocs.yml/badge.svg)](https://github.com/kulbachcedric/IncrementalTorch/actions/workflows/unit_test.yml)

<p align="center">
  <img height="150px" src="docs/img/logo.png" alt="incremental dl logo">
</p>

<p align="center">
    IncrementalTorch is a Python library for online deep learning.
    IncrementalTorch ambition is to enable <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a> for neural networks.
    It combines the <a href="https://www.riverml.xyz">river</a> API with the capabilities of designing neural networks based on <a href="https://pytorch.org">PyTorch</a>.
</p>

## 💈 Installation
```shell
pip install deepriver
```
You can install the latest development version from GitHub as so:
```shell
pip install https://github.com/kulbachcedric/DeepRiver.git --upgrade
```

Or, through SSH:
```shell
pip install git@github.com:kulbachcedric/DeepRiver.git --upgrade
```


## 🍫 Quickstart
We build the development of neural networks on top of the <a href="https://www.riverml.xyz">river API</a> and refer to the rivers design principles.
The following example creates a simple MLP architecture based on PyTorch and incrementally predicts and trains on the website phishing dataset.
For further examples check out the <a href="http://kulbachcedric.github.io/DeepRiver/">Documentation</a>.
### Classification
```python
from river import datasets
from river import metrics
from river import preprocessing
from river import compose
from DeepRiver import classification
from torch import nn
from torch import optim
from torch import manual_seed

_ = manual_seed(0)


def build_torch_mlp_classifier(n_features):  # build neural architecture
    net = nn.Sequential(
        nn.Linear(n_features, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 1),
        nn.Sigmoid()
    )
    return net


model = compose.Pipeline(
    preprocessing.StandardScaler(),
    classification.Classifier(build_fn=build_torch_mlp_classifier, loss_fn='bce', optimizer_fn=optim.Adam,
                              learning_rate=1e-3)
)

dataset = datasets.Phishing()
metric = metrics.Accuracy()

for x, y in dataset:
    y_pred = model.predict_one(x)  # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model.learn_one(x, y)  # make the model learn

print(f'Accuracy: {metric.get()}')
```

### Anomaly Detection

```python
import math

from river import datasets, metrics
from DeepRiver.base import AutoencodedAnomalyDetector
from DeepRiver.utils import get_activation_fn
from torch import manual_seed, nn

_ = manual_seed(0)

def get_fully_connected_encoder(activation_fn="selu", dropout=0.5, n_features=3):
    activation = get_activation_fn(activation_fn)

    encoder = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features=n_features, out_features=math.ceil(n_features / 2)),
        activation(),
        nn.Linear(in_features=math.ceil(n_features / 2), out_features=math.ceil(n_features / 4)),
        activation(),
    )
    return encoder

def get_fully_connected_decoder(activation_fn="selu", dropout=0.5, n_features=3):
    activation = get_activation_fn(activation_fn)

    decoder = nn.Sequential(
        nn.Linear(in_features=math.ceil(n_features / 4), out_features=math.ceil(n_features / 2)),
        activation(),
        nn.Linear(in_features=math.ceil(n_features / 2), out_features=n_features),
    )
    return decoder

if __name__ == '__main__':

    dataset = datasets.CreditCard().take(5000)
    metric = metrics.ROCAUC()

    model = AutoencodedAnomalyDetector(encoder_fn=get_fully_connected_encoder, decoder_fn=get_fully_connected_decoder, lr=0.01)

    for x, y in dataset:
        score = model.score_one(x)
        metric.update(y_true=y, y_pred=score)
        model.learn_one(x=x)
    print(f'ROCAUC: {metric.get()}')
```

## 🏫 Affiliations
<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>
