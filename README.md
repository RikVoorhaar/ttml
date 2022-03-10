# ttml
Tensor train based machine learning estimator.

Uses existing machine learning estimators to initialize a tensor train
decomposition on a particular feature space discretization. Then this tensor
train is further optimized with Riemannian conjugate gradient descent.

This library also implements much functionality related to tensor trains, and
Riemannian optimization of tensor trains. Finally there is some functionality
for turning decision trees and forests into CP/tensor trains.

This software is the companion to the preprint [arXiv:2203.04352](https://doi.org/10.48550/arXiv.2203.04352)

# Installation

The `ttml` python package can be installed using `pip` by running
```
    pip install ttml
```
or by cloning this repository and running the following command in the root directory of this project:
```
    git clone git@github.com:RikVoorhaar/ttml.git
    pip install .
```

If you want to reproduce the experiments discussed in our paper, then first clone this repository. Then run
the script `datasets/download_datasets.py` to download all relevant datasets from the [UCI Machine Learning
Repository](https://archive.ics.uci.edu/ml/index.php). Then all figures and results can be reproduced by the
scripts in the `notebooks` folder. To install all the dependencies for the scripts and tests, you can use the
`conda` environment defined in `environment.yml`. 

# Documentation

The documentation for this project lives on [ttml.readthedocs.io](https://ttml.readthedocs.io/en/latest/).

For a more informal explanation of how the machine learning estimator works, see [this blog post](https://www.rikvoorhaar.com/discrete-function-tensor/).

# Credits
All code for this library has been written by [Rik Voorhaar](https://www.rikvoorhaar.com/), in a joint project
with [Bart Vandereycken](https://www.unige.ch/math/vandereycken/). This work was supported by the Swiss
National Science Foundation under [research project 192363](https://data.snf.ch/grants/grant/192363).

This software is free to use and edit. When using this software for academic purposes, please cite the following preprint:
```
@article{
    title = {TTML: Tensor Trains for general supervised machine learning},
    journal = {arXiv:2203.04352},
    author = {Vandereycken, Bart and Voorhaar, Rik},
    year = {2022}, 
}
```

All figures in the preprint have been produced using version 1.0 of this software.