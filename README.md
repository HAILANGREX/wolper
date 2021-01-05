# Wolpertinger Training with DDPG for Cache Environment(Pytorch, Multi-GPU/single-GPU/CPU)
## Overview
Pytorch version of Wolpertinger Training for cache environment with DDPG. <br>
The code is compatible with training in multi-GPU, single-GPU or CPU. <br>

## Dependencies
* python 3.6.8
* torch 1.1.0
* gym 0.14.0
* [pyflann](http://www.galaxysofts.com/new/pyflann-for-python-3x/)
  * This is the library (FLANN, [Muja & Lowe, 2014](https://ieeexplore.ieee.org/abstract/document/6809191)) with approximate nearest-neighbor methods allowed for logarithmic-time lookup complexity relative to the number of actions. However, the python binding of FLANN (pyflann) is written for python 2 and is no longer maintained. Please refer to [pyflann](http://www.galaxysofts.com/new/pyflann-for-python-3x/) for the pyflann package compatible with python3. Just download and place it in your (virtual) environment.

## Usage
* To use CPU only:
    ```
    $ python main.py --gpu-ids -1
    ```
* To use single-GPU only:
    ```
    $ python main.py --gpu-ids 0 --gpu-nums 1
    ```
* To use multi-GPU (e.g., use GPU-0 and GPU-1):
    ```
    $ python main.py --gpu-ids 0 1 --gpu-nums 2
    ```
* You can set your experiment parameters in the [`arg_parser.py`](./arg_parser.py)
 
## Supplement
* The [`train_test.py`](./train_test.py) is used for the baseline experiment.
* The [`train_test_window.py`](./train_test_window.py) is used for the window experiment.

## Result
* Please refer to [`output`](./output) for the trained policy and training log.
* The  [`runs`](./runs) is tensorboard result

## Project Reference
* [Original paper of Wolpertinger Training with DDPG, Google DeepMind](https://arxiv.org/abs/1512.07679)
* I used and modified part of the code in https://github.com/ghliu/pytorch-ddpg under Apache License 2.0.
* I used and modified part of the code in https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces under MIT License.
