<h1 align="center"> OS-FPI: A Coarse-to-Fine One-Stream Network for UAV Geo-Localization and Navigation </h1>

This repository contains code the paper titled [OS-FPI: A Coarse-to-Fine One-Stream Network for UAV Geo-Localization and Navigation](https://arxiv.org/pdf/2208.06561).



## News

- **`2024/12/22`**: Our code are released.(Due to the author's extreme laziness, this is a very rough version)


## Prerequisites

- Python 3.7+
- GPU Memory >= 8G
- Numpy 1.26.0
- Pytorch 2.0.0+cu118
- Torchvision 0.15.0+cu118

## Installation

It is best to use cuda version 11.8 and pytorch version 2.0.0. You can download the corresponding version from this [website](https://download.pytorch.org/whl/torch_stable.html) and install it through `pip install`. Then you can execute the following command to install all dependencies.

```
pip install -r requirments.txt
```

Create the directory for saving the training log and ckpts.

```
mkdir checkpoints
```

## Dataset & Preparation

You can obtain the dataset from this [repository](https://github.com/Dmmm1997/DRL).

## Train & Evaluation

### Training and Testing

You could execute the following command to implement the entire process of training and testing.

```
bash train_test_local.sh
```

### Evaluation

The following code is used to test the performance of the model.

```
 tool/model_test_server.py
```

## Citation

The following paper uses and reports the result of the baseline model. You may cite it in your paper.

```bibtex
@article{drl,
  title={OS-FPI: A Coarse-to-Fine One-Stream Network for UAV Geo-Localization},
  author={Chen, Jiahao and Zheng, Enhui and Dai, Ming and Chen, Yifu and Lu, Yusheng},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```

## Related Work
- DenseUAV [https://github.com/Dmmm1997/DenseUAV](https://github.com/Dmmm1997/DenseUAV)
- FSRA [https://github.com/Dmmm1997/FSRA](https://github.com/Dmmm1997/FSRA)