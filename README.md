<h1 align="center"> OS-FPI: A Coarse-to-Fine One-Stream Network for UAV Geo-Localization</h1>

This repository contains code for the paper titled [OS-FPI: A Coarse-to-Fine One-Stream Network for UAV Geo-Localization](https://ieeexplore.ieee.org/document/10478125).

![](img/OS_FPI_backbone_v1.drawio.png)

## News
- **`2024/12/22`**: Strongly recommend DRL's article and its [repository](https://github.com/Dmmm1997/DRL)
- **`2024/12/22`**: Our code are released.(Due to the author's extreme laziness, this is a very rough version)


## Prerequisites

- Python 3.7+
- GPU Memory >= 8G
- Numpy 1.26.0
- Pytorch 2.0.0+cu118
- Torchvision 0.15.0+cu118

## Installation

```
pip install -r requirments.txt
```

Create the directory for saving the training log and ckpts.

```
mkdir checkpoints
```

## Dataset

You can obtain the dataset from this [repository](https://github.com/Dmmm1997/DRL).

## Train
You can use the following script for training while obtaining test results. Remember to change the config and dataset addresses before use.

```
bash train_test_local.sh
```

### Testing

The following code is used to test the performance of the model.

```
 tool/model_test_server.py
```

## Pretrained model
The relevant model weights have already been uploaded to this repository.

```
 net_040.pth
```

### Visualization

You can use the following code to visualize the results.

```
 tool/demo_visualization.py
 tool/demo_visualization_xy.py
```

## Citation

Please consider citing our papers in your publications if the project helps your research. BibTeX reference is as follows.
```bibtex
@ARTICLE{OS-FPI,
  title={OS-FPI: A Coarse-to-Fine One-Stream Network for UAV Geo-Localization},
  author={Chen, Jiahao and Zheng, Enhui and Dai, Ming and Chen, Yifu and Lu, Yusheng},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2024},
  publisher={IEEE}
}

@ARTICLE{wamf-fpi,
  title={Wamf-fpi: A weight-adaptive multi-feature fusion network for uav localization},
  author={Wang, Guirong and Chen, Jiahao and Dai, Ming and Zheng, Enhui},
  journal={Remote Sensing},
  volume={15},
  number={4},
  pages={910},
  year={2023},
  publisher={MDPI}
}

@misc{drl,
      title={Drone Referring Localization: An Efficient Heterogeneous Spatial Feature Interaction Method For UAV Self-Localization}, 
      author={Ming Dai and Enhui Zheng and Zhenhua Feng and Jiahao Chen and Wankou Yang},
      year={2024},
      eprint={2208.06561},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2208.06561}, 
}

@ARTICLE{DenseUAV,
  author={Dai, Ming and Zheng, Enhui and Feng, Zhenhua and Qi, Lei and Zhuang, Jiedong and Yang, Wankou},
  journal={IEEE Transactions on Image Processing},
  title={Vision-Based UAV Self-Positioning in Low-Altitude Urban Environments},
  year={2024},
  volume={33},
  number={},
  pages={493-508},
  doi={10.1109/TIP.2023.3346279}}
```

## Related Work
- WAMF-FPI [https://www.mdpi.com/2072-4292/15/4/910](https://www.mdpi.com/2072-4292/15/4/910)
- DRL [https://github.com/Dmmm1997/DRL](https://github.com/Dmmm1997/DRL)
- DenseUAV [https://github.com/Dmmm1997/DenseUAV](https://github.com/Dmmm1997/DenseUAV)
- FSRA [https://github.com/Dmmm1997/FSRA](https://github.com/Dmmm1997/FSRA)
