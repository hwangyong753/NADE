# Climate Modeling with Neural Advection-Diffusion Equation
[![BigDyL Link](https://img.shields.io/static/v1?label=&message=BigDyL&color=blue)](https://sites.google.com/view/npark/home?authuser=0) ![GitHub Repo stars](https://img.shields.io/github/stars/hwangyong753/NADE)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhwangyong753%2FNADE&count_bg=%230BADED&title_bg=%233B2424&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/climate-modeling-with-neural-advection/weather-forecasting-on-la)](https://paperswithcode.com/sota/weather-forecasting-on-la?p=climate-modeling-with-neural-advection)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/climate-modeling-with-neural-advection/weather-forecasting-on-sd)](https://paperswithcode.com/sota/weather-forecasting-on-sd?p=climate-modeling-with-neural-advection)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/climate-modeling-with-neural-advection/weather-forecasting-on-noaa-atmospheric)](https://paperswithcode.com/sota/weather-forecasting-on-noaa-atmospheric?p=climate-modeling-with-neural-advection)

This repository contains the implementation of NADE (Neural Advection-Diffusion Equation), a novel deep learning approach for climate modeling as described in our paper "[Climate Modeling with Neural Advection-Diffusion Equation](https://github.com/hwangyong753/NADE)".

> :arrow_right: NADE is the successor to our previous model, [Neural Diffusion Equation (NDE)](https://ieeexplore.ieee.org/abstract/document/9679162). Please refer to the [NDE GitHub repository](https://github.com/jeehyunHwang/Neural-Diffusion-Equation) and [paper](https://ieeexplore.ieee.org/abstract/document/9679162) for more details.
>
> :bulb: To see other papers from BigDyL, please visit [BigDyL homepage](https://sites.google.com/view/npark) or [awesome-bigdyl](https://github.com/bigdyl-kaist/awesome-bigdyl) to see the list of papers with links to papers and code.
>
> :mag_right: If you're interested in the application of reaction-diffusion equations to graph neural networks for node classification tasks, we recommend checking out our [GREAD (Graph Neural Reaction-Diffusion Networks)](https://proceedings.mlr.press/v202/choi23a) model. GREAD explores a comprehensive set of reaction equations in combination with diffusion processes, offering potential benefits in mitigating oversmoothing and handling varying levels of graph homophily. For more details, please refer to the [GREAD GitHub repository](https://github.com/jeongwhanchoi/gread) and [paper](https://proceedings.mlr.press/v202/choi23a).

## Overview
<img src="asset/overview_NADE.png" width="800">
NADE combines the advection-diffusion equation with neural networks to model climate systems. Key features include:

- Edge weight learning to model diffusion coefficients and flow velocities
- Neural network-based uncertainty modeling
- Integration with the advection-diffusion equation

### Comparison between NDE and NADE
| Feature | NDE | NADE |
| --- | --- | --- |
| Base Equation | Diffusion Equation | Advection-Diffusion Equation|
|Physical Processes Modeled| Diffusion | Advection and Diffusion | 
| Edge Weight | Learning Heat capacity generation | Diffusion coefficients and velocities |
| Uncertainty Modeling | ✓ | ✓ | 

NADE builds upon the success of NDE by incorporating advection processes, allowing it to model a wider range of climate phenomena with improved accuracy.

## Usage
To run the model:
```
$ cd model
$ python Run.py --dataset DATASET_NAME --mode MODE --device DEVICE
```
where:
- `DATASET_NAME` can be 'LA', 'SD', or 'NOAA'
- `MODE` can be 'train' or 'test'
- `DEVICE` can be 'cuda' or 'cpu'

Example:
```
python Run.py --dataset NOAA --mode train --device cuda
```

###  Model details
- The model supports different types: 
    - `AD` (Advection-Diffusion),
    - `diff` (Diffusion only),
    - `adv` (Advection only), 
    - `k` (constant coefficient), 
    - `withoutf` (without uncertainty modeling),
    - `onlyf` (only uncertainty modeling).
- Time dependence and time division can be enabled/disabled via configuration.
- Supports various loss functions: MAE, MSE, and masked MAE.
- Learning rate decay and early stopping are implemented.

## Datasets
We provide experiments on real-world datasets:
- Los Angeles (LA) and San Diego (SD) climate data
- NOAA temperature data

Each dataset has its own configuration file (e.g., `LA.conf`, `SD.conf`, `NOAA.conf`) in `model` directory.

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{choi2023climate,
title={Climate modeling with neural advection--diffusion equation},
author={Choi, Hwangyong and Choi, Jeongwhan and Hwang, Jeehyun and Lee, Kookjin and Lee, Dongeun and Park, Noseong},
journal={Knowledge and Information Systems},
volume={65},
number={6},
pages={2403--2427},
year={2023},
publisher={Springer}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hwangyong753/NADE&type=Date)](https://star-history.com/#hwangyong753/NADE&Date)