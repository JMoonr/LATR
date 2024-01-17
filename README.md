<br />
<p align="center">
  
  <h3 align="center"><strong>LATR: 3D Lane Detection from Monocular Images with Transformer</strong></h3>

<p align="center">
  <a href="https://arxiv.org/abs/2308.04583" target='_blank'>
    <!-- <img src="https://img.shields.io/badge/arXiv-%F0%9F%93%83-yellow"> -->
    <img src="https://img.shields.io/badge/arXiv-2308.04583-b31b1b.svg">
  </a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=JMoonr.LATR&left_color=gray&right_color=yellow">
  </a>
    <a href="https://github.com/JMoonr/LATR" target='_blank'>
     <img src="https://img.shields.io/github/stars/JMoonr/LATR?style=social">
  </a>
  
</p>


This is the official PyTorch implementation of [LATR: 3D Lane Detection from Monocular Images with Transformer](https://arxiv.org/abs/2308.04583).

![fig2](/assets/fig2.png)  

## News

  - **2024-01-15** :confetti_ball: Our new work [DV-3DLane: End-to-end Multi-modal 3D Lane Detection with Dual-view Representation](https://openreview.net/pdf?id=l1U6sEgYkb) is accepted by ICLR2024.

  - **2023-08-12** :tada: LATR is accepted as an Oral presentation at ICCV2023! :sparkles:


## Environments

To set up the required packages, please refer to the [installation guide](./docs/install.md).

## Data

Please follow [data preparation](./docs/data_preparation.md) to download dataset.

## Pretrained Models

Note that the performance of pretrained model is higher than our paper due to code refactoration and optimization. All models are uploaded to [google drive](https://drive.google.com/drive/folders/1AhvLvE84vayzFxa0teRHYRdXz34ulzjB?usp=sharing).

| Dataset | Pretrained | Metrics | md5 |
| - | - | - | - |
| OpenLane-1000 | [Google Drive](https://drive.google.com/file/d/1jThvqnJ2cUaAuKdlTuRKjhLCH0Zq62A1/view?usp=sharing) | F1=0.6297 | d8ecb900c34fd23a9e7af840aff00843 |
| OpenLane-1000 (Lite version) | [Google Drive](https://drive.google.com/file/d/1WD5dxa6SI2oR9popw3kO2-7eGM2z-IHY/view?usp=sharing) | F1=0.6212 | 918de41d0d31dbfbecff3001c49dc296 |
| ONCE | [Google Drive](https://drive.google.com/file/d/12kXkJ9tDxm13CyFbB1ddt82lJZkYEicd/view?usp=sharing) | F1=0.8125 | 65a6958c162e3c7be0960bceb3f54650 |
| Apollo-balance | [Google Drive](https://drive.google.com/file/d/1hGyNrYi3wAQaKbC1mD_18NG35gdmMUiM/view?usp=sharing) | F1=0.9697 | 551967e8654a8a522bdb0756d74dd1a2 |
| Apollo-rare | [Google Drive](https://drive.google.com/file/d/19VVBaWBnWiEqGx1zJaeXF_1CKn88G5v0/view?usp=sharing) | F1=0.9641 | 184cfff1d3097a9009011f79f4594138 |
| Apollo-visual | [Google Drive](https://drive.google.com/file/d/1ZzaUODYK2dyiG_2bDXe5tiutxNvc71M2/view?usp=sharing) | F1=0.9611 | cec4aa567c264c84808f3c32f5aace82 |


## Evaluation

You can download the [pretrained models](#pretrained-models) to `./pretrained_models` directory and refer to the [eval guide](./docs/train_eval.md#evaluation) for evaluation.

## Train

Please follow the steps in [training](./docs/train_eval.md#train) to train the model.

## Benchmark

### OpenLane

| Models | F1 | Accuracy | X error <br> near \| far | Z-error <br> near \| far |
| ----- | -- | -------- | ------- | ------- |
| 3DLaneNet | 44.1 | - | 0.479 \| 0.572 | 0.367 \| 0.443 |
| GenLaneNet | 32.3 | - | 0.593 \| 0.494 | 0.140 \| 0.195 |
| Cond-IPM | 36.3 | - | 0.563 \| 1.080 | 0.421 \| 0.892 |
| PersFormer | 50.5 | 89.5 | 0.319 \| 0.325 | 0.112 \| 0.141 |
| CurveFormer | 50.5 | - | 0.340 \| 0.772 | 0.207 \| 0.651 |
| PersFormer-Res50 | 53.0 | 89.2 | 0.321 \| 0.303 | 0.085 \| 0.118 |
| **LATR-Lite** | 61.5 | 91.9 | 0.225 \| 0.249 | 0.073 \| 0.106 |
| **LATR** | 61.9 | 92.0 | 0.219 \| 0.259 | 0.075 \| 0.104 |


### Apollo

Plaes kindly refer to our paper for the performance on other scenes.

<table>
    <tr>
        <td>Scene</td>
        <td>Models</td>
        <td>F1</td>
        <td>AP</td>
        <td>X error <br> near | far </td>
        <td>Z error <br> near | far </td>
    </tr>
    <tr>
        <td rowspan="8">Balanced Scene</td>
        <td>3DLaneNet</td>
        <td>86.4</td>
        <td>89.3</td>
        <td>0.068 | 0.477</td>
        <td>0.015 | 0.202</td>
    </tr>
    <tr>
        <td>GenLaneNet</td>
        <td>88.1</td>
        <td>90.1</td>
        <td>0.061 | 0.496</td>
        <td>0.012 | 0.214</td>
    </tr>
    <tr>
        <td>CLGo</td>
        <td>91.9</td>
        <td>94.2</td>
        <td>0.061 | 0.361</td>
        <td>0.029 | 0.250</td>
    </tr>
    <tr>
        <td>PersFormer</td>
        <td>92.9</td>
        <td>-</td>
        <td>0.054 | 0.356</td>
        <td>0.010 | 0.234</td>
    </tr>
    <tr>
        <td>GP</td>
        <td>91.9</td>
        <td>93.8</td>
        <td>0.049 | 0.387</td>
        <td>0.008 | 0.213</td>
    </tr>
    <tr>
        <td>CurveFormer</td>
        <td>95.8</td>
        <td>97.3</td>
        <td>0.078 | 0.326</td>
        <td>0.018 | 0.219</td>
    </tr>
    <tr>
        <td><b>LATR-Lite</b></td>
        <td>96.5</td>
        <td>97.8</td>
        <td>0.035 | 0.283</td>
        <td>0.012 | 0.209</td>
    </tr>
    <tr>
        <td><b>LATR</b?</td>
        <td>96.8</td>
        <td>97.9</td>
        <td>0.022 | 0.253</td>
        <td>0.007 | 0.202</td>
    </tr>
</table>


### ONCE

| Method     | F1  | Precision(%) | Recall(%) | CD error(m) |
| :- | :- | :- | :- | :- |   
| 3DLaneNet  | 44.73 | 61.46 | 35.16 | 0.127 |
| GenLaneNet | 45.59 | 63.95 | 35.42 | 0.121 |
| SALAD <ONCE-3DLane> | 64.07 | 75.90 | 55.42 | 0.098 |
| PersFormer | 72.07 | 77.82 | 67.11 | 0.086 |
| **LATR** | 80.59 | 86.12 | 75.73 | 0.052 |

## Acknowledgment

This library is inspired by [OpenLane](https://github.com/OpenDriveLab/PersFormer_3DLane), [GenLaneNet](https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection), [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [SparseInst](https://github.com/hustvl/SparseInst), [ONCE](https://github.com/once-3dlanes/once_3dlanes_benchmark) and many other related works, we thank them for sharing the code and datasets.


## Citation
If you find LATR is useful for your research, please consider citing the paper:

```tex
@article{luo2023latr,
  title={LATR: 3D Lane Detection from Monocular Images with Transformer},
  author={Luo, Yueru and Zheng, Chaoda and Yan, Xu and Kun, Tang and Zheng, Chao and Cui, Shuguang and Li, Zhen},
  journal={arXiv preprint arXiv:2308.04583},
  year={2023}
}
```