# Environment

It is recommanded to build a new virtual environment.

## 1. Install pytorch and requirements.

```bash
# first install pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch

# then clone LATR and change directory to it to install requirements
cd ${LATR_PATH}
python -m pip install -r requirements.txt
```

## 2. Install mm packages

### 2.1 Install `mmcv`

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv && git checkout v1.5.0
FORCE_CUDA=1 MMCV_WITH_OPS=1 python -m pip install .
```

### 2.2 Install other mm packages

Install [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d). Note that we use `mmdet==2.24.0` and `mmdet3d==1.0.0rc3`.
