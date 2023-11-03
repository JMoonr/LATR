# Training and Evaluation

## Train

### Openlane

- Base version:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iccv/latr_1000_baseline.py
```

- lite version:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iccv/latr_1000_baseline_lite.py
```

### ONCE

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iccv/once.py
```

### Apollo

- Balanced Scene

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iccv/apollo_standard.py
```

- Rare Subset

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iccv/apollo_rare.py
```

- Visual Variations

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port=29284 --nproc_per_node 4 main.py --config config/release_iccv/apollo_illu.py
```

## Evaluation

### Openlane

- Base version:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iccv/latr_1000_baseline.py --cfg-options evaluate=true eval_ckpt=pretrained_models/openlane.pth
```

- lite version:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iccv/latr_1000_baseline_lite.py --cfg-options evaluate=true eval_ckpt=pretrained_models/openlane_lite.pth
```

### ONCE

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iccv/once.py --cfg-options evaluate=true eval_ckpt=pretrained_models/once.pth
```

### Apollo

- Balanced Scene

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iccv/apollo_standard.py --cfg-options evaluate=true eval_ckpt=pretrained_models/apollo_standard.pth
```

- Rare Subset

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node 4 main.py --config config/release_iccv/apollo_rare.py --cfg-options evaluate=true eval_ckpt=pretrained_models/apollo_rare.pth
```

- Visual Variations

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --master_port=29284 --nproc_per_node 4 main.py --config config/release_iccv/apollo_standard.py --cfg-options evaluate=true eval_ckpt=pretrained_models/apollo_illu.pth
```