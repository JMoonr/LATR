# Data Preparation

## OpenLane

Follow [OpenLane](https://github.com/OpenDriveLab/PersFormer_3DLane#dataset) to download dataset and then link it under `data` directory.

```bash
cd data && mkdir openlane && cd openlane
ln -s ${OPENLANE_PATH}/images .
ln -s ${OPENLANE_PATH}/lane3d_1000 .
```

## ONCE

Follow [ONCE](https://github.com/once-3dlanes/once_3dlanes_benchmark#data-preparation) to download dataset, and then link it under `data` directory.

```bash
cd data
ln -s ${once} .
```

## Apollo

Follow [Apollo](https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection#data-preparation) to download dataset and link it under `data` directory.

```bash
cd data && mkdir apollosyn_gen-lanenet
cd apollosyn_gen-lanenet
ln -s ${Apollo_Sim_3D_Lane_Release} .
ln -s ${data_splits} .
```


Your data directory should be like:

```bash
|-- apollosyn_gen-lanenet
    |-- Apollo_Sim_3D_Lane_Release
    |   |-- depth
    |   |-- images
    |   |-- img_list.txt
    |   |-- labels
    |   |-- laneline_label.json
    |   `-- segmentation
    `-- data_splits
        |-- illus_chg
        |-- rare_subset
        `-- standard
|-- Load_Data.py
|-- __init__.py
|-- apollo_dataset.py
|-- once -> ${once}
    |-- annotation
    |-- data
    |-- data_check.py
    |-- list
    |-- raw_cam01
    |-- raw_cam_multi
    |-- train
    `-- val
|-- openlane
|   |-- images -> ${openlane}/images/
        |-- training
        `-- validation
|   `-- lane3d_1000 -> ${openlane}/lane3d_1000/
        |-- test
        |-- training
        `-- validation
`-- transform.py
```