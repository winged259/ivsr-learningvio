Originally forked from repository deep_ekf_vio

## Dependencies:
See `docker/cuda10.1/Dockerfile` for list of dependencies

## Example Usage:
Change parameters "/home/cs4li/Dev/deep_ekf_vio" to the appropriate directory in param.py
Help:
`python main.py -h`

#### KITTI Folder Layout:
```
path_to_KITTI_dir/
    - dataset/
        - 2011_09_30/
            - 2011_09_30_drive_0034_extract
                - image_02
                - oxts
            - ...
        - 2011_10_03/
            - 2011_10_03_drive_0027_extract
                - image_02
                - oxts
            - ...
```

#### EUROC Folder Layout:
```
path_to_EUROC_dir/
    - MH_01/
        - mav0/
            - 2011_09_30_drive_0034_extract
            - cam0
            - imu0
            - state_groundtruth_estimate0
        - ...
```

#### Preprocessing:
Change parameters "/home/cs4li/Dev/deep_ekf_vio" to the appropriate directory the shell scripts

`preprocess_kitti_seqs.sh` (need MATLAB with geographic lib installed)

`preprocess_euroc_seqs.sh`

#### Training:
Get the pretrain flownet weights from [here](https://drive.google.com/drive/folders/16eo3p9dO_vmssxRoZCmWkTpNjKRzJzn5).

`python3 main.py --description <experiment_desciprtion> --gpu_id 0`

#### Evaluation:
`python3 results_directory/main.py  --gpu_id 0 --run_eval_only`


## System Architecture:
![Alt text](docs/e2evio_system.png)
