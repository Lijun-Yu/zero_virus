# Zero-VIRUS*: Zero-shot VehIcle Route Understanding System for Intelligent Transportation ([CVPR 2020 AI City Challenge Track 1](https://www.aicitychallenge.org/2020-data-and-evaluation/))

Authors: [Lijun Yu](https://me.lj-y.com), Qianyu Feng, Yijun Qian, Wenhe Liu, Alexander G. Hauptmann \
Email: lijun@lj-y.com

*Written in the era of Coronavirus Disease 2019 (COVID-19), with a sincere hope for a better world.

```bib
@inproceedings{yu2020zero,
  title={Zero-VIRUS: Zero-shot VehIcle Route Understanding System for Intelligent Transportation},
  author={Yu, Lijun and Feng, Qianyu and Qian, Yijun and Liu, Wenhe and Hauptmann, Alexander G.},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2020}
}
```

## Setup

Install [miniconda](https://conda.io/en/latest/miniconda.html), then create the environment and activate it via

```sh
conda env create -f environment.yml
conda activate zero_virus
```

Directory structure:

* datasets
  * Dataset_A (`AIC20_track1_vehicle_counting.zip/Dataset_A`)
  * Dataset_B (hidden evaluation)
* experiments
  * efficiency
    * aic2020-base.json
  * `<experiment_name>`
    * output.txt

## Evaluate

As a zero-shot system, no training is required.
We use [Mask R-CNN](http://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html) pretrained on [COCO](http://cocodataset.org/#home) from [detectron2](https://github.com/facebookresearch/detectron2) as detector, whose weights will be downloaded automatically at the first run.

As the dataset only provided screenshots of the pre-defined routes, we created our own [annotation](monitor/tracks) of them with [labelme](https://github.com/wkentaro/labelme).

To get system outputs, run

```sh
./evaluate.sh <experiment_name> <dataset_split>
# For example
./evaluate.sh submission Dataset_A
```

To get efficiency base score, run

```sh
python utils/efficiency_base.py
```

## Performance

On Dataset A with 8 V100 GPUs:

* S1: 0.9328
  * S1_Effectiveness: 0.9120
    * mwRMSE: 4.2738
  * S1_Efficiency: 0.9815
    * time: 3084.04
    * baseline: 0.546801

Visualizations available at [Google Drive](https://drive.google.com/drive/folders/1s3TPykPa3JTaPOHUVOQF8S4iUi3SduAN?usp=sharing).

## License

See [LICENSE](LICENSE). Please read before use.
