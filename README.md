# SICARA: Active Learning For Object Detection

This project implements various active learning strategies for object detection. The object detection model used here is YOLOv5.

The goal of our project is to be able to evaluate an Object Detection model at an
early stage of a project. This will be studied through the implementation of a functional
interactive Active Learning pipeline for Object Detection, the evaluation of existing Active
Learning strategies using a weak model, and the investigation of the influence of Active
Learning parameters on the relevance of the selected data.

## Implemented Active Learning strategies:

- Random Sampling
- MaxProba
- EntropySampling
- BALD
- KMeansSampling
- KCenterGreedy_HF
- Hybrid strategies

## Installation

To begin run the following command in order to install required packages for yolov5 and active learning strategies: 

```bash
cd sicara/yolov5
pip install -r requirements.txt  # install
cd ..

```


## Getting started

### Creating the dataset

To create the dataset of 'cars' class, run the following code:

```python
create_coco_dataset(classes=['cars'], dataset_name='coco_cars', max_samples=10000)

```

### Running an experiment with dvc

To run an experiment, fix required parameters in the parameters.yaml script then run the following code. In order to register the results of the experiment after after a successful run with dvc, commit the results. 

```python
!dvc exp run -n experiment_name

```

### Visualization successful experiments

Experiments already executed and recorded in the experiment_hisory.yaml file can be visualized  using plot_experiments function. We can vizualise one of the following aspects of the experiments: results, maps, time, description and cost. Here is an example code: 

```python
import yaml 
from experiment_history import plot_experiments
# loading all experiments
exp_all = yaml.safe_load(open('experiment_history.yaml')) 

# selecting specific experiments
exp = {'Hybrid': ['1b8e460'], 'Random':['662b1d9'],'BALD':['8125ce5']}

# plot all experiments contained in the experiment_history.yaml file  
plot_experiments(show='results', **exp }) 
```

## Benchmark results

![global benchmark](figs/global.png)









