# Setup and Usage

- Use Conda and Python3.6+ (3.8 Recommended)
- use `requirements.txt` to install the dependencies

# About

This Repo implements the baseline from [On Detecting Adversarial Perturbations](https://arxiv.org/abs/1702.04267) - Jan Hendrik Metzen, Tim Genewein, Volker Fischer, Bastian Bischoff


## BaseModel

- the basemodel is a resnet like classifier as defined in the above paper
- train the basemodel using `train.py`
- store the Outputs and the Layer Outputs for each CIFAR input  

## Adversarial Detectors

- train the adversarial detectors using `train_adversarial_detectors.py`
- `infer.py` contains functions to facilitate inference on the basemodel and also the adversarial detectors
- `run_adv_detector.py` uses various functions from the other files and generates a full dataframe for the datasets with the results of the individual files. This can be visualized using `visualize.ipynb`
- An html visualization has been generated using [this notebook](../results_analysis/merge_data.ipynb) and can be seen here
    - https://tarangshah.com/adversarial-detection/results_analysis/baseline1_advimages_visualization.html
    -  https://tarangshah.com/adversarial-detection/results_analysis/baseline1_realimages_visualization.html
