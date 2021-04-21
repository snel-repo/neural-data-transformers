# Neural Data Transformers
<p align="center">
    <img width="80%" src="assets/teaser.png" />
</p>

This is the code for the paper "Representation learning for neural population activity with Neural DataTransformers". We provide the code as reference, but we are unable to help debug specific issues e.g. in using the model, at this time.

## Setup
We recommend you set up your code environment with `conda/miniconda`.
The dependencies necessary for this project can then be installed with:
    `conda env create -f environment.yml`
This project was developed with Python 3.6.

## Data
The Lorenz dataset is provided in `data/lfads_lorenz.h5`. This file is stored on this repo with [`git-lfs`](https://git-lfs.github.com/). Therefore, if you've not used `git-lfs` before, please run `git lfs install` and `git lfs pull` to pull down the full h5 file.

The autonomous chaotic RNN dataset can be generated by running `./data/gen_synth_data_no_inputs.sh`. The generating script is taken from [the Tensorflow release](https://github.com/tensorflow/models/tree/master/research/lfads/synth_data) from LFADS, Sussillo et al. The maze dataset is unavailable at this time.

## Training + Evaluation
Experimental configurations are set in `./configs/`. To train a single model with a configuration `./configs/<variant_name>.yaml`, run `./scripts/train.sh <variant_name>`.

The provided sample configurations in `./configs/arxiv/` were used in the HP sweeps for the main results in the paper, with the sweep parameters in `./configs/*json`. Note that sweeping is done with the `ray[tune]` package. To run a sweep, run `python ray_random.py -e <variant_name>` (the same config system is used).

R2 is reported automatically for synthetic datasets. Maze analyses + configurations are unfortunately unavailable at this time.

## Analysis
Reference scripts that were used to produce most figures are available in `scripts`. They were created and iterated on as VSCode notebooks. They may require external information, run directories, even codebases etc. Scripts are provided to give a sense of analysis procedure, not to use as an out-of-the-box reproducibility notebook.

## Citation
```
@article {ye2021ndt,
	author = {Ye, Joel and Pandarinath, Chethan},
	title = {Representation learning for neural population activity with Neural Data Transformers},
	elocation-id = {2021.01.16.426955},
	year = {2021},
	doi = {10.1101/2021.01.16.426955},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/01/19/2021.01.16.426955},
	eprint = {https://www.biorxiv.org/content/early/2021/01/19/2021.01.16.426955.full.pdf},
	journal = {bioRxiv}
}
```
