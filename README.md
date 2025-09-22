# FlowerVLA

[Paper](https://www.arxiv.org/pdf/2509.04996), [Project Page](https://intuitive-robots.github.io/flower_vla/), [Finetuning Code](https://github.com/intuitive-robots/flower_vla_calvin)

[Moritz Reuss](https://mbreuss.github.io/)<sup>1</sup>,
[Hongyi Zhou](https://hongyizhoucn.github.io/)<sup>1</sup>,
[Marcel Ruehle]()<sup>1</sup>,
[Ömer Erdinç Yağmurlu](https://scholar.google.com/citations?user=I_Mxp5cAAAAJ&hl=en)<sup>1</sup>,
[Fabian Otto](https://ottofabian.github.io/)<sup>2</sup>,
[Rudolf Lioutikov](http://rudolf.intuitive-robots.net/)<sup>1</sup>

<sup>1</sup>Intuitive Robots Lab (IRL), Karlsruhe Institute of Technology (KIT)
<sup>2</sup>Microsoft Research


## An efficient Vision-Language-Action Model for Robot Learning

FLOWER VLA is a lightweight, efficient Vision-Language-Action (VLA) policy for robotic manipulation tasks that achieves state-of-the-art performance on multiple benchmarks. Built on a rectified flow architecture with several key architecture features:

- **Efficient Architecture**: At less than ~1B parameters, FLOWER is significantly smaller than other VLA models
- **Low Training Cost**: Only requires ~200 GPU hours of pretraining
- **Low Memory Footprint**: Uses <8GB of GPU memory for inference
- **SOTA Performance**: Achieves sota results on CALVIN and LIBERO benchmarks

For the finetuning code for FLOWER for CALVIN and LIBERO heck out our other codebase: [flower_vla_calvin](https://github.com/intuitive-robots/flower_vla_calvin)

## Table of Contents

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Basic Setup](#basic-setup)
  - [Optional Dependencies](#optional-dependencies)
- [Pretraining Guide](#pretraining-guide)
  - [Dataset Preparation](#dataset-preparation)
  - [Configuration Setup](#configuration-setup)
  - [Training](#training)
  - [Monitoring & Debugging](#monitoring--debugging)
- [Common Issues](#common-issues)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## Installation

### Requirements
- Python 3.10
- CUDA 11.8+
- 24GB+ GPU memory (training) (more is better:))
- 20GB+ disk space (datasets can be loaded from the google cloud)

### Basic Setup
```bash
# Create conda environment
conda create -n flower python=3.10
conda activate flower

# Clone repository
git clone --recurse-submodules git@github.com:mbreuss/flower_vla.git
cd flower_vla

# Install requirements
pip install -r requirements_simpler.txt
```


## Pretraining Guide

First you need to chose a pretraining mix. 
Some datasets are not included in the google cloud storage and need to be loaded from the local storage instead. 
Below you will find guides for the most important datasets and how to download them:

### Dataset Preparation
#### Standard Datasets
Create a central dataset directory:
```bash
export DATA_DIR=~/tensorflow_datasets
```

#### Bridge Dataset

This is the recommended bridge dataset from Berkley, that is not part of OXE.

```bash
wget -r -np -nd -A '*' \
  https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/ \
  -P $DATA_DIR/bridge_dataset
```

#### BiPlay Dataset

BiPlay is a diverse bimanual aloha dataset from [project page](https://www.oiermees.com/publication/dit_policy/).

```bash
git lfs install
git clone https://huggingface.co/datasets/oier-mees/BiPlay \
  $DATA_DIR/aloha_play_dataset
```


### Configuration Setup

FLOWER uses huggingface accelerate library for efficient multi-GPU training. 
If you run it locally on a multi GPU system you can config the training config using the following answers:

#### Accelerate Configuration
```bash
accelerate config
```
Example settings for 2-GPU training:
```
This machine
multi-GPU
1  # Number of machines
NO  # fp16
NO  # bf16
NO  # Gradient accumulation
NO  # Gradient clipping
NO  # CPU offload
2   # Number of GPUs
0,1 # GPU indices
yes # Use DDP
bf16 # Mixed precision type
```

For training on a slurm cluster we provide an example script used for pretraining FLOWER on 4 H100 GPUs. 
Note, that it is important to have a main process port for being able to download the required datasets from the google cloud.


#### Training Configuration
Modify `conf/training.yaml`:
```yaml
# Basic Training Settings
batch_size: 512  # Total higher is better
gradient_accumulation_steps: 4 # recommended to use for llimited GPU memory settings to achieve larger batch sizes
max_train_steps: 500000
eval_every_n_steps: 10000  # does a short validation loss prediction for sanity checking NOTE: the validation loss does not correlate with the evaluation success rate and it is normal that it stagnates after some time. The model is still getting better. 
max_eval_steps: 100 # how many batches to use for validation loss

# Dataset Configuration
DATA_NAME: "trinity"  # datamix yo want to use 
DATA_PATH: "~/tensorflow_datasets"

# Optimization Settings
learning_rate_dit: 1e-4 # we use seuperate lr for the Flow Transformer and VLM to achieve the best results
learning_rate_vlm: 1e-5 # lower lr for VLM is crucial while the higher one for the flow helps too
weight_decay: 0.1 # high weight decay for the flow part and low one for the VLM part

# Hardware Settings
num_workers: 8  # Adjust based on CPU cores
pin_memory: true 
```

### Training
#### Single Node Training
```bash
accelerate launch flower/training.py
```

Continue from checkpoint:
```bash
accelerate launch flower/training.py \
  +step=100 \
  +continue_training=/path/to/checkpoint_100
```

#### Multi-Node Training
```bash
# Node 1 (Master)
accelerate launch --multi_gpu --num_processes=2 \
  --main_process_ip="MASTER_IP" \
  --main_process_port=29500 \
  --num_machines=2 \
  --machine_rank=0 \
  flower/training.py

# Node 2
accelerate launch --multi_gpu --num_processes=2 \
  --main_process_ip="MASTER_IP" \
  --main_process_port=29500 \
  --num_machines=2 \
  --machine_rank=1 \
  flower/training.py
```


#### Enhanced Debugging

Tensorflow is a bit annoying to debug when adding new datasets and transforms. 
Therefore use the debug_transforms.py script to get proper error messages.

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python flower/test_dataloader.py
python flower/debug_transforms.py
```


## Advanced Usage

You can create custom dataset mixes for pretraining and finetuning. The code for the oxe dataset is based on the code from [Octo](https://github.com/octo-models/octo) and [OpenVLA](https://github.com/openvla/openvla). 

### Custom Dataset Mixes
Modify `flower_vla/dataset/oxe/mixes.py`:
```python
CUSTOM_MIX = [
    ("bridge_dataset", 4.0),
    ("fractal20220817_data", 2.0),
    ("eef_droid", 0.2),
]
```

#### Adding a new dataset

You need to handle several things to integrate a new dataset into the code:

1. Define a datset config in `flower_vla/dataset/oxe/configs.py`
2. Define a transform for it in `flower_vla/dataset/oxe/transforms.py`
3. Add the value for the frequency to `flower_vla/dataset/utils/frequency_mapping.py`
4. Add it to the dataset index `flower_vla/dataset/utils/dataset_index.py`
5. Add the desired action chunk length to `flower_vla/dataset/utils/act_seq_mapping.py`

Now you should be good to go. If you still encounter issues use the `debug_transforms.py` script for testing.
Otherwise feel free to raise an issue or write me an email.


## Citation

If you found the code useful, please cite our work:

```bibtex
@inproceedings{
  reuss2025flower,
  title={{FLOWER}: Democratizing Generalist Robot Policies with Efficient Vision-Language-Flow Models},
  author={Moritz Reuss and Hongyi Zhou and Marcel R{\"u}hle and {\"O}mer Erdin{\c{c}} Ya{\u{g}}murlu and Fabian Otto and Rudolf Lioutikov},
  booktitle={9th Annual Conference on Robot Learning},
  year={2025},
  url={https://openreview.net/forum?id=JeppaebLRD}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work is only possible because of the code from the following open-source projects and datasets:

- Octo team: [octo](https://github.com/octo-models/octo)
- OpenVLA team [openvla](https://github.com/openvla/openvla)
- [Starcycle](https://github.com/StarCycle) for his mimictest codebase: [mimictest](https://github.com/EDiRobotics/mimictest)


---
