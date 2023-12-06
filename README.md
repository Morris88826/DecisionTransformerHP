# Enhancing Offline Learning Models with Human Preferences

Link to the paper: https://www.overleaf.com/project/6514bfd3fe326dbd8582e800
Link to the video: 

## Overview
Here is the overview of the architecture of our Decision Transformer with the Human Preference model (DTHP).
![Model Overview](https://github.com/Morris88826/DecisionTransformerHP/assets/32810188/4e92f7ff-8f52-41e6-82aa-4942cd21c4ef)

The main idea of our model is based on the [Decision Transformer](https://arxiv.org/pdf/2106.01345.pdf) while integrating the idea of having human preference embeddings to address the biases associated with determining the return-to-go, a notable issue in the context of the Decision Transformer. 

We use the [Preference Transformer](https://arxiv.org/pdf/2303.00957.pdf) for analyzing the past trajectory and generate the human preference score that will be further put into our Human Preference Integration Layer that combines it with the return-to-go. Here is the structure of our Human Preference Integration Layer.

<p align="center">
    <img src="https://github.com/Morris88826/DecisionTransformerHP/assets/32810188/bdeca022-968c-47ac-846b-f72f1bd6e159"  width="30%">
</p>

We show that when trained on D4RL benchmarks with suboptimal rewards-to-go, our
model outperforms the vanilla Decision Transformer on both the hopper-medium-
expert and the walker2d-medium-expert dataset

## Get Started

### Download mujoco
1. Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
2. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.
3. Add the following variables to ~/.bashrc
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/morris88826/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
4. source ~/.bashrc
#### Side Note
* If want to render using env.render()
```
export LD_PRELOAD=/home/morris88826/anaconda3/envs/trajectory/lib/libGLEW.so:/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
* Need to unset it if using MjRenderContextOffscreen from mujoco_py
```
unset LD_PRELOAD
```

### Setup the environment
```
conda env create -f environment.yml
conda activate dthp
pip install "cython<3"

cd ./decision-transformer-hp/trajectory-transformer
pip install e .

cd ..
pip install -r requirements.txt
```

### Download
#### Dataset
- D4RL
```
cd ./decision-transformer-hp/data
python download_d4rl_datasets.py
```
- D4RL with human preferences: [here]()

#### Model Weights
- DTHP: TODO

## Usage

### Train
```
cd decision-transformer-hp
python experiment.py
```

### Inference


### Visualization
Here we also provide the code for showing the sampled trajectory from inference time.

```
cd visualize
```

1. Replay in the renderer
- Make sure that the following is being set
```
echo $LD_PRELOAD
/home/morris88826/anaconda3/envs/trajectory/lib/libGLEW.so:/usr/lib/x86_64-linux-gnu/libstdc++.so.6

ex:
python replay.py --dataset hopper-medium-expert-v2 --trajectory ./replays/DT+PT/hopper-medium-expert-v2/10/best_traj_36000.pkl
```

2. Save the videos
- Make sure you unset LD_PRELOAD
```
unset LD_PRELOAD

ex:
python save_replay.py --dataset hopper-medium-expert-v2 --trajectory ./replays/DT+PT/hopper-medium-expert-v2/10/best_traj_36000.pkl
```
The result will be saved in the demo folder.

## Acknowledgements
Our backbone implementation is from
- [Decision Transformer](https://github.com/kzl/decision-transformer)
- [Preference Transformer](https://github.com/csmile-1006/PreferenceTransformer)
