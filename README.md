# Enhancing Offline Learning Models with Human Preferences

Link to the paper: [here](https://drive.google.com/file/d/1L4eVsseg-rjYQwhN0rhZrJsqf37uDtmY/view?usp=sharing)

Link to the video: [here](https://www.youtube.com/watch?v=J95IBloy2l4&feature=youtu.be)

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
expert and the walker2d-medium-expert datasets.

## Get Started

### Colab Notebook Demo
For a quick self-contained way to run our code, simply upload the Jupyter notebook (train_colab.ipynb) into Google Colab and run every cell. To use a different dataset, uncomment the appropriate code in the block right above the main training block, titled "Download dataset (subset) from HF Datasets". You will be prompted with Weights & Biases to enter your authorization key - simply follow the instructions provided and hit 'enter'.

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
- D4RL with human preferences: [here](https://drive.google.com/drive/folders/1Ep1xnN_32VqEYym1LSkvUvausyCZSgKj?usp=drive_link)
- Move the dataset into  ``` ./decision-transformer-hp/data ```

#### Model Weights
- DTHP: [here](https://drive.google.com/drive/folders/1iAuLOMRdWH_HY4zDMqGDx_EC8mrRypAt?usp=drive_link)

## Usage

### Train
```
cd decision-transformer-hp
python experiment.py --env {env_name} --embed_hf --hf_model_path {preference transformer weight path} --from_d4rl
```

### Inference
Set ```--replay``` flag to generate trajectories record for visualization
```
cd decision-transformer-hp
python experiment.py --env {env_name} --embed_hf --hf_model_path {preference transformer weight path} --from_d4rl --inference_only --replay 
```

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
- [Trajectory Transformer](https://github.com/JannerM/trajectory-transformer)
