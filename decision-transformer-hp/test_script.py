import subprocess 
env_list = ["hopper-medium-expert-v2", "hopper-medium-replay-v2","walker2d-medium-replay-v2", "walker2d-medium-expert-v2", "antmaze-medium-play-v2",  "hammer-human-v1", "hammer-cloned-v1"]
# env_list = ["walker2d-medium-replay-v2", "walker2d-medium-expert-v2", "antmaze-medium-play-v2",  "hammer-human-v1", "hammer-cloned-v1"]
# env_list = ["hopper-medium-replay-v2"]
for env in env_list:
    subprocess.run([
    "python", "-m", "experiment",
    "--env", f"{env}", 
    "--num_eval_episodes", "50", #[DEBUG]
    "--max_iters", "10", #[DEBUG]
    # "--log_to_wandb",
    "--embed_hf", 
    "--hf_model_path", f"pref_model/model_{env}.pkl", 
    "--from_d4rl", 
    # "--replay", 
    # "--save_model",
    # "--inference_only",
    # "--dt_model_path", f"ckpt/{env}/model_10.pth"
])
