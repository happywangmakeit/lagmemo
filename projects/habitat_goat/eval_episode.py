import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
from tqdm import tqdm

# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)

from config_utils import get_config
from habitat.core.env import Env

from home_robot.agent.goat_agent.goat_agent import GoatAgent
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot_sim.env.habitat_goat_env.habitat_goat_env import HabitatGoatEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="goat/modular_goat_hm3d.yaml",
        # default="objectnav/modular_objectnav_hm3d.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_goat/configs/agent/hm3d_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    print("Arguments:")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

    config = get_config(args.habitat_config_path, args.baseline_config_path)
    # config['habitat']['dataset']['split'] = 'train'
    # config['habitat']['dataset']['data_path'] = 'data/datasets/goat/hm3d/v0.2/train/train.json.gz'
    config['habitat']['dataset']['data_path'] = 'data/datasets/goat/hm3d/val_seen/val_seen.json.gz'
    print("可以在pdb里调整config，不需要的话直接continue即可")
    import ipdb;ipdb.set_trace()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1

    agent = GoatAgent(config=config)
    habitat_env = Env(config)
    env = HabitatGoatEnv(habitat_env, config=config)

    results_dir = os.path.join(config.DUMP_LOCATION, "results", config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)

    metrics = {}
    # success_data = {}

    for i in range(len(env.habitat_env.episodes)):
        env.reset()
        agent.reset()
        # t记录是在一整个episode中的当前序数
        t = 0

        scene_id = env.habitat_env.current_episode.scene_id.split("/")[-1].split(".")[0]
        # if scene_id not in success_data.keys():
        #     success_data[scene_id] = {'description':(0,0),'image':(0,0),'object':(0,0)}
        episode = env.habitat_env.current_episode
        episode_id = episode.episode_id
        agent.planner.set_vis_dir(scene_id, f"{episode_id}_{agent.current_task_idx}")
        agent.imagenav_visualizer.set_vis_dir(
            f"{scene_id}_{episode_id}_{agent.current_task_idx}"
        )
        agent.imagenav_obs_preprocessor.matching.set_vis_dir(
            f"{scene_id}_{episode_id}_{agent.current_task_idx}"
        )
        env.visualizer.set_vis_dir(scene_id, f"{episode_id}_{agent.current_task_idx}")

        all_subtask_metrics = []
        pbar = tqdm(total=config.AGENT.max_steps)
        subtask_num = 0
        while not env.episode_over:
            t  += 1
            obs = env.get_observation()
            if t == 1:
                obs_tasks = []
                for task in obs.task_observations["tasks"]:
                    obs_task = {}
                    for key, value in task.items():
                        if key == "image":
                            continue
                        obs_task[key] = value
                    obs_tasks.append(obs_task)

                pprint(obs_tasks)
            # print("想知道前面在print什么东西，obs是不是已经什么都有了")
            # zht 20250210 输出 instance map
            # from PIL import Image
            # instance_map = obs.task_observations["instance_map"]
            # rgb_image = np.zeros((instance_map.shape[0], instance_map.shape[1], 3), dtype=np.uint8)
            # instance_int = np.unique(instance_map)
            # for value in instance_int:
            #     np.random.seed(value)
            #     color = tuple(np.random.randint(0, 256, 3, dtype=np.uint8))
            #     rgb_image[instance_map==value]=color
            # Image.fromarray(rgb_image).save(f"/home/zht/git_repo/workspaces/home-robot/zht/instance_map{t-1}.png")
            # planner_snap_shot自非初始化后，在act中输出  image goal的可视化也在act里输出
            # import ipdb; ipdb.set_trace()
            action, info = agent.act(obs)
            # snap_shot - 非image goal的输出在apply_action的visualize里输出
            env.apply_action(action, info=info)
            pbar.set_description(
                f"Action: {str(action).split('.')[-1]} (sub-task: {agent.current_task_idx})"
            )
            pbar.update(1)

            if action == DiscreteNavigationAction.STOP:
                print(f"这里subtask{subtask_num}结束了")
                subtask_num += 1
                ep_metrics = env.get_episode_metrics()
                ep_metrics.pop("goat_top_down_map", None)
                print(ep_metrics)

                all_subtask_metrics.append(ep_metrics)
                if not env.episode_over:
                    agent.imagenav_visualizer.set_vis_dir(
                        f"{scene_id}_{episode_id}_{agent.current_task_idx}"
                    )
                    agent.imagenav_obs_preprocessor.matching.set_vis_dir(
                        f"{scene_id}_{episode_id}_{agent.current_task_idx}"
                    )
                    agent.planner.set_vis_dir(
                        scene_id, f"{episode_id}_{agent.current_task_idx}"
                    )
                    env.visualizer.set_vis_dir(
                        scene_id, f"{episode_id}_{agent.current_task_idx}"
                    )
                    pbar.reset()

        print(f"这里episode{i+1}开始了")

        pbar.close()

        ep_metrics = env.get_episode_metrics()
        scene_ep_id = f"{scene_id}_{episode_id}"
        metrics[scene_ep_id] = {"metrics": all_subtask_metrics}
        metrics[scene_ep_id]["total_num_steps"] = t
        metrics[scene_ep_id]["sub_task_timesteps"] = agent.sub_task_timesteps[0]
        metrics[scene_ep_id]["tasks"] = obs_tasks

        try:
            for metric in list(metrics.values())[0]["metrics"][0].keys():
                metrics[scene_ep_id][f"{metric}_mean"] = np.round(
                    np.nanmean(
                        np.array([y[metric] for y in metrics[scene_ep_id]["metrics"]])
                    ),
                    4,
                )
                metrics[scene_ep_id][f"{metric}_median"] = np.round(
                    np.nanmedian(
                        np.array([y[metric] for y in metrics[scene_ep_id]["metrics"]])
                    ),
                    4,
                )
        except Exception as e:
            print(e)
            import pdb

            pdb.set_trace()

        print("---------------------------------")

        with open(os.path.join(results_dir, "per_episode_metrics.json"), "w") as fp:
            json.dump(metrics, fp, indent=4)

        stats = {}

        for metric in list(metrics.values())[0]["metrics"][0].keys():
            stats[f"{metric}_mean"] = np.round(
                np.nanmean(
                    np.array(
                        [
                            y[metric]
                            for scene_ep_id in metrics.keys()
                            for y in metrics[scene_ep_id]["metrics"]
                        ]
                    )
                ),
                4,
            )
            stats[f"{metric}_median"] = np.round(
                np.nanmedian(
                    np.array(
                        [
                            y[metric]
                            for scene_ep_id in metrics.keys()
                            for y in metrics[scene_ep_id]["metrics"]
                        ]
                    )
                ),
                4,
            )

        with open(os.path.join(results_dir, "cumulative_metrics.json"), "w") as fp:
            json.dump(stats, fp, indent=4)
