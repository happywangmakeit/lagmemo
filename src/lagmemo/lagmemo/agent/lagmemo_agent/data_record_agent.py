# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import scipy
import torch
from sklearn.cluster import DBSCAN
from torch.nn import DataParallel

import lagmemo.utils.pose as pu

from lagmemo.agent.imagenav_agent.visualizer import NavVisualizer
from lagmemo.core.abstract_agent import Agent
from lagmemo.core.interfaces import DiscreteNavigationAction, Observations
from lagmemo.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from lagmemo.mapping.semantic.constants import MapConstants as MC
from lagmemo.mapping.semantic.instance_tracking_modules import InstanceMemory
from lagmemo.perception.detection.maskrcnn.coco_categories import coco_categories

from .lagmemo_agent_module import GoatAgentModule
from .lagmemo_matching import GoatMatching

# For visualizing exploration issues
debug_frontier_map = False
# from lagmemo.vlfm_vis.habitat_visualizer import HabitatVis
# hab_vis = HabitatVis()

class DataRecordAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(self, config, device_id: int = 0):
        self.max_steps = config.AGENT.max_steps
        self.num_environments = config.NUM_ENVIRONMENTS
        self.store_all_categories_in_map = getattr(
            config.AGENT, "store_all_categories", False
        )
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0

        self.panorama_rotate_steps = int(360 / config.ENVIRONMENT.turn_angle)

        self.goal_matching_vis_dir = f"{config.DUMP_LOCATION}/goal_grounding_vis"
        Path(self.goal_matching_vis_dir).mkdir(parents=True, exist_ok=True)

        self.instance_memory = None
        self.record_instance_ids = getattr(
            config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
        )

        if self.record_instance_ids:
            self.instance_memory = InstanceMemory(
                self.num_environments,
                config.AGENT.SEMANTIC_MAP.du_scale,
                debug_visualize=config.PRINT_IMAGES,
                config=config,
                mask_cropped_instances=False,
                padding_cropped_instances=200
            )

        ## imagenav stuff
        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.goal_policy_config = config.AGENT.SUPERGLUE

        # self.instance_seg = Detic(config.AGENT.DETIC)
        self.matching = GoatMatching(
            device=0,  # config.simulator_gpu_id
            score_func=self.goal_policy_config.score_function,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=config.PRINT_IMAGES,
            instance_memory=self.instance_memory,
        )

        if self.goal_policy_config.batching:
            self.image_matching_function = self.matching.match_image_batch_to_image
        else:
            self.image_matching_function = self.matching.match_image_to_image

        self._module = GoatAgentModule(
            config, matching=self.matching, instance_memory=self.instance_memory
        )

        if config.NO_GPU:
            self.device = torch.device("cpu")
            self.module = self._module
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
            self._module = self._module.to(self.device)
            # Use DataParallel only as a wrapper to move model inputs to GPU
            self.module = DataParallel(self._module, device_ids=[self.device_id])

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            instance_memory=self.instance_memory,
        )
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.max_num_sub_task_episodes = config.ENVIRONMENT.max_num_sub_task_episodes

        if (
            "planner_type" in config.AGENT.PLANNER
            and config.AGENT.PLANNER.planner_type == "old"
        ):
            print("Using old planner")
            from lagmemo.navigation_planner.old_discrete_planner import (
                DiscretePlanner,
            )
        else:
            print("Using new planner")
            from lagmemo.navigation_planner.discrete_planner import DiscretePlanner

        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
        )
        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device
        )

        self.goal_update_steps = self._module.goal_update_steps
        self.sub_task_timesteps = None
        self.total_timesteps = None
        self.timesteps_before_goal_update = None
        self.episode_panorama_start_steps = None
        self.last_poses = None
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0

        self.imagenav_visualizer = NavVisualizer(
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
        )
        # self.imagenav_visualizer = None
        self.found_goal = torch.zeros(
            self.num_environments, 1, dtype=bool, device=self.device
        )
        self.goal_map = torch.zeros(
            self.num_environments,
            1,
            *self.semantic_map.local_map.shape[2:],
            dtype=self.semantic_map.local_map.dtype,
            device=self.device,
        )
        self.goal_pose = None
        self.goal_filtering = config.AGENT.SEMANTIC_MAP.goal_filtering
        self.prev_task_type = None
        

    # ------------------------------------------------------------------
    # Inference methods to interact with vectorized simulation
    # environments
    # ------------------------------------------------------------------

    def reset_vectorized(self):
        """Initialize agent state."""
        self.total_timesteps = [0] * self.num_environments
        self.sub_task_timesteps = [
            [0] * self.max_num_sub_task_episodes
        ] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.semantic_map.init_map_and_pose()
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.current_task_idx = 0
        self.fully_explored = [False] * self.num_environments
        self.force_match_against_memory = False

        if self.imagenav_visualizer is not None:
            self.imagenav_visualizer.reset()

        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None

        self.found_goal[:] = False
        self.goal_map[:] *= 0
        self.prev_task_type = None
        self.planner.reset()
        self._module.reset()

    def reset_sub_episode(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self._module.reset_sub_episode()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.total_timesteps[e] = 0
        self.sub_task_timesteps[e] = [0] * self.max_num_sub_task_episodes
        self.timesteps_before_goal_update[e] = 0
        self.last_poses[e] = np.zeros(3)
        self.semantic_map.init_map_and_pose_for_env(e)
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        if self.instance_memory is not None:
            self.instance_memory.reset_for_env(e)
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0
        self.planner.reset()
        self._module.reset()
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None

    # ---------------------------------------------------------------------
    # Inference methods to interact with the robot or a single un-vectorized
    # simulation environment
    # ---------------------------------------------------------------------

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.planner.reset()

        self.goal_image = None
        self.goal_mask = None
        self.goal_image_keypoints = None
        self.prev_position = None
        self.ctr = 0

    # from PIL import Image
    # def keep_img(img, path, type = ''):
    #     if type == 'depth':
    #         depth_map = np.array(img)
    #         depth_min = depth_map.min()
    #         depth_max = depth_map.max()

    #         # normalized_depth = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    #         # img = Image.fromarray(normalized_depth)
    #         img = Image.fromarray(depth_map.astype(np.uint8))
    #         img.save(path)
    #         return 

    #     img = np.array(img,dtype=np.uint8)
    #     img = Image.fromarray(img)
    #     img.save(path)
    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        task_type = current_task["type"]

        # t0 = time.time()
        # 1 - obs preprocessing
        import matplotlib.pyplot as plt
        import os
        output_vis_path = "/home/wxl/lagmemo/data_record/home_robot_obs/"
        output_data_path = "/home/wxl/lagmemo/data_record/RGB_depth/"
        rgb_path = output_data_path + "rgb/"
        dep_path = output_data_path + "depth/"
        pos_path = output_data_path
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)
        if not os.path.exists(dep_path):
            os.makedirs(dep_path)

        image_rgb = obs.rgb
        image_depth = obs.depth
        plt.imsave(output_data_path + "rgb_" + str(self.total_timesteps[0]) + ".png", image_rgb)
        max_depth = 5
        min_depth = 0.5
        # 10000代表需要调成0的值
        # 10001代表需要调成1的值
        # 部分depth出现10000的异常值，需要设为最近深度值
        image_depth[image_depth==10000] = min_depth 
        image_depth[image_depth==10001] = max_depth
        # 深度图处理成范围0-1
        normalized_depth = (image_depth-min_depth)/(max_depth-min_depth)
        normalized_depth[normalized_depth>1] = 1
        plt.imshow(normalized_depth, cmap='gray')
        plt.savefig(output_data_path + "depth_" + str(self.total_timesteps[0]) + ".png")

        # 2 - frontier map 不需要
        x, y = obs.gps
        robot_xy = np.array([x, y])
        print("agent position:", robot_xy)
        
        
        # 3 - value_map 暂空

        # 键盘控制action
        import sys
        def get_keyboard_input():
            """阻塞式等待用户输入并返回对应动作"""
            while True:
                try:
                    # 读取输入（需用户按回车）
                    user_input = input("请输入指令 [前进/左转/右转/停止] [w/a/d/q] + 回车: ").strip().lower()
                    if user_input == 'w':
                        return DiscreteNavigationAction.MOVE_FORWARD
                    elif user_input == 'a':
                        return DiscreteNavigationAction.TURN_LEFT
                    elif user_input == 'd':
                        return DiscreteNavigationAction.TURN_RIGHT
                    elif user_input == 'q':
                        return DiscreteNavigationAction.STOP
                    else:
                        print(f"无效输入: {user_input}，请重新输入！")
                except KeyboardInterrupt:
                    print("\n程序已终止。")
                    sys.exit(0)
                    
        action = get_keyboard_input()
        # import keyboard
        # def wait_for_key():
        #     while True:
        #         event = keyboard.read_event()
        #         if event.event_type == keyboard.KEY_DOWN:
        #             if keyboard.is_pressed('w'):
        #                 return DiscreteNavigationAction.MOVE_FORWARD
        #             elif keyboard.is_pressed('a'):
        #                 return DiscreteNavigationAction.TURN_LEFT
        #             elif keyboard.is_pressed('d'):
        #                 return DiscreteNavigationAction.TURN_RIGHT
        #             elif keyboard.is_pressed('q'):
        #                 return DiscreteNavigationAction.STOP
        # action = wait_for_key()
        # 6 - visualize
        # 这里的vis是在yaml文件中的输出
        goal_text_desc = {
                    x: y
                    for x, y in obs.task_observations["tasks"][
                        self.current_task_idx
                    ].items()
                    if x != "image"
                }
        info = {
            "goal_name": goal_text_desc,
            "semantic_frame": obs.task_observations["semantic_frame"],
            "timestep": self.total_timesteps[0],
            "found_goal": False,
        }
        self.total_timesteps[0] += 1
        # 7 - reset
        if action == DiscreteNavigationAction.STOP:
            if len(obs.task_observations["tasks"]) - 1 > self.current_task_idx:
                self.current_task_idx += 1
                self.total_timesteps = [0] * self.num_environments
                self.found_goal = torch.zeros(
                    self.num_environments, 1, dtype=bool, device=self.device
                )
                self.reset_sub_episode()
        self.prev_task_type = task_type
        return action, info

