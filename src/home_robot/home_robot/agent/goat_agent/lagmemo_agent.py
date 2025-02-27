# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import clip
import numpy as np
import scipy
import torch
from sklearn.cluster import DBSCAN
from torch.nn import DataParallel
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)

import home_robot.utils.pose as pu
from home_robot.agent.goat_agent.utils.agent_utils import get_matches_against_memory
from home_robot.agent.imagenav_agent.obs_preprocessor_img import ObsPreprocessor
from home_robot.agent.imagenav_agent.visualizer import NavVisualizer
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_state_lagmemo import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.instance_module_3d import GSMemory
from home_robot.navigation_planner.discrete_planner import DiscretePlanner

from .lagmemo_agent_module import LagMemoAgentModule

# For visualizing exploration issues
debug_frontier_map = False

# TODO 4 删没用的语句，包括image obs里的多余的都删。
class LagMemoAgent(Agent):
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

        self.instance_memory = None
        self.using_memory = getattr(
            config.AGENT.SEMANTIC_MAP, "using_memory", False
        )
        
        if self.using_memory:
            self.instance_memory = GSMemory(
                self.num_environments,
                config.AGENT.SEMANTIC_MAP.du_scale,
                debug_visualize=config.PRINT_IMAGES,
                config=config,
            )

        self._module = LagMemoAgentModule(config, instance_memory=self.instance_memory)

        if config.NO_GPU:
            self.device = torch.device("cpu")
            self.module = self._module
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
            self._module = self._module.to(self.device)
            # Use DataParallel only as a wrapper to move model inputs to GPU
            self.module = DataParallel(self._module, device_ids=[self.device_id])

        self.naive_landmark_conditioned = (
            config.AGENT.PLANNER.naive_landmark_conditioned
        )
        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            using_memory=getattr(
                config.AGENT.SEMANTIC_MAP, "using_memory", False
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
        self.landmark_found = False
        self.seen_landmarks = []
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0

        self.imagenav_obs_preprocessor = ObsPreprocessor(config, self.device)
        self.imagenav_visualizer = NavVisualizer(
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
        )
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
        self.goal_filtering = config.AGENT.SEMANTIC_MAP.goal_filtering
        # generate clip embeddings by loading clip model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device)
        n_px = self.clip_model.visual.input_resolution
        self.clip_image_preprocess = Compose(
            [
                Resize(n_px, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(n_px),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.prev_task_type = None
        self.STAGE = "Recall"  # 总共有 Recall Explore Memorize Clean 三个阶段

    # ------------------------------------------------------------------
    # Inference methods to interact with vectorized simulation
    # environments
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        object_goal_category: torch.Tensor = None,
        camera_pose: torch.Tensor = None,
        reject_visited_targets: bool = False,
        blacklist_target: bool = False,
        matches=None,
        confidence=None,
        all_matches=None,
        all_confidences=None,
    ) -> Tuple[List[dict], List[dict]]:
        """Prepare low-level planner inputs from an observation - this is
        the main inference function of the agent that lets it interact with
        vectorized environments.

        This function assumes that the agent has been initialized.

        Args:
            obs: current frame containing (RGB, depth, segmentation) of shape
             (num_environments, 3 + 1 + num_sem_categories, frame_height, frame_width)
            pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
             of shape (num_environments, 3)
            object_goal_category: semantic category of small object goals
            camera_pose: camera extrinsic pose of shape (num_environments, 4, 4)

        Returns:
            planner_inputs: list of num_environments planner inputs dicts containing
                obstacle_map: (M, M) binary np.ndarray local obstacle map
                 prediction
                sensor_pose: (7,) np.ndarray denoting global pose (x, y, o)
                 and local map boundaries planning window (gx1, gx2, gy1, gy2)
                goal_map: (M, M) binary np.ndarray denoting goal location
            vis_inputs: list of num_environments visualization info dicts containing
                explored_map: (M, M) binary np.ndarray local explored map
                 prediction
                semantic_map: (M, M) np.ndarray containing local semantic map
                 predictions
        """
        dones = torch.tensor([False] * self.num_environments)
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0
                for e in range(self.num_environments)
            ]
        )

        if object_goal_category is not None:
            object_goal_category = object_goal_category.unsqueeze(1)
        (
            self.goal_map,
            self.found_goal,
            frontier_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose,
            self.found_goal,
            self.goal_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
            seq_object_goal_category=object_goal_category,
            reject_visited_targets=reject_visited_targets,
            blacklist_target=blacklist_target,
            matches=matches,
            confidence=confidence,
            all_matches=all_matches,
            all_confidences=all_confidences,
        )
        # if self.total_timesteps[0] == 60:
        #     import pdb
        #     pdb.set_trace()
        #     print("20250226，每10帧观察最终的local map global map self.semantic_map.local_map")
        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        if matches is not None or confidence is not None:
            goal_map = self._prep_goal_map_input()
        else:
            goal_map = self.goal_map.squeeze(1).cpu().numpy()

        # found_goal = self.found_goal.squeeze(1).cpu()
        # 为了防止bug index out of range 增大config中max_num_sub_task_episodes
        # sub_task_timesteps现在是一个list [0]为一个长度为max_num_sub_task_episodes的list
        for e in range(self.num_environments):
            if frontier_map is not None:
                self.semantic_map.update_frontier_map(
                    e, frontier_map[e][0].cpu().numpy()
                )
            if self.found_goal[e] or self.timesteps_before_goal_update[e] == 0:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                if self.timesteps_before_goal_update[e] == 0:
                    self.timesteps_before_goal_update[e] = self.goal_update_steps
            self.total_timesteps[e] = self.total_timesteps[e] + 1
            self.sub_task_timesteps[e][self.current_task_idx] += 1
            self.timesteps_before_goal_update[e] = (
                self.timesteps_before_goal_update[e] - 1
            )

        # if debug_frontier_map:
        #     import matplotlib.pyplot as plt

        #     plt.subplot(131)
        #     plt.imshow(self.semantic_map.get_frontier_map(e))
        #     plt.subplot(132)
        #     plt.imshow(frontier_map[e][0])
        #     plt.subplot(133)
        #     plt.imshow(self.semantic_map.get_goal_map(e))
        #     plt.show()

        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "frontier_map": self.semantic_map.get_frontier_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": self.found_goal[e].item(),
            }
            for e in range(self.num_environments)
        ]
        if self.visualize:
            vis_inputs = [
                {
                    "explored_map": self.semantic_map.get_explored_map(e),
                    "semantic_map": self.semantic_map.get_semantic_map(e),
                    "been_close_map": self.semantic_map.get_been_close_map(e),
                    "blacklisted_targets_map": self.semantic_map.get_blacklisted_targets_map(
                        e
                    ),
                    "timestep": self.total_timesteps[e],
                }
                for e in range(self.num_environments)
            ]
            if self.using_memory:
                for e in range(self.num_environments):
                    vis_inputs[e]["instance_map"] = self.semantic_map.get_instance_map(
                        e
                    )
        else:
            vis_inputs = [{} for e in range(self.num_environments)]

        return planner_inputs, vis_inputs

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
        # zht 20250218 把record instance改为None，以防报错
        if self.instance_memory is not None:
            self.instance_memory.reset()
        self.landmark_found = False
        self.seen_landmarks = []
        self.reject_visited_targets = False
        self.blacklist_target = False
        self.current_task_idx = 0

        if self.imagenav_visualizer is not None:
            self.imagenav_visualizer.reset()
        self.imagenav_obs_preprocessor.reset()

        self.found_goal[:] = False
        self.goal_map[:] *= 0
        self.prev_task_type = None
        self.planner.reset()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.total_timesteps[e] = 0
        self.sub_task_timesteps[e] = [0] * self.max_num_sub_task_episodes
        self.timesteps_before_goal_update[e] = 0
        self.last_poses[e] = np.zeros(3)
        self.semantic_map.init_map_and_pose_for_env(e)
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.reached_goal_panorama_rotate_steps = self.panorama_rotate_steps
        self.instance_memory.reset_for_env(e)
        self.landmark_found = False
        self.seen_landmarks = []
        self.reject_visited_targets = False
        self.blacklist_target = False

        self.current_task_idx = 0
        self.planner.reset()

    # ---------------------------------------------------------------------
    # Inference methods to interact with the robot or a single un-vectorized
    # simulation environment
    # ---------------------------------------------------------------------

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.planner.reset()

    # def compare_img_goal_with_instances(self, img_goal):
    
    # zht 20250220 把clip_matching_fn移出来
    def clip_matching_fn(self, views, language_goal, **kwargs):
        batch_size = 64
        language_goal = language_goal.replace("Instruction: ", "")
        language_goal = clip.tokenize(language_goal).to(self.device)
        language_goal = self.clip_model.encode_text(language_goal)
        # get clip embedding for views with a batch size of batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if isinstance(views, list):
            views = np.stack(views, 0)
        if views.dtype == np.uint8:
            views = views.astype(np.float32) / 255
        views = torch.cat(
            [
                self.clip_model.encode_image(
                    self.clip_image_preprocess(v.permute(0, 3, 1, 2)).to(device)
                )
                for v in torch.tensor(views).split(batch_size)
            ],
            dim=0,
        )
        # compute similarity
        similarity = (language_goal @ views.T).softmax(dim=-1)
        return [[1]] * similarity.shape[0], np.expand_dims(
            similarity.detach().cpu().numpy(), 1
        )

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        current_task = obs.task_observations["tasks"][self.current_task_idx]
        task_type = current_task["type"]
        
        # 每次act都输出当前goal
        if task_type == "description":
            print(f"当前是language goal, language为 {current_task['instruction']} , 其中隐藏的target为 {current_task['target']}")
        elif task_type == "object":
            print(f"当前是object goal, object为 {current_task['target']}")
        elif task_type == "image":
            print(f"当前是image goal, image中的target为 {current_task['target']}")
        # 看了一下都有target字段和semantic_id字段，输出
        print("对应的target字段是：", current_task["target"], "对应的semantic_id字段是：", current_task["semantic_id"])
        
        
        # 干脆 stage - Recall ， 专门用于get_matches_against_memory
        if self.STAGE == "Recall":
            # 分模态处理
            if task_type == "object":
                all_matches = []
                all_confidences = []
            elif task_type == "description":
                # zht 20250210 这个地方也改  原本是languagenav， current_task没有instruction字段
                # zht 20250218 判断条件直接改为为一段subtask的开始
                # if self.prev_task_type != "description":
                # 20250218 wxl 修改为数据集中的instruction

                if self.using_memory:
                    all_matches, all_confidences = get_matches_against_memory(
                        self.instance_memory,
                        self.clip_matching_fn,
                        self.total_timesteps[0],
                        language_goal=current_task["instruction"],
                    )
                else:
                    all_matches = []
                    all_confidences = []
            elif task_type == "image":
                # 得给imagenav做准备
                if self.imagenav_obs_preprocessor.goal_image is None:
                    img_goal = obs.task_observations["tasks"][self.current_task_idx]["image"]
                    (
                        self.imagenav_obs_preprocessor.goal_image,
                        self.imagenav_obs_preprocessor.goal_image_keypoints,
                    ) = self.imagenav_obs_preprocessor.matching.get_goal_image_keypoints(img_goal)
                    self.imagenav_obs_preprocessor.goal_mask, _ = self.imagenav_obs_preprocessor.instance_seg.get_goal_mask(img_goal)

                if self.using_memory:
                    all_matches, all_confidences = get_matches_against_memory(
                        self.instance_memory,
                        self.imagenav_obs_preprocessor.matching,
                        self.total_timesteps[0],
                        image_goal=self.imagenav_obs_preprocessor.goal_image,
                        goal_image_keypoints=self.imagenav_obs_preprocessor.goal_image_keypoints,
                    )
                else:
                    all_matches = []
                    all_confidences = []
            else:
                print("无此模态goal")
                raise NotImplementedError
            self.STAGE =  "Explore"

        # 非回忆阶段，这两个值都是空
        elif self.STAGE == "Explore":
            if task_type == "object":
                all_matches = []
                all_confidences = []
            elif task_type == "description":
                all_matches = []
                all_confidences = []
            elif task_type =="image":
                all_matches = []
                all_confidences = []
            else:
                print("无此模态goal")
                raise NotImplementedError

        # 1 - Obs preprocessing
        if task_type == "image":
            self.imagenav_obs_preprocessor.current_task_idx = self.current_task_idx
            (
                obs_preprocessed,
                img_goal,
                pose_delta,
                camera_pose,
                matches,
                confidence
            ) = self.imagenav_obs_preprocessor.preprocess(
                obs, last_pose=self.last_poses[0], instance_memory=self.instance_memory
            )
            object_goal_category = current_task["semantic_id"]
            object_goal_category = torch.tensor(object_goal_category).unsqueeze(0)
            planner_inputs, vis_inputs = self.prepare_planner_inputs(
                obs_preprocessed,
                pose_delta,
                object_goal_category=object_goal_category,
                matches=matches,
                confidence=confidence,
                camera_pose=camera_pose,
                all_matches=all_matches,
                all_confidences=all_confidences,
            )
            self.last_poses[0] = self.imagenav_obs_preprocessor.last_pose
        elif task_type in ["object", "description"]:
            (
                obs_preprocessed,
                pose_delta,
                object_goal_category,
                landmarks,
                camera_pose,
                matches,
                confidence,
                all_matches,
                all_confidences,
            ) = self._preprocess_obs(obs, task_type, all_matches, all_confidences)
            # 2 - Semantic mapping + policy
            planner_inputs, vis_inputs = self.prepare_planner_inputs(
                obs_preprocessed,
                pose_delta,
                object_goal_category=object_goal_category,
                camera_pose=camera_pose,
                reject_visited_targets=self.reject_visited_targets,
                blacklist_target=self.blacklist_target,
                matches=matches,
                confidence=confidence,
                all_matches=all_matches,
                all_confidences=all_confidences,
            )

        # 3 - Planning
        closest_goal_map = None
        dilated_obstacle_map = None
        if planner_inputs[0]["found_goal"]:
            # import pdb
            # pdb.set_trace()
            # print("看一下何时这个变量清零")
            self.episode_panorama_start_steps = 0
        if self.total_timesteps[0] < self.episode_panorama_start_steps:
            action = DiscreteNavigationAction.TURN_RIGHT
        elif self.sub_task_timesteps[0][self.current_task_idx] >= self.max_steps:
            action = DiscreteNavigationAction.STOP
        else:
            (
                action,
                closest_goal_map,
                short_term_goal,
                dilated_obstacle_map,
            ) = self.planner.plan(
                **planner_inputs[0],
                use_dilation_for_stg=self.use_dilation_for_stg,
                timestep=self.sub_task_timesteps[0][self.current_task_idx],
            )

            # if self.reached_goal_candidate:
            #     # move to next sub-task
            #     # update semantic map
            #     # reset timesteps
            #     pass

        if self.visualize:
            vis_inputs[0]["dilated_obstacle_map"] = dilated_obstacle_map
            # zht 2025.2.9 把imagenav改成image了
            if task_type == "image":
                collision = {"is_collision": False}
                info = {
                    **planner_inputs[0],
                    **vis_inputs[0],
                    "semantic_frame": obs.rgb,
                    "closest_goal_map": closest_goal_map,
                    "last_goal_image": obs.task_observations["tasks"][
                        self.current_task_idx
                    ]["image"],
                    "last_collisions": collision,
                    "last_td_map": obs.task_observations.get("top_down_map"),
                    "curr_action": action,
                }
                
                # 摘到image goal图片, wxl, 2025.2.25
                # from PIL import Image
                # goal_img = info['last_goal_image']
                # img = np.array(goal_img,dtype=np.uint8)
                # img = Image.fromarray(img)
                # path = obs.task_observations["tasks"][self.current_task_idx]
                # import ipdb;ipdb.set_trace()
                self.imagenav_visualizer.visualize(**info)
            else:
                
                # start
                # 用imagenav的visualizer 可视化object、language nav，2025.2.24，wxl
                collision = {"is_collision": False}
                goal_text_desc = {
                    x: y
                    for x, y in obs.task_observations["tasks"][
                        self.current_task_idx
                    ].items()
                    if x != "image"
                }
                info = {
                    **planner_inputs[0], 
                    **vis_inputs[0],
                    "semantic_frame": obs.rgb,
                    "closest_goal_map": closest_goal_map,
                    # "last_goal_image": obs.task_observations["tasks"][
                    #     self.current_task_idx
                    # ]["image"],
                    "last_collisions": collision,
                    "last_td_map": obs.task_observations.get("top_down_map"),
                    "goal_info": goal_text_desc,
                    "curr_action": action,
                }
                # import ipdb; ipdb.set_trace()
                self.imagenav_visualizer.visualize(**info)
                # end
                
                goal_text_desc = {
                    x: y
                    for x, y in obs.task_observations["tasks"][
                        self.current_task_idx
                    ].items()
                    if x != "image"
                }
                vis_inputs[0]["goal_name"] = goal_text_desc
                vis_inputs[0]["semantic_frame"] = obs.task_observations[
                    "semantic_frame"
                ]
                vis_inputs[0]["closest_goal_map"] = closest_goal_map
                vis_inputs[0]["third_person_image"] = obs.third_person_image
                vis_inputs[0]["short_term_goal"] = None
                vis_inputs[0]["instance_memory"] = self.instance_memory

                info = {**planner_inputs[0], **vis_inputs[0]}

        if action == DiscreteNavigationAction.STOP:
            self.STAGE = "Memorize"
            print("往memory中加入语义")
            if len(obs.task_observations["tasks"]) - 1 > self.current_task_idx:
                self.current_task_idx += 1
                self.timesteps_before_goal_update[0] = 0
                self.total_timesteps = [0] * self.num_environments
                self.found_goal = torch.zeros(
                    self.num_environments, 1, dtype=bool, device=self.device
                )
                self.imagenav_obs_preprocessor.reset_sub_episode()
            print("完成memory构建后，下一个任务进行索引")
            self.STAGE = "Recall"
        self.prev_task_type = task_type
        return action, info

    def _preprocess_obs(self, obs: Observations, task_type: str, all_matches=[], all_confidences=[]):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm

        current_task = obs.task_observations["tasks"][self.current_task_idx]
        current_goal_semantic_id = current_task["semantic_id"]

        # if self.store_all_categories_in_map:
        semantic = obs.semantic
        # else:
        #     semantic = np.full_like(obs.semantic, 4)
        #     semantic[
        #         obs.semantic == current_goal_semantic_id
        #     ] = current_goal_semantic_id

        semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(self.device)]
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

        if self.using_memory:
            instances = obs.task_observations["instance_map"]
            # first create a mapping to 1, 2, ... num_instances
            instance_ids = np.unique(instances)
            # map instance id to index
            instance_id_to_idx = {
                instance_id: idx for idx, instance_id in enumerate(instance_ids)
            }
            # convert instance ids to indices, use vectorized lookup
            instances = torch.from_numpy(
                np.vectorize(instance_id_to_idx.get)(instances)
            ).to(self.device)
            # create a one-hot encoding
            instances = torch.eye(len(instance_ids), device=self.device)[instances]
            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])
        ).unsqueeze(0)
        self.last_poses[0] = curr_pose

        # goals = obs.task_observations["tasks"]

        object_goal_category = torch.tensor(current_goal_semantic_id).unsqueeze(0)
        if "landmarks" in current_task.keys():
            landmarks = current_task["landmarks"]
        else:
            landmarks = []

        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)

        matches, confidences, all_matches, all_confidences = None, None, all_matches, all_confidences
        # languagenav 并非第一帧时要match当前rgb和language
        if (task_type == "description") & (self.total_timesteps[0] != 0):
            matches, confidences = self.clip_matching_fn(
                np.expand_dims(obs.rgb, 0), current_task["instruction"]
            )
            matches = matches[0]
            confidences = confidences[0]

        return (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            landmarks,
            # goals,
            camera_pose,
            matches,
            confidences,
            all_matches,
            all_confidences,
        )

    def _prep_goal_map_input(self) -> None:
        """
        Perform optional clustering of the goal channel to mitigate noisy projection
        splatter.
        """
        goal_map = self.goal_map.squeeze(1).cpu().numpy()

        if not self.goal_filtering:
            return goal_map

        for e in range(goal_map.shape[0]):
            if not self.found_goal[e]:
                continue

            # cluster goal points
            c = DBSCAN(eps=4, min_samples=1)
            data = np.array(goal_map[e].nonzero()).T
            # 以下语句在_5 第一个subtask 报错ValueError: Found array with 0 sample(s) (shape=(0, 2)) while a minimum of 1 is required by DBSCAN.
            # import pdb
            # pdb.set_trace()
            # print("解决ValueError")
            #zht 20250211 不知道直接返回这个全是0的600*600的goal map是否会报错
            # wxl 20250226 可能continue会更好一些？对应上面的if not self.found_goal(e)
            if data.size == 0:
                return goal_map 
            c.fit(data)

            # mask all points not in the largest cluster
            mode = scipy.stats.mode(c.labels_, keepdims=False).mode.item()
            mode_mask = (c.labels_ != mode).nonzero()
            x = data[mode_mask]
            goal_map_ = np.copy(goal_map[e])
            goal_map_[x] = 0.0

            # adopt masked map if non-empty
            if goal_map_.sum() > 0:
                goal_map[e] = goal_map_

        return goal_map
