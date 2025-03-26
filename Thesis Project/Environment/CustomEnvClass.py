
# Import necessary libraries

import numpy as np
import logging
import time
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box

from typing import Dict, Tuple, Optional

import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from ray import tune
from ray.tune.analysis import ExperimentAnalysis
from ray.rllib.connectors.env_to_module import MeanStdFilter
from ray.tune.schedulers import HyperBandScheduler
#import diffcp
#import cvxpy

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls


#import matplotlib.pyplot as plt
#import pandas as pd
#from pprint import pprint


# from enum import Enum

import pygame
from IPython.display import display, Image as IPImage           # Aliasing IPython's Image
from PIL import Image as PILImage                               # Aliasing Pillow's Image

# Importing env_helpers
import env_helpers

# Import everything manually
for name in dir(env_helpers):
    if not name.startswith("__"):  # Skip system names
        globals()[name] = getattr(env_helpers, name)


class ContinuousPathfindingEnv(MultiAgentEnv):
    """
    The ContinuousPathfindingEnv is a multi-agent reinforcement learning environment for continuous action
    pathfinding tasks, such as navigating agents through obstacles to reach predefined goals. 
    It takes env_config and render-mode as input arguments.

    Default Environment Configuration Setup:
    simulation_env_config = { 
    "agent_count": 3, "env_name": "Factory-1", "scale": 1, 
    "starting_point": False, "agent_size": (0.2, 0.2), "goal_size": (0.5, 0.5), 
    "timesteps_per_episode": 5000,"max_speed": 1.0, "goal_radius": 0.1, "dt": 0.05, "sensor_range_factor": 1.5,
    "num_sensor_rays": 45, "sensor_fov": np.pi*2, "obstacle_density": 0.1, "render_mode": None,
    }

    
    env_config = EnvContext(simulation_env_config, worker_index=0,)
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi", "state_pixels"],
        "render_fps": 4,
    }

    def __init__(self, env_config: EnvContext, render_mode: Optional[str] = None):
        #super().__init__()

        # Defines expected bounds or acceptable ranges for each configuration parameter
        config_bounds = {
            "agent_count": (2, 7),             # Min 1 agent, max 7 agents
            "scale": (1, 50), 
            "timesteps_per_episode": (1, 50000), # Reasonable bounds for episode length
            "max_speed": (0.0001, 1.0),          # Max speed between 0.0001 and 1.0
            "goal_radius": (0.01, 2.0),          # Radius from 0.01 to 2.0
            "dt": (0.0001, 1.0),                
            "sensor_range": (1, 10),             # Sensor range from 1 to 10 units
            "num_sensor_rays": (3, 360), 
            "sensor_fov": (np.pi/6, np.pi*2),  
            "obstacle_density": (0, 1.0),        # Density between 0 (no obstacles) and 1 (full area covered)
        }

        print(":"*100) 
        print("BEGINING DEBUG OUTPUT FOR INIT METHOD:<") 
        
        # Validate each parameter in env_config
        for key, value in env_config.items():
            if key in config_bounds:
                min_val, max_val = config_bounds[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Parameter '{key}' with value {value} is out of bounds. "
                        f"Expected range is [{min_val}, {max_val}]."
                    )
            else:
                print(f"Warning: '{key}' is not a recognized configuration parameter.")
        

        # Environment configurations and name initialization (Assign the validated config to the class attribute)
        self.env_config = env_config
        self.env_name = self.env_config.get("env_name", "Factory-1")
        
        # Number of agents and agent names
        self.num_agents = self.env_config.get("agent_count", 3)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Initialize environment state
        self.current_time_step = 0  # Initialize time step 
#        self.max_timesteps = env_config.get("max_timesteps", 100)
        
        # For accessing RLlib-specific metadata like worker index
#        self.worker_index = env_config.worker_index
#        self.vector_index = env_config.vector_index




        """
        print('-'*100)
        print("screen_width:", self.screen_width)
        print('-'*100)        
        print("screen_height:", self.screen_height)
        print('-'*100)        
            """

        # Initialize agent and goal sizes in world units
        self.agent_size = self.env_config.get("agent_size", (0.35, 0.35)) # Logical units
        self.goal_size = self.env_config.get("goal_size", (0.25, 0.25)) # Logical units

        # Initialize Scale (in pixels per unit) and margin
        self.agent_scale, self.agent_margin = compute_optimal_values(object_size=self.agent_size)
        self.goal_scale, self.goal_margin = compute_optimal_values(object_size=self.goal_size)
        self.margin = self.agent_margin
        self.scale = self.env_config.get("scale", 1)
        
        # Set up grid and agent start and goal positions based on environment config
        self.starting_point = self.env_config.get("starting_point", False)
        
        # Get start and goal positions
        if self.starting_point:
            self.start_positions = get_predefined_start_positions(env_name=self.env_name, num_agents=self.num_agents)
            self.goal_positions = get_predefined_goal_positions(env_name=self.env_name, num_agents=self.num_agents)
        else:
            self.start_positions = get_random_start_positions(env_name=self.env_name, 
                                                              num_agents=self.num_agents,
                                                              object_size=self.agent_size,
                                                              margin=self.agent_margin, 
                                                              scale=self.agent_scale)
            
            self.goal_positions = get_random_goal_positions(env_name=self.env_name, 
                                                            num_agents=self.num_agents, 
                                                            goal_size=self.goal_size, 
                                                            margin=self.goal_margin, 
                                                            scale=self.goal_scale
                                                            )


        # Initialize agent states (positions, goals, speeds, and orientations)
        self.agent_positions = {agent: np.array(self.start_positions[agent]) for agent in self.agents}
        self.agent_goals = {agent: np.array(self.goal_positions[agent]) for agent in self.agents}
        
        # Ensure goals are unique
        for agent, goal in self.agent_goals.items():
            for other_agent, other_goal in self.agent_goals.items():
                if agent != other_agent and (np.all(goal == other_goal)):  # Compare arrays element-wise
                    print(f"Regenerating goal for {agent} due to collision with {other_agent}")
                    while True:
                        # Generate a new goal for the agent
                        new_goal =  get_random_goal_positions(env_name=self.env_name, 
                                                            num_agents=self.num_agents, 
                                                            goal_size=self.goal_size, 
                                                            margin=self.goal_margin, 
                                                            scale=self.goal_scale
                                                            )[agent]
                        # Ensure the new goal is not the same as the current position or any other goal
                        if not np.array_equal(new_goal, self.agent_positions[agent]) and not np.array_equal(new_goal, other_goal):
                            self.agent_goals[agent] = new_goal
                            break

        self.agent_speeds = {agent: 0.0 for agent in self.agents}
        self.agent_orientations = {agent: 0.0 for agent in self.agents}





        
        # Agent and goal colors
        self.unique_agent_goal_colors = {}
        existing_colors = [(255, 165, 0),]
        for agent in self.agents:
            color = generate_unique_color(existing_colors, min_distance=50)
            self.unique_agent_goal_colors[agent] = color
            existing_colors.append(color)
        """
        print('#'*100)
        print("Contents in self.unique_agent_goal_colors:")
        print(self.unique_agent_goal_colors)
        print('-'*100)
        print("Contents in list of existing_colors")
        print(existing_colors)
        print('#'*100)
            """
        # Initialize previous distances to goal for each agent
        self.previous_distances = {agent: self.distance_to_goal(agent) for agent in self.agents}
        
        # Episode and agent-specific attributes
        self.current_step = 0
        self.max_timesteps = self.env_config.get("timesteps_per_episode", 5)
        self.max_speed = self.env_config.get("max_speed", 1.0)
        self.goal_radius = self.env_config.get("goal_radius", 0.1)  # Distance within which the goal is considered reached
        self.dt = self.env_config.get("dt", 0.5)   # dt
#        self.sensor_range = self.env_config.get("sensor_range", 1)

        # Initialize sensor parameters
        self.sensor_range_factor = env_config.get("sensor_range_factor", 1.5)  # Number of rays
        self.sensor_range = max(self.agent_size[0], self.agent_size[1]) * self.sensor_range_factor  # Maximum range of sensors
#        self.sensor_range = round(max(self.agent_size[0], self.agent_size[1]) * 2)  # Maximum range of sensors. Round to keep as integer
        self.num_sensor_rays = env_config.get("num_sensor_rays", 45)  # Number of rays
        self.sensor_fov = env_config.get("sensor_fov", np.pi*2)  # Field of view in radians (equiv 360 degrees)
        # Initialize as empty dictionaries to avoid any potential attribute error if the code accidentally tries to reference them before computation
        self.sensor_rays = {}  # Initialize as empty dictionaries
        self.lidar_readings = {}  # Initialize as empty dictionaries
        self.sensor_rays = {}     # placeholder data to avoid errors before the first step.
        self.lidar_readings = {}

        # Initialize Reward parameters
        self.SAFE_DISTANCE = 1.25 * calculate_safe_dist((self.agent_size[0] / 2), (self.agent_size[1] / 2))

        # Fetch the environment layout from env_helper
        self.env_layout = get_env_layout(self.env_name)
        """
        print('='*100)
        print("env_layout:", self.env_layout)
        print('-'*100)
            """
        self.field_size = self.env_layout["field_size"]
        """
        print("field_size:", self.field_size)
        print('-'*100)
            """
        self.obstacles = [
            (tuple(entry["position"]), tuple(entry["dimensions"]))  
            for entry in self.env_layout.get("obstacles", [])
        ]
        """
        print("obstacles:", self.obstacles)
        print('-'*100)
            """
        self.workstations = [
            (tuple(entry["position"]), tuple(entry["dimensions"]))  
            for entry in self.env_layout.get("workstations", [])
        ]
        """
        print("workstations:", self.workstations)
        print('-'*100)
            """
        self.entrances = [
            (tuple(entry["position"]), tuple(entry["dimensions"]))    # Ensure "position" is a tuple (x, y)
            for entry in self.env_layout.get("entrances", [])
            if isinstance(entry["position"], (tuple, list)) and len(entry["position"]) == 2
        ]
        """
        print("entrances:", self.entrances)
        print('-'*100)
            """
        self.storage_areas = [
            (tuple(entry["position"]), tuple(entry["dimensions"])) 
            for entry in self.env_layout.get("storage_areas", [])
        ]
        """
        print("storage_areas:", self.storage_areas)
        print('-'*100) 
            """
        self.loading_docks = [
            (tuple(entry["position"]), tuple(entry["dimensions"]))  
            for entry in self.env_layout.get("loading_docks", [])
        ]
        """
        print("loading_docks:", self.loading_docks)
        print('-'*100) 
            """

        self.walls = [
            (tuple(entry["position"]), tuple(entry["dimensions"])) 
            for entry in self.env_layout.get("walls", [])
        ]
        """
        print("walls:", self.walls)
        print('-'*100) 
            """

        # Split walls into horizontal and vertical based on dimensions (for initialization logic -> instance_sprites)
        self.horizontal_walls = [wall for wall in self.walls if wall[1][1] < wall[1][0]]  # wall[1] is dimensions (width, height)
        self.vertical_walls = [wall for wall in self.walls if wall[1][0] < wall[1][1]]

        # Initialize wall types into horizontal and vertical (for rendering logic)
        self.wall_types = ["horizontal" if wall[1][0] > wall[1][1] else "vertical" for wall in self.walls]
      
        # Ensure the field size is an integer tuple
        self.field_size = tuple(map(int, self.field_size))
        

        # Calculate margins based on the first wall's dimensions
        self.horizontal_margin = (self.agent_size[0] / 2) + self.walls[0][1][0]  # Access width of the first wall
        self.vertical_margin = (self.agent_size[1] / 2) + self.walls[0][1][1]    # Access height of the first wall
        
        # Store the larger of the horizontal and vertical margins
        self.computed_margin = max(self.horizontal_margin, self.vertical_margin)

        # Initialize environment display parameters
        self.screen_width, self.screen_height, self.common_scale = compute_screen_size(self.field_size)
        
        #Initialize scale_x and scale_y
        #self.scale_x = self.screen_width / self.field_size[0]  # Pixels per logical unit (x)
        #self.scale_y = self.screen_height / self.field_size[1]  # Pixels per logical unit (y) 
        self.scale_x = self.common_scale  # Pixels per logical unit (x)
        self.scale_y = self.common_scale  # Pixels per logical unit (y)
        print("scale_x, scale_y:", self.scale_x, self.scale_y)
        print('-'*100) 

        # Initialize agent_states dictionary.Each agent's state should include velocity and orientation.
        self.agent_states = {}
        #self.terminated_agents = set()  # Keeps track of agents that have reached their goals
        self.terminated_agents = {}  # Keeps track of agents that have reached their goals

        # Initialize the policy
        #self.policy = SmoothMovementPolicy(self.SAFE_DISTANCE)
        
        # Initialize Agent and Goal Rects using the predefined functions
        self.agent_rects = self.create_agent_rects()
        assert all(isinstance(rect, pygame.Rect) for rect in self.agent_rects.values()), "All elements in self.agent_rects must be pygame.Rect"
        self.goal_rects = self.create_goal_rects()
        assert all(isinstance(rect, pygame.Rect) for rect in self.goal_rects.values()), "All elements in self.goal_rects must be pygame.Rect"
        
        """
        # DEBUG: print
        print(":"*100)
        print("contents of self.agent_rects computed using the predefined create_agent_rects function:")
        for key, rect in self.agent_rects.items():
            print(f"{key}:")
            if isinstance(rect, pygame.Rect):  # Ensure it's a valid pygame.Rect
                print(f"  Rectangle: {rect}, Center: {rect.center}")
            else:
                print(f"  Rectangle {i}: INVALID RECT ({rect})")
                
        print(":"*100)       
        print("Contents of self.goal_rects using the predefined create_goal_rects function:")
        for key, rect in self.goal_rects.items():
            print(f"{key}:")
            if isinstance(rect, pygame.Rect):  # Ensure it's a valid pygame.Rect
                print(f"  Rectangle: {rect}, Center: {rect.center}")
            else:
                print(f"  Rectangle {i}: INVALID RECT ({rect})")
         
        print(":"*100)
            """
        
        # Initialize collision boundaries for all categories using setup_collision_boundaries method        
        self.collision_boundaries = self.setup_collision_boundaries()
        
        # Update collision boundaries and LIDAR sensor readings
        self.collision_boundaries["agent_boundaries"] = self.agent_rects
        
        self.sensor_rays = compute_all_sensor_rays(
            agent_positions=self.agent_positions,
            agent_orientations=self.agent_orientations,
            sensor_fov=self.sensor_fov,
            num_sensor_rays=self.num_sensor_rays,
            sensor_range=self.sensor_range,
        )
        self.lidar_readings = compute_lidar(
            self.sensor_rays,
            self.collision_boundaries,
            sensor_range=self.sensor_range,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
        )
        
        # DEBUG: see contents inside self.collision_boundaries
        #debug_collision_boundaries(self.collision_boundaries)


        # Access boundaries for individual categories
        self.obstacle_rects = self.collision_boundaries.get("obstacles", [])
        self.entrance_rects = self.collision_boundaries.get("entrances", [])
        self.workstation_rects = self.collision_boundaries.get("workstations", [])
        self.storage_area_rects = self.collision_boundaries.get("storage_areas", [])
        self.loading_dock_rects = self.collision_boundaries.get("loading_docks", [])
        self.wall_rects = self.collision_boundaries.get("walls", [])
        self.goal_boundaries = self.collision_boundaries.get("goal_boundaries", [])
        self.agent_boundaries = self.collision_boundaries.get("agent_boundaries", [])


        
        # Initialize action and observation spaces by calling generate_spaces function        
        #  Define observation space for each agent with added complexity (hybrid obs-> common and agent specific)
        self.obs_space, self.act_space = generate_space(
            agents=list(self.agents),  #  self.agents contains all agent IDs
            field_size=self.field_size,  #  field_size is an attribute of the environment
            obstacles=self.obstacles,
            workstations=self.workstations,
            entrances=self.entrances,
            storage_areas=self.storage_areas,
            loading_docks=self.loading_docks,
            walls=self.walls,
            max_speed=self.max_speed,
            sensor_range=self.sensor_range,
            num_sensor_rays=self.num_sensor_rays,
            collision_boundaries=self.collision_boundaries,
            num_agents=len(self.agents),
            max_timesteps=self.max_timesteps,
        )
            
         # Define spaces for each agent
        #self.spaces = {f"agent_{i}": {"action_space": self.action_space, "observation_space": self.observation_space} for i in range(self.num_agents)}

        
        # Initialize action and observation spaces by calling generate_spaces function        
        #  Define observation space for each agent with added complexity (hybrid obs-> common and agent specific)
        self.observation_spaces = {f"agent_{i}": self.obs_space for i in range(self.num_agents)}
        self.action_spaces = {f"agent_{i}": self.act_space for i in range(self.num_agents)}

        
        """
        
        self.observation_spaces = {}
        self.action_spaces = {}

        # Dynamically create observation and action spaces for each agent
        for i in range(self.num_agents):
            self.observation_spaces[f"agent_{i}"] = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,))
            self.action_spaces[f"agent_{i}"] = gym.spaces.Discrete(2)  # Example: 2 possible actions for each agent

        
            """
        """
        self.observation_spaces, self.action_spaces = generate_spaces(
            agents=list(self.agents),  #  self.agents contains all agent IDs
            field_size=self.field_size,  #  field_size is an attribute of the environment
            obstacles=self.obstacles,
            workstations=self.workstations,
            entrances=self.entrances,
            storage_areas=self.storage_areas,
            loading_docks=self.loading_docks,
            walls=self.walls,
            max_speed=self.max_speed,
            sensor_range=self.sensor_range,
            num_sensor_rays=self.num_sensor_rays,
            collision_boundaries=self.collision_boundaries,
            num_agents=len(self.agents),
            max_timesteps=self.max_timesteps,
        )

        # Initialize Global Action and Observation Space (self.action_space, self.observation_space) into gymnasium.spaces.Dict
        # This combines all individual agents' observation spaces (self.observation_spaces) into a single spaces.Dict object.
        #self.action_space = spaces.Dict(self.action_spaces)
        #self.observation_space = spaces.Dict(self.observation_spaces) 
            """
        
        """
        # If the concerned Framework (e.g. rllib) Expects a Flattened Observation/Action Space:
        #    Then, Flatten all agent-specific spaces into a single global space:
        # Resulting structure: 
                            self.observation_space = spaces.Dict({
                                "agent_1": spaces.Dict(...),  # Observation space for agent_1
                                "agent_2": spaces.Dict(...),  # Observation space for agent_2
                                ...
                            })
       

        # To flatten, use the following                    
        self.observation_space = spaces.Dict({
            "agent_observations": spaces.Tuple(
                tuple(self.observation_spaces[agent] for agent in self.agents)
            )
        })
        self.action_space = spaces.Tuple(
            tuple(self.action_spaces[agent] for agent in self.agents)
        )
        """
        """
        # Debug print for agent and goal rectangles
        print(":"*100)
        print("Agent Positions inside the init method stage:")
        for key, value in self.agent_positions.items():
            print(f"{key}: {value}")
        print(":"*100)
        print("Goal Positions inside the init method stage:")
        for key, value in self.goal_positions.items():
            print(f"{key}: {value}")
        print(":"*100)

            """
        

        
        # Initialize reward-related attributes/parameters
        self.reward_scale_goal_proximity = 10.0*0.001  # Scaling factor for goal proximity reward
        self.penalty_no_progress = 1.0*0.001
        self.penalty_collision = 5.0*0.001
        self.penalty_proximity = 2.0*0.001
        self.penalty_proximity_obstacle = 1.0*0.001
        self.reward_goal_reached = 100.0

        
   
        # Rendering setup
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode in self.metadata["render_modes"]:
            pygame.init()  # Initialize pygame

            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
#            self.scale = self.screen_width / max(self.field_size) 
#            self.scale = min(self.screen_width / self.field_size[0], self.screen_height / self.field_size[1])
            
                              
            # Handle Simulation Background
            self.background = pygame.image.load("company_assets/background-01.png").convert_alpha()
            self.background = pygame.transform.scale(self.background, (self.screen_width, self.screen_height))
            
            # Define image paths
            base_path = "company_assets/"
            image_paths = {
                "agent": [f"{base_path}agent_1_sprite.png", f"{base_path}agent_2_sprite.png",
                         f"{base_path}agent_3_sprite.png", f"{base_path}agent_4_sprite.png",
                         f"{base_path}agent_5_sprite.png",f"{base_path}agent_6_sprite.png", 
                         f"{base_path}agent_7_sprite.png"],
                "goal": [f"{base_path}goal_1_sprite.jpg", f"{base_path}goal_2_sprite.jpg",
                        f"{base_path}goal_3_sprite.jpg", f"{base_path}goal_4_sprite.jpg",
                        f"{base_path}goal_5_sprite.jpg", f"{base_path}goal_6_sprite.jpg", 
                        f"{base_path}goal_7_sprite.jpg"],
                "obstacle": [f"{base_path}obstacle_3_sprite.png",f"{base_path}obstacle_1_sprite.png", 
                             f"{base_path}obstacle_2_sprite.png"],
                "entrance": [f"{base_path}entrance_1_sprite.jpg"],
                "workstation": [f"{base_path}workstation_2_sprite.png", f"{base_path}workstation_1_sprite.png"],
                "storage_area": [f"{base_path}storage_area_1_sprite.png"],
                "loading_dock": [f"{base_path}loading_dock_1_sprite.png"],
                "wall_horizontal": [f"{base_path}wall_horizontal_1_sprite.png"],      # Horizontal wall sprite
                "wall_vertical": [f"{base_path}wall_vertical_1_sprite.png"],      # Vertical wall sprite

            }

            
            # Define target sizes (in pixels) for different sprite categories
            sprite_sizes = get_sprite_sizes(self.env_layout, self.agent_size, self.goal_size, self.scale_x, self.scale_y)  
            """
            print("+"*100)
            print("Contents stored in sprite_sizes dictionary for each category:")
            for key, value in sprite_sizes.items():
                print(f"{key}: {value}")
            print("+"*100)
                
            
            # Define target sizes (in pixels) for different sprite categories
            old_sprite_sizes = {
                "agent": (32, 32),        # Small size for agents
                "goal": (16, 16),         # Same size as goal
                "obstacle": (64, 64),     # Small obstacles
                "entrance": (32, 32),     # Medium-sized entrances
                "workstation": (64, 64),  # Medium-sized workstations
                "storage_area": (128, 128),  # Large storage areas
                "loading_dock": (64, 64),  # Large loading docks
                "wall_horizontal": (960, 32),  # Horizontal wall spanning the width of the field
                "wall_vertical": (32, 640),     # Vertical wall spanning the height of the field            
            }

            print("&"*100)
            print("Contents stored in old_sprite_sizes dictionary for each category:")
            for key, value in old_sprite_sizes.items():
                print(f"{key}: {value}")
            print("+"*100)
                """
            # Preprocess images and store in self.sprites
            transparency_color = (255, 255, 255)
            
            # Preprocess images and store in self.sprites
            self.sprites = {
                key: preprocess_images(
                    {f"{key}_{i}": path for i, path in enumerate(paths)},
                    sprite_sizes[key],  # Use specific sprite size for this category
                    transparency_color
                )
                for key, paths in image_paths.items()
            }
            """
            print("-"*100)
            print("Surface Sprites stored in self.sprites for each category:")
            for key, value in self.sprites.items():
                print(f"{key}: {value}")
            print("-"*100)
                """
            # Map individual instances to CustomSprite objects
            self.instance_sprites = _initialize_sprites(
                agent_positions=self.agent_positions,
                goal_positions=self.goal_positions,
                obstacles=self.obstacles,
                horizontal_walls=self.horizontal_walls,
                vertical_walls=self.vertical_walls,
                entrances=getattr(self, "entrances", None),
                workstations=getattr(self, "workstations", None),
                storage_areas=getattr(self, "storage_areas", None),
                loading_docks=getattr(self, "loading_docks", None),
                sprites=self.sprites,
                scale_x=self.scale_x,
                scale_y=self.scale_y,
            )
            
            """
            print(":" * 100)
            # Debug:
            print(self.instance_sprites)
            print("=" * 100)
            
            # Debug:
            for category, sprites in self.instance_sprites.items():
                print(f"Inspecting category: {category}")
            
                if isinstance(sprites, dict):  # Dictionary of sprites (e.g., 'agents', 'goals', 'walls')
                    for sprite_id, sprite in sprites.items():
                        if hasattr(sprite, 'rect') and hasattr(sprite.rect, 'center'):  # Ensure the sprite has a rect and rect.center attribute
                            print(f"Sprite ID: {sprite_id}, Rect: {sprite.rect}, Center: {sprite.rect.center}")
                            
                        else:
                            print(f"Sprite ID: {sprite_id} does not have a rect or rect.center attribute!")
                    
            
                        # Check if the sprite is a dictionary or list itself
                        if isinstance(sprite, dict):
                            for sub_id, sub_sprite in sprite.items():
                                if hasattr(sub_sprite, 'rect') and hasattr(sub_sprite.rect, 'center'):
                                    print(f"Sub-Sprite ID: {sub_id}, Rect: {sub_sprite.rect}, Center: {sub_sprite.rect.center}")
                         
                                else:
                                    print(f"Sub-Sprite ID: {sub_id} does not have a rect or rect.center attribute!")
                           
            
                        elif isinstance(sprite, list):
                            for i, sub_sprite in enumerate(sprite):
                                if hasattr(sub_sprite, 'rect') and hasattr(sub_sprite.rect, 'center'):
                                    print(f"Sub-Sprite {i} in Sprite ID '{sprite_id}', Rect: {sub_sprite.rect}, Center: {sub_sprite.rect.center}")
                        
                                else:
                                    print(f"Sub-Sprite {i} in Sprite ID '{sprite_id}' does not have a rect or rect.center attribute!")
                                 
            
                elif isinstance(sprites, list):  # List of sprites (e.g., 'obstacles', 'entrances', 'workstations', 'storage_areas', 'loading_docks')
                    for i, sprite in enumerate(sprites):
                        if hasattr(sprite, 'rect') and hasattr(sprite.rect, 'center'):  # Ensure the sprite has a rect and rect.center attribute
                            print(f"Sprite {i} in category '{category}', Rect: {sprite.rect}, Center: {sprite.rect.center}")
                     
                        else:
                            print(f"Sprite {i} in category '{category}' does not have a rect or rect.center attribute!")
                       
            
                        # Check if the sprite is a dictionary or list itself
                        if isinstance(sprite, dict):
                            for sub_id, sub_sprite in sprite.items():
                                if hasattr(sub_sprite, 'rect') and hasattr(sub_sprite.rect, 'center'):
                                    print(f"Sub-Sprite ID: {sub_id}, Rect: {sub_sprite.rect}, Center: {sub_sprite.rect.center}")
                        
                                else:
                                    print(f"Sub-Sprite ID: {sub_id} does not have a rect or rect.center attribute!")
                                 
            
                        elif isinstance(sprite, list):
                            for j, sub_sprite in enumerate(sprite):
                                if hasattr(sub_sprite, 'rect') and hasattr(sub_sprite.rect, 'center'):
                                    print(f"Sub-Sprite {j} in Sprite {i}, Rect: {sub_sprite.rect}, Center: {sub_sprite.rect.center}")
                                 
                                else:
                                    print(f"Sub-Sprite {j} in Sprite {i} does not have a rect or rect.center attribute!")
                                   
            
                else:
                    print(f"Unexpected type in category '{category}': {type(sprites)}")
                  
                print(":" * 100)    

                """
        print("ENDING DEBUG OUTPUT FOR INIT METHOD:>")
        print(":"*100) 

        super().__init__()

    
#--------------------------------------HELPER METHODS----------------------------------------

    @property
    def num_agents(self):        # Modification for setting number of agents to avoid naming conflict
        return self._num_agents  # Getter
    
    @num_agents.setter
    def num_agents(self, value):
        if value < 0:
            raise ValueError("Number of agents cannot be negative.")
        self._num_agents = value  # Setter with validation 

    def distance_to_goal(self, agent_id: str) -> float:
        """Calculates the Euclidean distance from the agent's current position to its goal."""
        # Convert position and goal to NumPy arrays for vectorized operations
        current_agent_position = np.array(self.agent_positions[agent_id])
        target_goal_position = np.array(self.agent_goals[agent_id])
        # Calculate and return the Euclidean distance
        return np.linalg.norm(current_agent_position - target_goal_position)

    

    def _get_observation(self, agent_id):
        """
        Get the observation for a specific agent by calling the helper function.
    
        Args:
            agent_id (str): The ID of the agent.
    
        Returns:
            dict: Observation for the agent.
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found in the environment.")
            
        #return {i: self.get_observation_space(i).sample() for i in self.agents}
        return self.get_observation_space(agent_id).sample()

    
  

    
    def get_local_sensor_data(self, agent_id):
        """
        Computes local sensor observations for the agent (e.g., LIDAR or proximity data).
        """
        # Replace with actual sensor observation logic
        return self.sensors[agent_id]


    def clamp_position(self, x, y):
        """
        Clamp the agent's position to ensure it stays within the bounds of the screen.
    
        Args:
            position (tuple): Current position of the agent (x, y). 
    
        Returns:
            tuple: Clamped (x, y) position.
        """
        # Account for the agent's size if necessary (e.g., using self.agent_size)
        clamped_x = max(self.margin * self.scale, min(self.screen_width - self.margin * self.scale, x))
        clamped_y = max(self.margin * self.scale, min(self.screen_height - self.margin * self.scale, y))
        
        # Optional: If I have an agent size, I need to ensure it's not partially off-screen
        # For example:
        # agent_size = self.agent_size * self.scale
        # clamped_x = max(agent_size / 2, min(self.screen_width - agent_size / 2, clamped_x))
        # clamped_y = max(agent_size / 2, min(self.screen_height - agent_size / 2, clamped_y))
    
        return clamped_x, clamped_y


    def resolve_collision(self, agent, proposed_position):
        """
        Resolves collisions by adjusting the agent's position to avoid restricted areas.
    
        Args:
            agent (str): The agent's identifier.
            proposed_position (tuple): The proposed position that caused a collision.
    
        Returns:
            tuple: The corrected position of the agent.
        """
        x, y = proposed_position
        
        field_width, field_height = self.field_size  # Unpack the field dimensions
    
        # Apply random small displacement to move away from collision
        angle = np.random.uniform(0, 2 * np.pi)
        dx = 0.1 * np.cos(angle)
        dy = 0.1 * np.sin(angle)
        corrected_x = max(0, min(x + dx, field_width))
        corrected_y = max(0, min(y + dy, field_height))
    
        return corrected_x, corrected_y
    
    def _apply_action(self, agent, action):
        """
        Applies the given action to update the agent's state (position, velocity, orientation),
        ensuring smooth movement and handling collisions.
        
        Args:
            agent (str): The ID of the agent.
            action (list or array): The action for the agent, assumed to be [delta_orientation, delta_speed].
        """

                # Helper to create a centered rectangle
        def _create_centered_rect(position, size):
            rect = pygame.Rect(0, 0, size[0] * self.scale_x, size[1] * self.scale_y)
            rect.center = (
                int(position[0] * self.scale_x),
                int(position[1] * self.scale_y),
            )
            return rect
        
        # Parse the action
        delta_orientation, delta_speed = action
    
        # Update orientation and wrap within [-π, π]
        self.agent_states[agent]['orientation'] = (self.agent_states[agent]['orientation'] + delta_orientation) % (2 * np.pi)
        if self.agent_states[agent]['orientation'] > np.pi:
            self.agent_states[agent]['orientation'] -= 2 * np.pi
    
        # Update velocity, ensuring it remains within the valid range
        self.agent_states[agent]['velocity'] = max(0, min(self.max_speed, self.agent_states[agent]['velocity'] + delta_speed))
    
        # Calculate the new position based on updated velocity and orientation
        x, y = self.agent_positions[agent]
        velocity = self.agent_states[agent]['velocity']
        orientation = self.agent_states[agent]['orientation']
    
        proposed_x = x + velocity * np.cos(orientation)
        proposed_y = y + velocity * np.sin(orientation)
        proposed_position = (proposed_x, proposed_y)
    
        # Create a rect for the proposed position
        proposed_agent_rect = _create_centered_rect(proposed_position, self.agent_size)

        # Check if the proposed position is valid
        if self.is_position_valid_XMG(agent, proposed_agent_rect):
            # Update position if valid
            self.agent_positions[agent] = proposed_position
        else:
            # If Collision is detected using is_position_valid function: Proceed to Resolve the collision by calling resolve_collision function
            corrected_position = self.resolve_collision(agent, proposed_position)
            # Debug: check its correctness
            print("-"*100)
            print(f"Proposed position: {proposed_position}, Corrected position: ({corrected_position[0]}, {corrected_position[1]})")
            print("-"*100)
            self.agent_positions[agent] = corrected_position
    
        # Debugging output (optional)
        print("-" * 100)
        print(f"Agent {agent} -> Action: {action}, New Position: {self.agent_positions[agent]}, Velocity: {velocity}, Orientation: {orientation}")
        print("-" * 100)
    
        
    def create_category_rects(self, category):
        """
        Create pygame.Rect objects for a given category of items.
    
        Args:
            category (str): The name of the category (e.g., 'obstacles', 'walls').
    
        Returns:
            list: List of pygame.Rect objects representing the scaled boundaries for the items.
        """
        return create_category_rects(self.env_layout[category], self.scale_x, self.scale_y)


    
    def is_position_valid_XMG(self, agent, agent_rect):
        """
        Validate if an agent's position is free of collisions.
    
        Args:
            agent (str): The agent's ID.
            agent_rect (pygame.Rect): The agent's rectangle for collision testing.
    
        Returns:
            bool: True if the position is valid, False if there is a collision.
        """
        # Call _is_position_valid to check for collisions
        return _is_position_valid_XMG(agent, agent_rect, self.collision_boundaries, self.agent_rects, self.goal_rects)
        

    def is_position_valid_XMAG(self, agent, agent_rect):
        """
        Validate if an agent's position is free of collisions.
    
        Args:
            agent (str): The agent's ID.
            agent_rect (pygame.Rect): The agent's rectangle for collision testing.
    
        Returns:
            bool: True if the position is valid, False if there is a collision.
        """
        # Call _is_position_valid to check for collisions
        return _is_position_valid_XMAG(agent, agent_rect, self.collision_boundaries, self.agent_rects, self.goal_rects)
        

    def setup_collision_boundaries(self):
        """
        Set up collision boundaries for all categories in the environment.
    
        Returns:
            dict: A dictionary mapping category names to their respective pygame.Rect objects.
        """
        return setup_collision_boundaries(self.env_layout, self.goal_positions, self.goal_size, self.scale_x, self.scale_y)
    
    def create_agent_rects(self):
        """Create agent rects based on current positions."""
        return _create_agent_rects(self.agent_positions, self.agent_size, self.scale_x, self.scale_y)

    def create_goal_rects(self):
        """Create goal rects based on goal positions."""
        return _create_goal_rects(self.goal_positions, self.goal_size, self.scale_x, self.scale_y)

    
    def calculate_agent_reward(self, agent, distance_to_goal):
        """
        Calls the reward calculation function from env_helpers.py
        to calculate the reward for the given agent.

        Args:
            agent (str): The ID of the agent.
            distance_to_goal (float): The current distance to the goal.

        Returns:
            float: The calculated reward for the agent.
        """
        reward = _calculate_reward(
            agent,
            distance_to_goal,
            self.previous_distances,
            self.agent_rects,
            self.goal_rects,
            self.collision_boundaries,
            self.penalty_no_progress,
            self.penalty_collision,
            self.penalty_proximity,
            self.penalty_proximity_obstacle,
            self.reward_scale_goal_proximity,
            self.goal_radius,
            self.reward_goal_reached,
            self.SAFE_DISTANCE,
            self.lidar_readings
        )

        return reward




    
#------------------------------------RESET METHOD------------------------------------------------
    
    def is_position_valid_XM(self, agent, agent_rect):
        """
        Validate if an agent's position is free of collisions.
    
        Args:
            agent (str): The agent's ID.
            agent_rect (pygame.Rect): The agent's rectangle for collision testing.
    
        Returns:
            bool: True if the position is valid, False if there is a collision.
        """
        # Call _is_position_valid to check for collisions
        return _is_position_valid_XM(agent, agent_rect, self.collision_boundaries, self.agent_rects, self.goal_rects)

    def is_position_valid_XG(self, agent, agent_rect):
        """
        Validate if an agent's position is free of collisions.
    
        Args:
            agent (str): The agent's ID.
            agent_rect (pygame.Rect): The agent's rectangle for collision testing.
    
        Returns:
            bool: True if the position is valid, False if there is a collision.
        """
        # Call _is_position_valid to check for collisions
        return _is_position_valid_XG(agent, agent_rect, self.collision_boundaries, self.agent_rects, self.goal_rects) 



    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to its initial state and returns the initial observations.
    
        Args:
            seed (int, optional): Seed for random number generation.
            options (dict, optional): Additional options for reset.
    
        Returns:
            tuple: A dictionary of observations and a dictionary of info.
        """
        # Seed the environment for reproducibility
        super().reset(seed=seed)

        
        self.current_step = 0  # Reset step counter
        
        print(":"*100) 
        print("BEGINING DEBUG OUTPUT FOR RESET METHOD:<")
        
        # Reset state variables
        #self.terminated_agents = set()  # Clear terminated agents
        self.terminated_agents = {}  # Clear terminated agents
        self.agent_positions = {}
        self.agent_goals = {}
        self.agent_rects = {}
        self.previous_distances = {}
        self.agent_states = {}
        #print("-"*100)
        #print("Contents inside self.agent_states:", self.agent_states)
        self.sensor_rays = {}  # Initialize as empty dictionaries
        self.lidar_readings = {}  # Initialize as empty dictionaries
        self.sensor_rays = {}     # placeholder data to avoid errors before the first step.
        self.lidar_readings = {}
        self.agent_statuses = {agent: 0 for agent in self.agents}
        #print("-"*100)
        #print("Contents inside self.agent_statuses:", self.agent_statuses)
        #print("-"*100)

        
        # Helper to create a centered rectangle
        def create_centered_rect(position, size):
            rect = pygame.Rect(0, 0, size[0] * self.scale_x, size[1] * self.scale_y)
            rect.center = (
                int(position[0] * self.scale_x),
                int(position[1] * self.scale_y),
            )
            return rect
    
        # Process predefined start and goal positions
        if self.starting_point:
            for agent, start_pos, goal_pos in zip(self.agents, self.start_positions, self.goal_positions):
                # Validate and center-align predefined start position
                start_rect = create_centered_rect(start_pos, self.agent_size)
                if self.is_position_valid_XM(agent, start_rect):
                    self.agent_positions[agent] = start_pos
                    self.agent_rects[agent] = start_rect
                else:
                    print(f"Warning: Predefined start position for {agent} is invalid. Generating a random one.")
                    self.agent_positions[agent] = None  # Mark for random generation
                
                # Validate and center-align predefined goal position
                goal_rect = create_centered_rect(goal_pos, self.goal_size)
                if self.is_position_valid_XG(agent, goal_rect) and start_pos != goal_pos:
                    self.agent_goals[agent] = goal_pos
                else:
                    print(f"Warning: Predefined goal position for {agent} is invalid. Generating a random one.")
                    self.agent_goals[agent] = None  # Mark for random generation
        else:
            # Mark all for random generation if no predefined positions
            for agent in self.agents:
                self.agent_positions[agent] = None
                self.agent_goals[agent] = None
    
        # Generate random positions where needed
        for agent in self.agents:
            # Generate random start position
            if self.agent_positions[agent] is None:
                while True:
                    proposed_start = get_random_start_positions(env_name=self.env_name, 
                                                              num_agents=self.num_agents,
                                                              object_size=self.agent_size,
                                                              margin=self.agent_margin, 
                                                              scale=self.agent_scale)[agent]
                    start_rect = create_centered_rect(proposed_start, self.agent_size)
                    if self.is_position_valid_XM(agent, start_rect):
                        self.agent_positions[agent] = proposed_start
                        self.agent_rects[agent] = start_rect
                        break
    
            # Generate random goal position
            if self.agent_goals[agent] is None:
                while True:
                    proposed_goal = get_random_goal_positions(env_name=self.env_name, 
                                                            num_agents=self.num_agents, 
                                                            goal_size=self.goal_size, 
                                                            margin=self.goal_margin, 
                                                            scale=self.goal_scale
                                                            )[agent]
                    goal_rect = create_centered_rect(proposed_goal, self.goal_size)
                    if self.is_position_valid_XG(agent, goal_rect) and proposed_goal != self.agent_positions[agent]:
                        self.agent_goals[agent] = proposed_goal
                        break
    

        # Ensure goals are unique
        for agent, goal in self.agent_goals.items():
            for other_agent, other_goal in self.agent_goals.items():
                if agent != other_agent and (np.all(goal == other_goal)):  # Compare arrays element-wise
                    #print(f"Regenerating goal for {agent} due to collision with {other_agent}")
                    while True:
                        # Generate a new goal for the agent
                        new_goal = get_random_goal_positions(env_name=self.env_name, 
                                                            num_agents=self.num_agents, 
                                                            goal_size=self.goal_size, 
                                                            margin=self.goal_margin, 
                                                            scale=self.goal_scale
                                                            )[agent]
                        new_goal_rect = create_centered_rect(new_goal, self.goal_size)

                        # Ensure the new goal is not the same as the current position or any other goal
                        if self.is_position_valid_XG(agent, new_goal_rect) and not np.array_equal(new_goal, self.agent_positions[agent]) and not np.array_equal(new_goal, other_goal):
                            self.agent_goals[agent] = new_goal
                            break

        
        # Initialize agent states and attributes
        self.agent_states = {agent: {"velocity": 0.0, "orientation": 0.0} for agent in self.agents}
        self.agent_status = {agent: 1 for agent in self.agents}  # Active status

        # Reinitialize collision boundaries
        self.collision_boundaries = self.setup_collision_boundaries()
        # Update collision boundaries and LIDAR sensor readings
        self.collision_boundaries["agent_boundaries"] = self.agent_rects

        # Reinitialize sensor rays
        self.sensor_rays = compute_all_sensor_rays(
            agent_positions=self.agent_positions,
            agent_orientations=self.agent_orientations,
            sensor_fov=self.sensor_fov,
            num_sensor_rays=self.num_sensor_rays,
            sensor_range=self.sensor_range,
        )

        # Compute sensor readings and store LIDAR distances for each agent as a dictionary
        self.lidar_readings = compute_lidar(
            self.sensor_rays, 
            self.collision_boundaries,   # Collision boundaries to detect
            sensor_range=self.sensor_range, 
            scale_x=self.scale_x, 
            scale_y=self.scale_y
        )
        # Map individual instances to CustomSprite objects
        if self.render_mode in self.metadata["render_modes"]:
            self.instance_sprites = _initialize_sprites(
                agent_positions=self.agent_positions,
                goal_positions=self.goal_positions,
                obstacles=self.obstacles,
                horizontal_walls=self.horizontal_walls,
                vertical_walls=self.vertical_walls,
                entrances=getattr(self, "entrances", None),
                workstations=getattr(self, "workstations", None),
                storage_areas=getattr(self, "storage_areas", None),
                loading_docks=getattr(self, "loading_docks", None),
                sprites=self.sprites,
                scale_x=self.scale_x,
                scale_y=self.scale_y,
            )
        
        # Construct initial observations
        #observations = {agent: self._get_observation(agent) for agent in self.agents} # Dict of observations per agent
        #infos = {agent: {} for agent in self.agents}  # Empty info dict per agent


        # Construct initial observations
        observations = {}
        infos = {}
        
        observations = {agent: self.get_observation_space(agent).sample() for agent in self.agents}
        
        for agent in self.agents:
            infos[agent] = {}  # Initialize empty info dict per agent

        # Ensure the observation space keys are sorted
        #sorted_observation_keys = sorted(self.observation_spaces[agent].spaces.keys())
        
        """        
        for agent in self.agents:

            
            # Get the observation for the agent
            obs = self._get_observation(agent)

            # Sort the 'common' sub-dictionary if it exists
            if 'common' in obs:
                obs['common'] = sort_nested_dict(obs['common'])
            
            # Reorder the observation to match the observation space
            sorted_obs = {key: obs[key] for key in sorted_observation_keys}

            # Verify that the sorted observation matches the observation space
            if not self.observation_space[agent].contains(sorted_obs):
                raise ValueError(
                    f"Observation for agent {agent} does not match the defined observation space.\n"
                    f"Expected: {self.observation_space[agent]}\nActual: {sorted_obs}"
                )
            
            # Update observations and infos
            observations[agent] = self._get_observation(agent)
            #observations[agent] = sorted_obs
            infos[agent] = {}  # Initialize empty info dict per agent

        
        # Debug print for alignment
        print("#" * 100)
        for agent, rect in self.agent_rects.items():
            print(f"{agent} - Center: {rect.center}, Position: {self.agent_positions[agent]}")
        print("#" * 100)

        # Debug: Print the content of self.sensor_rays to check if and how it's populated
        print("="*100)
        print("Sensor rays contents stored in self.sensor_rays all agents:")
        for key, value in self.sensor_rays.items():
            print(f"{key}: {value}")
        print("-"*100) 
        """
        # Initialize rendering if required
        if self.render_mode in ["human", "rgb_array"]:
            if not pygame.get_init():
                pygame.init()
            if not self.screen:
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
    
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            self._render_rgb_array()


        print("ENDING DEBUG OUTPUT FOR RESET METHOD:>")
        print(":"*100)

        return observations, infos

#----------------------------STEP METHOD--------------------------------------------------


    
    def _apply_smooth_action(self, agent, action):
        
        """
        Applies the given action to update the agent's state (position, velocity, orientation),
        ensuring smooth movement.
        
        Args:
            agent (str): The ID of the agent.
            action (list or array): The action for the agent, assumed to be [delta_orientation, delta_speed].
        """
        # If agent is already terminated, do nothing
        if self.terminated_agents.get(agent, False):
            return
            
        delta_orientation, delta_speed = action
    
        # Get agent's current state
        x, y = self.agent_positions[agent]
        current_speed = self.agent_states[agent]['velocity']
        current_orientation = self.agent_states[agent]['orientation']
        goal_x, goal_y = self.agent_goals[agent]  # Get goal position
    
        # Check if agent is at goal (within tolerance)
        if np.linalg.norm(np.array([x, y]) - np.array([goal_x, goal_y])) < self.goal_radius:         
            # Store updated values to keep the agent in it's goal
            self.agent_states[agent]['orientation'] = 0  # Stop turning  
            self.agent_states[agent]['velocity'] = 0  # Stop moving  
            self.agent_states[agent]["position"] = (goal_x, goal_y)  # Lock position 
            self.terminated_agents[agent] = True  # Mark as terminated in dictionary
        
            self.agent_orientations[agent] = 0  # Stop turning 
            self.agent_speeds[agent] = 0  # Stop moving
            self.agent_positions[agent] = (goal_x, goal_y)  # Lock position   
        
            return  # Do not update position further
            
    
        # Define maximum orientation change per step (prevents sudden turns)
        max_rotation_speed = np.radians(45)  # Limit to 20 degrees per update (to be adjusted. optimal: 5-15)
        #max_rotation_speed = np.radians(10)  # Limit to 10 degrees per update (to be adjusted. optimal: 5-15)
        #max_rotation_speed = np.radians(7)  # Reduced from 10° for softer turns
        delta_orientation = np.clip(delta_orientation, -max_rotation_speed, max_rotation_speed)
        
        # Smoothly update orientation
        new_orientation = (current_orientation + delta_orientation) % (2 * np.pi)
    
        # Limit acceleration for gradual speed changes
        #max_acceleration = self.max_speed / 10  # to be adjusted for smoother acceleration
        max_acceleration = self.max_speed / 2  # Reduced acceleration for smoother transitions
        new_speed = np.clip(current_speed + np.clip(delta_speed, -max_acceleration, max_acceleration), 0, self.max_speed)
    
        # **Trapezoidal Velocity Integration (Smoother Position Updates)**
        #avg_speed = (current_speed + new_speed) / 2  # Prevents sudden jumps
        avg_speed = (3 * current_speed + new_speed) / 4  # Weighted average for smoother changes
        #avg_speed = (current_speed + 2 * new_speed) / 3  # More emphasis on new speed
        dx = avg_speed * np.cos(new_orientation) * self.dt # to be adjusted
        dy = avg_speed * np.sin(new_orientation) * self.dt
    
        # Compute new position
        new_x = x + dx
        new_y = y + dy
        new_position = (new_x, new_y)
    
        # Store updated values
        self.agent_states[agent]['orientation'] = new_orientation  
        self.agent_states[agent]['velocity'] = new_speed  
        self.agent_states[agent]["position"] = new_position  
    
        self.agent_orientations[agent] = new_orientation
        self.agent_speeds[agent] = new_speed
        self.agent_positions[agent] = new_position  
        
             
    def _apply_PPOTorch_action(self, agent, action):
        """
        Applies the given action to update the agent's state (position, velocity, orientation),
        ensuring smooth movement and handling collisions.
        
        Args:
            agent (str): The ID of the agent.
            action (list or array): The action for the agent, assumed to be [delta_orientation, delta_speed].
        """
        """
        # Extract action (steering and speed)
        action_steering, action_speed = action
        
        # Compute LIDAR readings distances for collision avoidance 
        lidar_distances = self.lidar_readings[agent]["distances"]

        # Avoid collisions using LIDAR reading output to Compute collision avoidance action
        collision_avoided_action = avoid_collision_action(lidar_distances, self.SAFE_DISTANCE, 
                                                          self.max_speed, action_steering, action_speed)
        
        #       adjusted_steering, adjusted_speed = collision_avoided_action
        
        # Parse the adjusted collision_avoided_action for updating agent states
        delta_orientation, delta_speed = collision_avoided_action
          """  
        delta_orientation, delta_speed = action

        # Update orientation and wrap within [-π, π]
        self.agent_states[agent]['orientation'] = (self.agent_states[agent]['orientation'] + delta_orientation) % (2 * np.pi)
        if self.agent_states[agent]['orientation'] > np.pi:
            self.agent_states[agent]['orientation'] -= 2 * np.pi

        # Update velocity, ensuring it remains within the valid range
        self.agent_states[agent]['velocity'] = max(0, min(self.max_speed, self.agent_states[agent]['velocity'] + delta_speed))
        
        self.agent_orientations[agent] = self.agent_states[agent]['orientation']
        self.agent_speeds[agent] = self.agent_states[agent]['velocity'] 
        
        # Calculate the new position based on updated velocity and orientation

        # COMPUTE ACTION AND USE IT TO UPDATE AGENT-POSITION DICTIONARY (self.agent_positions[agent])        
        x, y = self.agent_positions[agent]
        velocity = self.agent_states[agent]['velocity']
        orientation = self.agent_states[agent]['orientation']

        proposed_x = x + velocity * np.cos(orientation)*self.dt
        proposed_y = y + velocity * np.sin(orientation)*self.dt
        proposed_position = (proposed_x, proposed_y)
        
        # store that agent's new position for further processing
        self.agent_states[agent]['position'] = proposed_position
        self.agent_positions[agent] = proposed_position
             

    def _apply_PFavoid_action(self, agent, action):
        """
        Applies the given action to update the agent's state (position, velocity, orientation),
        ensuring smooth movement and handling collisions.
        
        Args:
            agent (str): The ID of the agent.
            action (list or array): The action for the agent, assumed to be [delta_orientation, delta_speed].
        """

        # Extract action (steering and speed)
        steering, speed = action
    
        
        # # Get computed LIDAR readings data for collision avoidance logic
        lidar_data = self.lidar_readings.get(agent, {
            "distances": [],
            "collision_points": [],
            "collision_flags": [],
        })  

        # Compute Repulsive forces from obstacles based on lidar data
        force_x, force_y = field_forces(agent, lidar_data, self.agent_positions[agent], 
                                           self.goal_positions[agent], self.agent_size, 
                                           self.scale_x, self.scale_y, self.SAFE_DISTANCE)

        # Update the agent's position based on the computed forces
        current_agent_position = self.agent_positions[agent]
        #new_position = current_position + np.array([force_x, force_y])
        current_x, current_y = current_agent_position
        velocity = self.agent_states[agent]['velocity']
        orientation = self.agent_states[agent]['orientation']
        #new_position_x = current_x + force_x
       # new_position_y = current_y + force_y
        
        new_position_x = current_x + force_x * velocity * np.cos(orientation)*self.dt
        new_position_y = current_y + force_y * velocity * np.sin(orientation)*self.dt
        
        new_position = (new_position_x, new_position_y)
        
        # store that agent's new position for further processing
        self.agent_positions[agent] = new_position
    
    def step(self, actions):
        """
        Perform one step of simulation in the environment, update agent states, and return new observations.
        
        Args:
            actions (dict): Dictionary of actions for each agent. Keys are agent IDs, 
                                values are actions (steering, speed) taken by each agent.
    
        Returns:
            dict: New observations for all agents.
            dict: Rewards for each agent.
            bool: Terminated flag indicating if the simulation is finished (goal reached).
            bool: Truncated flag indicating if the simulation is finished (episode end).
            dict: Extra information (e.g., info about collisions).
        """
        
        # Initialize or access the set of terminated agents
        if not hasattr(self, "terminated_agents"):
            #self.terminated_agents = set()  # Track agents that have reached their goals (Clear terminated agents)
            self.terminated_agents = {}  # Track agents that have reached their goals (Clear terminated agents)
        
        self.current_step += 1
    
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        print(":"*100) 
        print("BEGINING DEBUG OUTPUT FOR STEP METHOD:<")

        # Helper to create a centered rectangle
        def _create_centered_rect(position, size):
            rect = pygame.Rect(0, 0, size[0] * self.scale_x, size[1] * self.scale_y)
            rect.center = (
                int(position[0] * self.scale_x),
                int(position[1] * self.scale_y),
            )
            return rect
        """   
        # should be able to access self.instance_sprites here
        if hasattr(self, 'instance_sprites'):
            print("instance_sprites exists in step method", self.instance_sprites)
        else:
            print("instance_sprites does not exist in step method")
            """
        
        """
        rescaled_actions = {
            agent: [
                action[0] * np.pi,  # Scale angular velocity back to [-π, π]
                action[1] * self.max_speed  # Scale speed back to [0, max_speed]
            ]
            for agent, action in actions.items()
        }
        # Then, use rescaled_actions for environment logic
        """
        """
        # Debug: Print the received actions
        print("st"*60) 
        print("Raw actions data received:", actions)
        print("."*100) 
            """
        # Directly use actions for each agent
        for agent, action_data in actions.items():
            if self.terminated_agents.get(agent, False):   
            #if agent in self.terminated_agents:
                # Skip updates for terminated agents
                terminateds[agent] = True
                truncateds[agent] = False
                #observations[agent] = self._get_observation(agent)  
                observations[agent] = self.get_observation_space(agent).sample() # Keep last valid observation
                #rewards[agent] = 0  # No further rewards
                infos[agent] = {"status": "goal_reached"}
                continue


            """
            # ALTERNATIVELY REMOVE AGENT FROM SIMULATION
            if agent in self.terminated_agents:
                del self.agent_positions[agent]
                del self.agent_states[agent]
                continue
                """
    
            # Case 1: Check if action_data is a tuple (action_array, [], info)
            if isinstance(action_data, tuple) and len(action_data) >= 1:
                action = action_data[0]  # Extract the first item (action array)
    
            # Case 2: Check if action_data is a dictionary (for multi-agent case)
            elif isinstance(action_data, dict):
                action = action_data  # For handling case where it's a dict
    
            else:
                # Case 3: Directly use the action (if it's not in a tuple or dictionary)
                action = action_data
            
            # Ensure action is a flat array (in case it's an array or other nested structure)
            action = np.array(action).flatten()

            # Debug: Print the received actions
            #print("Flattened Raw agent's action received:", action)
            
            # Update the agent's state with the raw flattened action
            #self._apply_PPOTorch_action(agent, action)
            self._apply_smooth_action(agent, action)
            #self._apply_PFavoid_action(agent, action) # To be called when needed
            
            # Apply the computed action in the environment
            #self._apply_LidarAC_action(agent, action)

            # Debug output
            #print(f"Updated Position for {agent}: {self.agent_positions[agent]}")
                        
            # Clamp the agent's position within bounds
            x, y = self.agent_positions[agent]
            proposed_position = self.clamp_position(x, y)

            # Debug output
            #print(f"Updated proposed_position after Clamping, for {agent}: {proposed_position}")
                        
            # Create a rect for the proposed position
            proposed_agent_rect = _create_centered_rect(proposed_position, self.agent_size)
    
            # Validate position with collision checks
            #if self.is_position_valid_XMG(agent, proposed_agent_rect):  # Ignore that agent's goal only
            if self.is_position_valid_XMAG(agent, proposed_agent_rect):  # Ignore all agent goals in the simulation
                # Update position if valid
                self.agent_positions[agent] = proposed_position
                self.agent_rects[agent] = proposed_agent_rect

                # Sync sprite with the updated proposed_agent_rect
                if self.render_mode in self.metadata["render_modes"]:
                        
                    if "agents" in self.instance_sprites and agent in self.instance_sprites["agents"]:
                        # Sprite exists, proceed with syncing position
                        sprite = self.instance_sprites["agents"][agent]
                        sprite.rect.center = proposed_agent_rect.center
                    else:
                        # Handle the case where the agent's sprite is missing or not yet initialized
                        if "agents" not in self.instance_sprites:
                            print(f"Warning: 'agents' key is missing in self.instance_sprites.")
                        elif agent not in self.instance_sprites["agents"]:
                            print(f"Warning: Sprite for agent '{agent}' is not initialized in 'agents' dictionary.")
                        
                        # Attempt to reinitialize sprites
                        self.instance_sprites = _initialize_sprites(
                            agent_positions=self.agent_positions,
                            goal_positions=self.goal_positions,
                            obstacles=self.obstacles,
                            horizontal_walls=self.horizontal_walls,
                            vertical_walls=self.vertical_walls,
                            entrances=getattr(self, "entrances", None),
                            workstations=getattr(self, "workstations", None),
                            storage_areas=getattr(self, "storage_areas", None),
                            loading_docks=getattr(self, "loading_docks", None),
                            sprites=self.sprites,
                            scale_x=self.scale_x,
                            scale_y=self.scale_y,
                        )
                    
                        # After reinitializing, check again and sync the sprite
                        if "agents" in self.instance_sprites and agent in self.instance_sprites["agents"]:
                            sprite = self.instance_sprites["agents"][agent]
                            sprite.rect.center = proposed_agent_rect.center
                        else:
                            print(f"Error: Sprite for agent '{agent}' could not be reinitialized.")
    

            else:
                # Collision detected: Adjust position and log info
                corrected_position = self.resolve_collision(agent, proposed_position)
                corrected_position_rect = _create_centered_rect(corrected_position, self.agent_size)
                self.agent_positions[agent] = corrected_position
                self.agent_rects[agent] = corrected_position_rect

                # Sync sprite with the updated corrected_position
                if self.render_mode in self.metadata["render_modes"]:
                    if "agents" in self.instance_sprites and agent in self.instance_sprites["agents"]:
                        # Sprite exists, proceed with syncing position
                        sprite = self.instance_sprites["agents"][agent]
                        sprite.rect.center = corrected_position_rect.center
                    else:
                        # Handle the case where the agent's sprite is missing or not yet initialized
                        if "agents" not in self.instance_sprites:
                            print(f"Warning: 'agents' key is missing in self.instance_sprites.")
                        elif agent not in self.instance_sprites["agents"]:
                            print(f"Warning: Sprite for agent '{agent}' is not initialized in 'agents' dictionary.")
                        
                        # Attempt to reinitialize sprites
                        self.instance_sprites = _initialize_sprites(
                            agent_positions=self.agent_positions,
                            goal_positions=self.goal_positions,
                            obstacles=self.obstacles,
                            horizontal_walls=self.horizontal_walls,
                            vertical_walls=self.vertical_walls,
                            entrances=getattr(self, "entrances", None),
                            workstations=getattr(self, "workstations", None),
                            storage_areas=getattr(self, "storage_areas", None),
                            loading_docks=getattr(self, "loading_docks", None),
                            sprites=self.sprites,
                            scale_x=self.scale_x,
                            scale_y=self.scale_y,
                        )
                    
                        # After reinitializing, check again and sync the sprite
                        if "agents" in self.instance_sprites and agent in self.instance_sprites["agents"]:
                            sprite = self.instance_sprites["agents"][agent]
                            sprite.rect.center = corrected_position_rect.center
                        else:
                            print(f"Error: Sprite for agent '{agent}' could not be reinitialized.")
    
                    
                infos[agent] = {"status": "collision", "invalid_action": action}
    
            # Update distance to goal for reward calculation
            updated_x, updated_y = self.agent_positions[agent] # Get updated agent position
            goal_x, goal_y = self.agent_goals[agent]  # Get goal position
            #distance_to_goal = self.distance_to_goal(agent)
            distance_to_goal = np.linalg.norm(np.array([updated_x, updated_y]) - np.array([goal_x, goal_y])) 


            # Calculate the reward for the agent
            rewards[agent] = self.calculate_agent_reward(agent, distance_to_goal)
    
            # Update observations
            #observations[agent] = self._get_observation(agent)
            observations[agent] = self.get_observation_space(agent).sample() 

    
            # Check if the agent has reached its goal
            #print(f"Agent {agent}: Distance to goal = {distance_to_goal}, Goal radius = {self.goal_radius}")
            if distance_to_goal <= self.goal_radius:  
                #self.terminated_agents[agent] = True  # Mark agent as terminated
                #print(f"Agent {agent} marked as terminated.")
                terminateds[agent] = True
                truncateds[agent] = False
                infos[agent] = {"status": "goal_reached"}
            else:
                terminateds[agent] = False
                truncateds[agent] = self.current_step >= self.max_timesteps
                infos[agent] = infos.get(agent, {})  # Preserve collision info if it exists
                infos[agent]["status"] = "goal not reached at the end of episode"
    
            # Update previous distance for progress tracking
            self.previous_distances[agent] = distance_to_goal
    
        # Update collision boundaries for agent rects
        self.collision_boundaries["agent_boundaries"] = self.agent_rects
        
        # DEBUG: see contents inside self.collision_boundaries
        #debug_collision_boundaries(self.collision_boundaries)

    
        # Compute sensor rays for all agents after updating their positions and orientations
        self.sensor_rays = compute_all_sensor_rays(
            agent_positions=self.agent_positions,  # {agent_id: (x, y)}
            agent_orientations=self.agent_orientations,  # {agent_id: orientation in radians}
            sensor_fov=self.sensor_fov,  # Field of view for sensors
            num_sensor_rays=self.num_sensor_rays,  # Number of rays per sensor
            sensor_range=self.sensor_range,  # Sensor range
        )

        
        # Compute sensor readings and store LIDAR distances for each agent as a dictionary
        self.lidar_readings = compute_lidar(
            self.sensor_rays, 
            self.collision_boundaries,   # Collision boundaries to detect
            sensor_range=self.sensor_range, 
            scale_x=self.scale_x, 
            scale_y=self.scale_y
        )

    
        
        # Check if all agents are done or if the episode has been truncated
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())
        
        # Alternative way to Check if all agents are done or if the episode has been truncated
        #terminateds["__all__"] = all(terminateds.get(agent, False) for agent in self.agents)
        #truncateds["__all__"] = all(truncateds.get(agent, False) for agent in self.agents)
    
        # Perform rendering
        if self.render_mode == "human":
            self._render_human()  # Update the screen for human view
        elif self.render_mode == "rgb_array":
            self._render_rgb_array()  # Perform rendering for RGB array but do not return it here

        # Increment step counter
#        self.current_step += 1

                 
        print("ENDING DEBUG OUTPUT FOR STEP METHOD:>")
        print(":"*100)
        
        # Return the required values

        return observations, rewards, terminateds, truncateds, infos
                                    
    

 
#-------------------------RENDERING METHOD AND HELPER UTILS------------------------------------ 

    def _render_rgb_array(self):
        """
        Render the environment in 'rgb_array' mode and return an RGB image array.
        """

        print(":"*100) 
        print("BEGINING DEBUG OUTPUT FOR RENDER_RGB METHOD:<")
                 
        # Clear the screen with the background image
        self.screen.blit(self.background, (0, 0))
        
        # Render static elements (e.g., entrances, workstations, loading docks)
        for category in ["entrances", "workstations", "loading_docks"]:
            if category in self.instance_sprites:
                for (position, dimensions), sprite in zip(getattr(self, category), self.instance_sprites[category]):
                    center_x, center_y = position
                    width, height = dimensions
    
                    # Extract the pygame.Surface from the sprite object (CustomSprite)
                    sprite_image = sprite.image
    
                    # Scale the sprite image
                    category_sprite = pygame.transform.scale(sprite_image, (int(width * self.scale_x), int(height * self.scale_y)))
    
                    # Calculate the top-left corner for blitting
                    top_left_x = (center_x - width / 2) * self.scale_x
                    top_left_y = (center_y - height / 2) * self.scale_y
    
                    # Blit the scaled image at the calculated position
                    self.screen.blit(category_sprite, (top_left_x, top_left_y))
        
        # Render obstacles
        for (position, dimensions), sprite in zip(self.obstacles, self.instance_sprites["obstacles"]):
            center_x, center_y = position
            width, height = dimensions
    
            # Extract and scale the obstacle sprite
            obstacle_sprite = pygame.transform.scale(sprite.image, (int(width * self.scale_x), int(height * self.scale_y)))
    
            # Calculate the top-left corner for blitting
            top_left_x = (center_x - width / 2) * self.scale_x
            top_left_y = (center_y - height / 2) * self.scale_y
    
            # Blit the scaled sprite at the calculated position
            self.screen.blit(obstacle_sprite, (top_left_x, top_left_y))
        
        # Render storage areas
        for (position, dimensions), sprite in zip(self.storage_areas, self.instance_sprites["storage_areas"]):
            center_x, center_y = position
            width, height = dimensions
    
            # Extract and scale the storage area sprite
            storage_area_sprite = pygame.transform.scale(sprite.image, (int(width * self.scale_x), int(height * self.scale_y)))
    
            # Calculate the top-left corner for blitting
            top_left_x = (center_x - width / 2) * self.scale_x
            top_left_y = (center_y - height / 2) * self.scale_y
    
            # Blit the scaled sprite at the calculated position
            self.screen.blit(storage_area_sprite, (top_left_x, top_left_y))
        
        # Render walls
        for (position, dimensions), wall_type, sprite in zip(self.walls, self.wall_types, self.instance_sprites["walls"]["horizontal"] + self.instance_sprites["walls"]["vertical"]):
            center_x, center_y = position
            width, height = dimensions
    
            if sprite:
                # Extract and scale the wall sprite
                wall_sprite = pygame.transform.scale(sprite.image, (int(width * self.scale_x), int(height * self.scale_y)))
    
                # Calculate the top-left corner for blitting
                top_left_x = (center_x - width / 2) * self.scale_x
                top_left_y = (center_y - height / 2) * self.scale_y
    
                # Blit the scaled sprite at the calculated position
                self.screen.blit(wall_sprite, (top_left_x, top_left_y))
        
        # Render collision boundaries as orange rectangles but ignore goal_boundaries and agent_boundaries   
        for category, boundaries in self.collision_boundaries.items():
            if category in ["goal_boundaries", "agent_boundaries"]:
                continue  # Skip these categories since their logic is already handled by agent_rects and goal_rects
            # Handle lists of rectangles
            for rect in boundaries:  # Rectangles are already scaled
                pygame.draw.rect(self.screen, (255, 165, 0), rect, 2)  # Draw orange rectangles



        # Render sensor rays and collisions
        for agent_id, lidar_data in self.lidar_readings.items():
            distances = lidar_data["distances"]
            collision_flags = lidar_data["collision_flags"]
            collision_points = lidar_data["collision_points"]
            
            for i, distance in enumerate(distances):
                ray_start, ray_end = self.sensor_rays[agent_id][i]  # Define ray_start and ray_end       
                direction = np.array([ray_end[0] - ray_start[0], ray_end[1] - ray_start[1]])
                norm_direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
        
                # Scale endpoint based on computed distance
                scaled_end = (
                    ray_start[0] + norm_direction[0] * distance,
                    ray_start[1] + norm_direction[1] * distance
                )
                
                ray_color = (0, 255, 0) if not collision_flags[i] else (255, 0, 0)  # Red for collision, green otherwise
                pygame.draw.line(self.screen, ray_color, 
                                 (ray_start[0] * self.scale_x, ray_start[1] * self.scale_y), 
                                 (scaled_end[0] * self.scale_x, scaled_end[1] * self.scale_y), 2)
                
                if collision_flags[i] and collision_points[i]:
                    pygame.draw.circle(self.screen, (255, 255, 0), 
                                       (collision_points[i][0], collision_points[i][1]), 5)  # Yellow for collision point
                    
                
        # Render goals
        for agent_id, goal_sprite in self.instance_sprites["goals"].items():
            center_x, center_y = self.goal_positions[agent_id]
            width, height = goal_sprite.image.get_size()
    
            # Calculate the top-left corner for blitting
            top_left_x = (center_x - width / (2 * self.scale_x)) * self.scale_x
            top_left_y = (center_y - height / (2 * self.scale_y)) * self.scale_y
    
            # Blit the sprite at the calculated position
            self.screen.blit(goal_sprite.image, (top_left_x, top_left_y))


        
        # Render agents
        for agent_id, agent_sprite in self.instance_sprites["agents"].items():
            center_x, center_y = self.agent_positions[agent_id]
            width, height = agent_sprite.image.get_size()
    
            # Calculate the top-left corner for blitting
            top_left_x = (center_x - width / (2 * self.scale_x)) * self.scale_x
            top_left_y = (center_y - height / (2 * self.scale_y)) * self.scale_y
    
            # Blit the sprite at the calculated position
            self.screen.blit(agent_sprite.image, (top_left_x, top_left_y))
            
        
        # Render agent and goal collision rectangles color-mapping agents to their goals
        #draw_agent_goal_rects(self.screen, self.agent_rects, self.goal_rects, self.unique_agent_goal_colors)
        draw_agent_goal_rects(
            screen=self.screen,
            agent_rects=self.agent_rects,
            goal_rects=self.goal_rects,
            unique_agent_goal_colors=self.unique_agent_goal_colors,
            terminated_agents=self.terminated_agents,
            goal_positions=self.goal_positions, 
            goal_size=self.goal_size, 
            scale_x=self.scale_x, 
            scale_y=self.scale_y
        )

        
        # Convert the Pygame surface to an RGB array
        pygame.display.flip()
        rgb_array = pygame.surfarray.array3d(self.screen)
                 
        print("ENDING DEBUG OUTPUT FOR RENDER_RGB METHOD:>")
        print(":"*100)
        # Flip the array along the y-axis to match expected image orientation
        return rgb_array.swapaxes(0, 1)



    def render(self):
        print(":"*100) 
        print("BEGINING DEBUG OUTPUT FOR RENDER METHOD:<")

        print(f"Render Mode: {self.render_mode}")
        
        # Initialize pygame if it's not already initialized
        if not pygame.get_init():
            print("Pygame is not initialized, initializing now.")
            pygame.init()
        
        # Initialize the screen only if it has not been initialized yet
        if self.screen is None:
            print("Screen not initialized. Initializing screen...")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        if self.background is None:
            print("Background not initialized.")
        else:
            print(f"Background initialized: {self.background}")
    
        # Proceed with rendering
        if self.render_mode == "human":
            self._render_human()  # Update the screen for human view
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()  # Return an image array for algorithms 

    
        # Cap the FPS at 180 (ensure clock is initialized in the setup)
        self.clock.tick(180)  # This will cap the FPS to 180   

                 
        print("ENDING DEBUG OUTPUT FOR RENDER METHOD:>")
        print(":"*100)

    def close(self):
        """Clean up resources, called when the environment is done."""
        if hasattr(self, "screen") and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None  # Optionally set to None to prevent future access

