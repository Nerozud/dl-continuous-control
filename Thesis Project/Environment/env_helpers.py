# env_helpers.py

# Required Imports
import sys

import os

import numpy as np
import math
import random
import pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
from ray.rllib.utils.spaces.space_utils import flatten_space

import torch
#import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

import logging
# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Dynamically create __all__ to include everything, including private names.
#This will programmatically populate __all__ with every name defined in the module that doesn't start with double underscores (__).
__all__ = [name for name in dir() if not name.startswith("__")]


#-----------------------------------INDUSTRIAL LAYOUT--------------------------------------------------




def get_env_layout(env_name: str):
    """Get the grid or layout of the environment for continuous space navigation.

    Args:
        env_name (str): The name of the environment.

    returns: grids[env_name]

    Grids Info:
        positions (logical units): Positons are treated as center-based coordinates convention in the respective environment.
        dimensions (logical units): Dimensions are sizes of each category in the respective environment.
        
    """
    
    grids = {
        "Factory-1": {
            "obstacles": [
                {"position": (2.0, 3.0), "dimensions": (1.0, 1.0)},  # obstacle at (2.0, 3.0) with a size of (1.0x1.0)
                {"position": (5.0, 5.0), "dimensions": (1.0, 1.0)},  # Larger obstacle
            ],
            "workstations": [
                {"position": (8.0, 2.0), "dimensions": (1.0, 1.0)},  # Workstation at (8.0, 2.0)
                {"position": (10.0, 6.0), "dimensions": (1.0, 1.0)},  # Another workstation
            ],
            "entrances": [
                {"position": (0.5, 0.5), "dimensions": (0.5, 0.5)},  # Entrance/Exit point at (0.0, 0.0)
            ],
            "walls": [
                # Horizontal walls along the top and bottom
                {"position": (7.5, 0.125), "dimensions": (15.0, 0.25)},  # Bottom wall
                {"position": (7.5, 9.875), "dimensions": (15.0, 0.25)},  # Top wall
                
                # Vertical walls along the left and right sides
                {"position": (0.125, 5.0), "dimensions": (0.25, 10.0)},  # Left wall
                {"position": (14.875, 5.0), "dimensions": (0.25, 10.0)},  # Right wall
            ],
            "field_size": (15.0, 10.0),  # Overall factory floor dimensions
        },
        "Factory-2": {
            "obstacles": [
                {"position": (1.0, 1.0), "dimensions": (2.0, 1.0)},
                {"position": (6.0, 3.0), "dimensions": (1.0, 2.0)},
                {"position": (4.0, 7.0), "dimensions": (1.0, 1.0)},
            ],
            "workstations": [
                {"position": (12.0, 3.0), "dimensions": (1.5, 1.0)},
                {"position": (7.0, 5.0), "dimensions": (1.5, 1.0)},
            ],
            "entrances": [
                {"position": (0.0, 0.0), "dimensions": (1.0, 1.0)},
            ],
            "walls": [
                # Horizontal walls along the top and bottom
                {"position": (7.5, 0.0), "dimensions": (15.0, 0.5)},  # Bottom wall
                {"position": (7.5, 10.0), "dimensions": (15.0, 0.5)},  # Top wall
                
                # Vertical walls along the left and right sides
                {"position": (0.0, 5.0), "dimensions": (0.5, 10.0)},  # Left wall
                {"position": (15.0, 5.0), "dimensions": (0.5, 10.0)},  # Right wall
            ],
            "field_size": (15.0, 12.0),
        },

        "Conflict_situation-1.3": {
            "obstacles": [
                {"position": (3.0, 2.0), "dimensions": (2.0, 4.0)},  # obstacle at (2.0, 3.0) with a size of (1.0x1.0)
            ],
            "workstations": [
                {"position": (3.0, 8.0), "dimensions": (2.0, 4.0)},  # Workstation at (8.0, 2.0)
            ],
            "walls": [
                # Horizontal walls along the top and bottom
                {"position": (3, 0.05), "dimensions": (6.0, 0.1)},  # Bottom wall
                {"position": (3, 9.95), "dimensions": (6.0, 0.1)},  # Top wall
                
                # Vertical walls along the left and right sides
                {"position": (0.05, 5.0), "dimensions": (0.1, 10.0)},  # Left wall
                {"position": (5.95, 5.0), "dimensions": (0.1, 10.0)},  # Right wall
            ],
            "field_size": (6.0, 10.0),  # Overall factory floor dimensions
        },
        
        "Warehouse-1": {
            "obstacles": [
                {"position": (3.0, 4.0), "dimensions": (2.0, 1.5)},  # Large storage rack
                {"position": (8.0, 5.0), "dimensions": (1.0, 3.0)},  # Another storage rack
            ],
            "storage_areas": [
                {"position": (1.0, 1.0), "dimensions": (3.0, 3.0)},  # Storage area 1
                {"position": (10.0, 2.0), "dimensions": (4.0, 4.0)},  # Storage area 2
            ],
            "loading_docks": [
                {"position": (0.0, 9.0), "dimensions": (1.0, 1.0)},  # Dock 1
                {"position": (12.0, 9.0), "dimensions": (1.0, 1.0)},  # Dock 2
            ],
            "walls": [
                # Horizontal walls along the top and bottom
                {"position": (7.5, 0.0), "dimensions": (15.0, 0.5)},  # Bottom wall
                {"position": (7.5, 10.0), "dimensions": (15.0, 0.5)},  # Top wall
                
                # Vertical walls along the left and right sides
                {"position": (0.0, 5.0), "dimensions": (0.5, 10.0)},  # Left wall
                {"position": (15.0, 5.0), "dimensions": (0.5, 10.0)},  # Right wall
            ],
            "field_size": (15.0, 12.0),
        },
        "Warehouse-2": {
            "obstacles": [
                {"position": (2.0, 3.0), "dimensions": (1.0, 1.0)},
                {"position": (6.0, 6.0), "dimensions": (2.0, 2.0)},
            ],
            "storage_areas": [
                {"position": (1.0, 1.0), "dimensions": (4.0, 3.0)},
                {"position": (9.0, 2.0), "dimensions": (5.0, 3.0)},
            ],
            "loading_docks": [
                {"position": (0.0, 10.0), "dimensions": (1.0, 1.5)},
                {"position": (14.0, 10.0), "dimensions": (1.0, 1.5)},
            ],
            "walls": [
                # Horizontal walls along the top and bottom
                {"position": (7.5, 0.0), "dimensions": (15.0, 0.5)},  # Bottom wall
                {"position": (7.5, 10.0), "dimensions": (15.0, 0.5)},  # Top wall
                
                # Vertical walls along the left and right sides
                {"position": (0.0, 5.0), "dimensions": (0.5, 10.0)},  # Left wall
                {"position": (15.0, 5.0), "dimensions": (0.5, 10.0)},  # Right wall
            ],
            "field_size": (16.0, 12.0),
        },
    }

    if env_name not in grids:
        raise ValueError(f"Environment '{env_name}' not found!")

    return grids[env_name]


#---------------------------AGENT AND GOAL POSITIONS UTILLS----------------------------------------

def get_predefined_start_positions(env_name: str, num_agents: int):
    """
    Get predefined start positions for agents in specific environments.

    Args:
        env_name (str): The name of the environment.
        num_agents (int): The number of agents.

    Returns:
        dict: A dictionary where keys are agent names (e.g., 'agent_0') and values are predefined start positions (tuples).

    Raises:
        ValueError: If the environment name is unknown or if the number of agents exceeds available predefined positions.
    """

    # Define predefined starting positions for each environment
    predefined_positions = {
        "Factory-1": [
            (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)  # Example positions
        ],
        "Factory-2": [
            (1.0, 1.0), (2.0, 3.0), (3.0, 5.0), (4.0, 6.0), (6.0, 7.0)  # Example positions
        ],
        "Warehouse-1": [
            (1.0, 8.0), (2.0, 9.0), (3.0, 10.0), (4.0, 11.0), (5.0, 12.0)  # Example positions
        ],
        "Warehouse-2": [
            (1.0, 7.0), (2.0, 6.0), (3.0, 5.0), (4.0, 4.0), (5.0, 3.0)  # Example positions
        ]
    }

    # Check if the environment has predefined positions
    if env_name not in predefined_positions:
        raise ValueError(f"Environment '{env_name}' does not have predefined positions!")

    # Retrieve positions for the specified environment
    positions = predefined_positions[env_name]

    # Ensure there are enough predefined positions for the number of agents
    if num_agents > len(positions):
        raise ValueError(f"Number of agents ({num_agents}) exceeds available predefined positions in {env_name}.")

    # Create a dictionary mapping agent names to predefined positions
    agent_positions = {f"agent_{i}": positions[i] for i in range(num_agents)}
    
 
    return agent_positions


def compute_optimal_values(object_size, min_grid_resolution=10, safety_factor=0.1):
    object_width, object_height = object_size

    # Compute scale to ensure at least min_grid_resolution per goal width
    optimal_scale = int(min_grid_resolution / min(object_width, object_height))

    # Compute margin to ensure goals stay clear of obstacles
    optimal_margin = (min(object_width, object_height) / 2) + (safety_factor * min(object_width, object_height))

    return optimal_scale, optimal_margin

def get_restricted_regions(env_layout):
    """
    Extracts all restricted areas (obstacles, workstations, walls) as bounding boxes.

    Args:
        env_layout (dict): The environment layout containing obstacles, workstations, walls.

    Returns:
        list: A list of restricted bounding boxes [(x_min, y_min, x_max, y_max), ...]
    """
    restricted_regions = []

    
    for category in ["obstacles", "workstations", "entrances", "storage_areas", "loading_docks", "walls"]:
        for item in env_layout.get(category, []):
            center_x, center_y = item["position"]
            width, height = item["dimensions"]

            # Convert center-based position to bounding box (top-left, bottom-right)
            x_min = center_x - width / 2
            y_min = center_y - height / 2
            x_max = center_x + width / 2
            y_max = center_y + height / 2

            restricted_regions.append((x_min, y_min, x_max, y_max))
    
    return restricted_regions



def is_valid_position(center, size, restricted_regions, existing_positions, min_dist=0.5):
    """
    Checks if a position collides with restricted areas or other agents.

    Args:
        center (tuple): The (x, y) center of the position.
        size (tuple): The (width, height) of the object.
        restricted_regions (list): List of bounding boxes [(x_min, y_min, x_max, y_max), ...]
        existing_positions (list): List of existing centers to avoid overlap.
        min_dist (float): Minimum distance between agent positions.

    Returns:
        bool: True if the position is valid (non-colliding), False otherwise.
    """
    x, y = center
    width, height = size

    # Convert center to bounding box
    x_min, y_min = x - width / 2, y - height / 2
    x_max, y_max = x + width / 2, y + height / 2

    # Check for collisions with restricted regions
    for rx_min, ry_min, rx_max, ry_max in restricted_regions:
        if not (x_max <= rx_min or x_min >= rx_max or y_max <= ry_min or y_min >= ry_max):
            return False  # Collision detected

    # Ensure agents are not too close to each other
    for ex, ey in existing_positions:
        if ((x - ex) ** 2 + (y - ey) ** 2) ** 0.5 < min_dist:
            return False  # Too close to another agent

    return True  # No collisions detected

def get_random_start_positions(env_name, num_agents, object_size, margin, scale, min_dist=0.5):
    """
    Generate non-colliding positions for agents using center-based coordinates.

    Args:
        env_name (str): The name of the environment.
        num_agents (int): The number of agents.
        object_size (tuple): The (width, height) size of the object.
        margin (float): Minimum distance from obstacles and boundaries.
        scale (int): Pixels per logical unit.
        min_dist (float): Minimum distance between agents.

    Returns:
        dict: A dictionary mapping agent IDs to their positions (center-based).
    """
    env_layout = get_env_layout(env_name)
    field_size = env_layout["field_size"]
    restricted_regions = get_restricted_regions(env_layout)

    width, height = object_size
    valid_positions = []
    existing_positions = []  # Store assigned positions

    # Generate candidate positions
    for x in range(int((margin + width / 2) * scale), int((field_size[0] - margin - width / 2) * scale)):
        for y in range(int((margin + height / 2) * scale), int((field_size[1] - margin - height / 2) * scale)):
            center = (x / scale, y / scale)
            if is_valid_position(center, object_size, restricted_regions, existing_positions, min_dist):
                valid_positions.append(center)

    if num_agents > len(valid_positions):
        raise ValueError(f"Not enough valid positions for {num_agents} agents in '{env_name}'.")

    # Select agent positions randomly from valid ones
    agent_positions = random.sample(valid_positions, num_agents)
    
    # Store chosen positions
    for pos in agent_positions:
        existing_positions.append(pos)


    # Return center-based positions
    return {f"agent_{i}": pos for i, pos in enumerate(agent_positions)}



def get_predefined_goal_positions(env_name: str, num_agents: int):
    """
    Get predefined goal positions for agents based on the environment.

    Args:
        env_name (str): The name of the environment.
        num_agents (int): The number of agents.

    Returns:
        dict: Keys are agent names (e.g., 'agent_0') and values are the goal positions (tuples).

    Raises:
        ValueError: If the environment name is unknown or if the number of agents exceeds available goal positions.
    """
    
    predefined_goals = {
        "Factory-1": [(14.0, 9.0), (14.0, 5.0)],  # Example goals for Factory-1
        "Factory-2": [(14.0, 11.0), (13.0, 5.0)],  # Example goals for Factory-2
        "Warehouse-1": [(14.0, 10.0), (12.0, 8.0)],  # Example goals for Warehouse-1
        "Warehouse-2": [(15.0, 11.0), (10.0, 9.0)],  # Example goals for Warehouse-2
    }

    if env_name not in predefined_goals:
        raise ValueError(f"Environment '{env_name}' not found!")

    # Get the predefined goals for the given environment
    goals = predefined_goals[env_name]

    # If there are more agents than predefined goals, raise an error
    if num_agents > len(goals):
        raise ValueError(f"Not enough predefined goal positions for {num_agents} agents.")

    # Create a dictionary of agent goal positions
    goal_positions = {f"agent_{i}": goals[i] for i in range(num_agents)}

    return goal_positions





def is_valid_goal(goal_center, goal_size, restricted_regions):
    """
    Checks if a goal position collides with any restricted area.

    Args:
        goal_center (tuple): The (x, y) center of the goal.
        goal_size (tuple): The (width, height) of the goal.
        restricted_regions (list): List of bounding boxes [(x_min, y_min, x_max, y_max), ...]

    Returns:
        bool: True if the goal position is valid (non-colliding), False otherwise.
    """
    goal_x, goal_y = goal_center
    goal_width, goal_height = goal_size

    # Convert goal center to bounding box (top-left and bottom-right)
    goal_x_min = goal_x - goal_width / 2
    goal_y_min = goal_y - goal_height / 2
    goal_x_max = goal_x + goal_width / 2
    goal_y_max = goal_y + goal_height / 2

    # Check if goal overlaps with any restricted region
    for x_min, y_min, x_max, y_max in restricted_regions:
        if not (goal_x_max <= x_min or goal_x_min >= x_max or
                goal_y_max <= y_min or goal_y_min >= y_max):
            return False  # Collision detected

    return True  # No collision

def get_random_goal_positions(env_name: str, num_agents: int, goal_size: tuple, margin: float, scale: int):
    """
    Generate non-colliding goal positions for agents using center-based coordinates.

    Args:
        env_name (str): The name of the environment.
        num_agents (int): The number of agents.
        goal_size (tuple): The (width, height) size of the goal.
        margin (float): Minimum distance from obstacles and boundaries.
        scale (int): Pixels per logical unit.

    Returns:
        dict: A dictionary mapping agent IDs to their goal positions (center-based).
    """
    env_layout = get_env_layout(env_name)
    field_size = env_layout["field_size"]
    restricted_regions = get_restricted_regions(env_layout)

    goal_width, goal_height = goal_size
    valid_positions = []

    # Generate potential goal center positions
    for x in range(int((margin + goal_width / 2) * scale), int((field_size[0] - margin - goal_width / 2) * scale)):
        for y in range(int((margin + goal_height / 2) * scale), int((field_size[1] - margin - goal_height / 2) * scale)):
            goal_center = (x / scale, y / scale)

            if is_valid_goal(goal_center, goal_size, restricted_regions):
                valid_positions.append(goal_center)

    if num_agents > len(valid_positions):
        raise ValueError(f"Not enough valid positions for {num_agents} agents in '{env_name}'.")

    # Select goal positions randomly from valid ones
    goal_positions = random.sample(valid_positions, num_agents)

    
    # Return center-based goal positions
    return {f"agent_{i}": pos for i, pos in enumerate(goal_positions)}









#----------------------------OBSERVATION AND ACTION SPACES UTILS--------------------------------------------------------


def generate_space(
    agents, field_size, obstacles, workstations, entrances, storage_areas,
    loading_docks, walls, max_speed, sensor_range, num_sensor_rays,
    collision_boundaries, num_agents, max_timesteps
):
    """
    Generate unified observation and action spaces for multi-agent environments.
    """
    sensor_range = max(sensor_range, 1)
    
    # Define individual agent action space
    action_spec = spaces.Box(
        low=np.array([-np.pi, 0.0]),
        high=np.array([np.pi, max_speed]),
        shape=(2,),
        dtype=np.float32
    )
    
    # Observation space for a single agent

    common_dict = {}

    # Handle common dict categories dynamically
    for category_name, category_data in [
        ('obstacles', obstacles),
        ('workstations', workstations),
        ('entrances', entrances),
        ('storage_areas', storage_areas),
        ('loading_docks', loading_docks),
        ('walls', walls)
    ]:
        if category_data:
            # Flatten positions and dimensions into a consistent format
            flat_data = [
                (pos[0], pos[1], dim[0], dim[1])  # Extract elements from (position, dimensions) tuple
                for pos, dim in category_data
            ]
            common_dict[category_name] = spaces.Box(
                low=np.zeros((len(flat_data) * 4,)),
                high=np.tile(np.array([field_size[0], field_size[1], field_size[0], field_size[1]]), len(flat_data)),
                shape=(len(flat_data) * 4,),
                dtype=np.float32
            )
    
    # Define the observation space for the agent
    observation_space = spaces.Dict({
        "common": spaces.Dict(common_dict),
        "position": spaces.Box(
            low=np.array([1.0, 1.0]),
            high=np.array([field_size[0] - 1, field_size[1] - 1]),
            shape=(2,),
            dtype=np.float32
        ),
        "velocity": spaces.Box(
            low=np.array([-max_speed]),
            high=np.array([max_speed]),
            shape=(1,),
            dtype=np.float32
        ),
        "goal": spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([field_size[0]-1, field_size[1]-1]),
            shape=(2,),
            dtype=np.float32
        ),
        "orientation": spaces.Box(
            low=np.array([-np.pi]),
            high=np.array([np.pi]),
            shape=(1,),
            dtype=np.float32
        ),
        "local_sensor_observations": spaces.Box(
            low=np.zeros(num_sensor_rays, dtype=np.uint8),
            high=np.full(num_sensor_rays, sensor_range, dtype=np.uint8),
            shape=(num_sensor_rays,),
            dtype=np.uint8
        ),

        "other_agents_relative_positions": spaces.Box(
            low=np.zeros((num_agents - 1) * 2),
            high=np.tile(np.array([field_size[0] - 1, field_size[1] - 1]), num_agents - 1),
            shape=((num_agents - 1) * 2,),
            dtype=np.float32
        ),

        "other_collision_categories_to_avoid_distances": spaces.Box(
            low=np.zeros(len(collision_boundaries.keys()), dtype=np.float32),
            high=np.full(len(collision_boundaries.keys()), sensor_range, dtype=np.float32),
            shape=(len(collision_boundaries.keys()),),
            dtype=np.float32
        ),
       # "time_step": spaces.Discrete(max_timesteps),
        #"status": spaces.Discrete(3),
       # "action_mask": spaces.MultiBinary(2)
    })

    
    # Create spaces for all agents
    #observation_space = spaces.Dict({
    #    f"agent_{i}": single_observation_space for i in range(num_agents)
    #})
    #action_space = spaces.Dict({
    #    f"agent_{i}": action_spec for i in range(num_agents)
    #})

    #flattened_observation_space = flatten_space(observation_space)
    #flattened_action_spec = flatten_space(action_spec)
    
    flattened_observation_space = spaces.flatten_space(observation_space)
    #flattened_action_spec = spaces.flatten_space(action_spec)
    
    return flattened_observation_space, action_spec


def generate_spaces(agents, field_size, obstacles, workstations, entrances, 
                    storage_areas, loading_docks, walls, max_speed, 
                    sensor_range, num_sensor_rays, collision_boundaries, 
                    num_agents, max_timesteps):
    """
    Generate observation and action spaces for a multi-agent environment.

    Args:
        agents: List of agent IDs.
        field_size: Tuple representing the size of the field (width, height).
        obstacles, workstations, entrances, storage_areas, loading_docks, walls: Lists of object positions.
        max_speed: Maximum speed for agents.
        sensor_range: Maximum sensor range.
        num_sensor_rays: Number of rays for local sensors.
        collision_boundaries: Dictionary of collision categories and their boundaries.
        num_agents: Total number of agents.
        max_timesteps: Maximum number of timesteps in the environment.

    Returns:
        Tuple (observation_spaces, action_spaces) for all agents.
    """
    # Ensure sensor range is at least 1 for stability
    sensor_range = max(sensor_range, 1)

    # Action space (normalized continuous, [-1, 1] for direction, [0, 1] for acceleration/braking)
    action_spaces = {
        agent: spaces.Box(
            low=np.array([-np.pi, 0.0]),
            high=np.array([np.pi, max_speed]),
            shape=(2,),
            dtype=np.float32
        )
        for agent in agents
    }

    
    #observation_spaces = {}
    for agent in agents:
        common_dict = {}

        # Handle common dict categories dynamically
        for category_name, category_data in [
            ('obstacles', obstacles),
            ('workstations', workstations),
            ('entrances', entrances),
            ('storage_areas', storage_areas),
            ('loading_docks', loading_docks),
            ('walls', walls)
        ]:
            if category_data:
                # Flatten positions and dimensions into a consistent format
                flat_data = [
                    (pos[0], pos[1], dim[0], dim[1])  # Extract elements from (position, dimensions) tuple
                    for pos, dim in category_data
                ]
                common_dict[category_name] = spaces.Box(
                    low=np.zeros((len(flat_data) * 4,)),
                    high=np.tile(np.array([field_size[0], field_size[1], field_size[0], field_size[1]]), len(flat_data)),
                    shape=(len(flat_data) * 4,),
                    dtype=np.float32
                )
        
        # Define the observation space for the agent
        single_observation_space = spaces.Dict({
            "common": spaces.Dict(common_dict),
            "position": spaces.Box(
                low=np.array([1.0, 1.0]),
                high=np.array([field_size[0] - 1, field_size[1] - 1]),
                shape=(2,),
                dtype=np.float32
            ),
            "velocity": spaces.Box(
                low=np.array([-max_speed]),
                high=np.array([max_speed]),
                shape=(1,),
                dtype=np.float32
            ),
            "goal": spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([field_size[0], field_size[1]]),
                shape=(2,),
                dtype=np.float32
            ),
            "orientation": spaces.Box(
                low=np.array([-np.pi]),
                high=np.array([np.pi]),
                shape=(1,),
                dtype=np.float32
            ),
            "local_sensor_observations": spaces.Box(
                low=np.zeros(num_sensor_rays, dtype=np.uint8),
                high=np.full(num_sensor_rays, sensor_range, dtype=np.uint8),
                shape=(num_sensor_rays,),
                dtype=np.uint8
            ),

            "other_agents_relative_positions": spaces.Box(
                low=np.zeros((num_agents - 1) * 2),
                high=np.tile(np.array([field_size[0] - 1, field_size[1] - 1]), num_agents - 1),
                shape=((num_agents - 1) * 2,),
                dtype=np.float32
            ),

            "other_collision_categories_to_avoid_distances": spaces.Box(
                low=np.zeros(len(collision_boundaries.keys()), dtype=np.float32),
                high=np.full(len(collision_boundaries.keys()), sensor_range, dtype=np.float32),
                shape=(len(collision_boundaries.keys()),),
                dtype=np.float32
            ),
           # "time_step": spaces.Discrete(max_timesteps),
            #"status": spaces.Discrete(3),
           # "action_mask": spaces.MultiBinary(2)
        })

        
        observation_spaces = spaces.Dict({
            f"agent_{i}": single_observation_space for i in range(num_agents)
        })    

    return observation_spaces, action_spaces



#----------------------------RENDERING UTILS--------------------------------------------------------


def _render_human(self):
    """
    Render the environment in 'human' mode.
    """

    print(":"*100) 
    print("BEGINING DEBUG OUTPUT FOR RENDER_HUMAN METHOD:<")
             
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


    """""
    # Render sensor rays and collisions
    for agent, agent_rays in self.sensor_rays.items():  # Since `self.sensor_rays` stores agents' rays
        for ray_start, ray_end in agent_rays:
            # Scale ray coordinates
            start = (ray_start[0] * self.scale_x, ray_start[1] * self.scale_y)
            end = (ray_end[0] * self.scale_x, ray_end[1] * self.scale_y)
    
            # Check for intersections with collision boundaries
            collision_detected = False
            collision_point = None
    
            for category, boundaries in self.collision_boundaries.items():
                if isinstance(boundaries, dict):  # Handle dictionary-based categories
                    for agent_id, rect in boundaries.items():
                        # Skip the current agent's own goal or agent's rectangle
                        if agent_id == agent:
                            continue
                        # Detect collision with other agents' goal/agent boundary areas
                        if rect.clipline(start, end):
                            collision_detected = True
                            collision_point = rect.clipline(start, end)[0]
                            break
                elif isinstance(boundaries, list):  # Handle list-based categories boundary areas
                    for rect in boundaries:
                        if rect.clipline(start, end):
                            collision_detected = True
                            collision_point = rect.clipline(start, end)[0]
                            break
    
                # If collision has already been detected in one of the categories, break out of the loop
                if collision_detected:
                    break
    
            # Draw rays
            ray_color = (0, 255, 0) if not collision_detected else (255, 0, 0)  # Green for clear, Red for collision
            pygame.draw.line(self.screen, ray_color, start, end, 2)
    
            # Draw collision point if detected
            if collision_detected and collision_point:
                pygame.draw.circle(self.screen, (255, 255, 0), collision_point, 5)  # Yellow dot for collision

    """""

    
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
        terminated_agents=self.terminated_agents
    )

                
    print("ENDING DEBUG OUTPUT FOR RENDER_HUMAN METHOD:>")
    print(":"*100)

    # Update the display
    pygame.display.flip()
    self.clock.tick(self.metadata["render_fps"])



#-------------------------------SPRITES, IMAGES AND RENDERING HELPER UTILS-----------------------------------------------

def compute_screen_size(field_size, target_resolution=(1100, 900), min_scale=50):
    """
    Computes optimal screen width and height for a given environment field size.

    Args:
        field_size (tuple): The (width, height) of the environment in logical units.
        target_resolution (tuple): Approximate desired screen resolution (width, height).
        min_scale (int): Minimum scaling factor to maintain visibility.

    Returns:
        tuple: (screen_width, screen_height, scale)
    """
    field_width, field_height = field_size

    # Compute scale to fit within target resolution
    scale_x = target_resolution[0] / field_width
    scale_y = target_resolution[1] / field_height

    # Choose the smallest scale to fit within the screen while keeping minimum visibility
    scale = max(min(scale_x, scale_y), min_scale)

    # Compute screen size
    screen_width = int(field_width * scale)
    screen_height = int(field_height * scale)

    return screen_width, screen_height, scale


def get_sprite_sizes(env_layout, agent_size, goal_size, scale_x, scale_y):
    """
    Returns a dictionary of sprite sizes for different environment components, scaled accordingly.

    Parameters:
    env_layout (dict): Layout of the environment, including obstacles, walls, workstations, etc.
    agent_size (tuple): Size of the agent in logical units (width, height).
    goal_size (tuple): Size of the goal in logical units (width, height).
    scale_x (float): Scaling factor for width.
    scale_y (float): Scaling factor for height.

    Returns:
    dict: A dictionary mapping environment components to their sprite sizes in pixels.

    Conditions:
        If a category has more than one dictionary in that category's list inside the environment layout dictionary, 
        the first dimension in each category's dictionary is used to create the sprite size for all sprites in that category.

    Usage:
        used by the "init" method inside the con class for creating self.sprite_sizes dictionary.   
    
    """
    def scale_dimensions(logical_size):
        """Helper function to scale logical dimensions to pixel dimensions."""
        width, height = logical_size
        return int(width * scale_x), int(height * scale_y)

    # Retrieve dimensions with a fallback to default sizes
    def get_dimensions(category, index, default):
        if env_layout.get(category) and len(env_layout[category]) > index:
            return env_layout[category][index].get("dimensions", default)
        return default

    return {
        "agent": scale_dimensions(agent_size),
        "goal": scale_dimensions(goal_size),
        "obstacle": scale_dimensions(get_dimensions("obstacles", 0, (0.0, 0.0))),
        "entrance": scale_dimensions(get_dimensions("entrances", 0, (0.0, 0.0))),
        "workstation": scale_dimensions(get_dimensions("workstations", 0, (0.0, 0.0))),
        "storage_area": scale_dimensions(get_dimensions("storage_area", 0, (0.0, 0.0))),
        "loading_dock": scale_dimensions(get_dimensions("loading_dock", 0, (0.0, 0.0))),
        "wall_horizontal": scale_dimensions(get_dimensions("walls", 0, (0.0, 0.0))),
        "wall_vertical": scale_dimensions(get_dimensions("walls", 2, (0.0, 0.0))),
    }




def preprocess_images(image_paths, target_size, transparency_color=None):
    """
    Preprocesses images for use as sprites in the environment.

    Args:
        image_paths (dict): A dictionary mapping sprite types (e.g., 'agent', 'obstacle') to file paths.
        target_size (tuple): The target size (width, height) to which the images should be resized.
        color_key (tuple or None): If specified, a color (R, G, B) to be treated as transparent.

    Returns:
        dict: A dictionary mapping sprite types to preprocessed Pygame surfaces.
    """
    processed_images = {}

    for name, path in image_paths.items():
        # Load the image
        try:
            image = pygame.image.load(path).convert_alpha()  # Ensure image has an alpha channel
        except pygame.error as e:
            print(f"Error loading image {path}: {e}")
            continue

        # Resize the image
        image = pygame.transform.scale(image, target_size)

        # Handle transparency
        if transparency_color:
            image.set_colorkey(transparency_color)  # Set transparency color
            
        # Store the preprocessed image
        processed_images[name] = image

    return processed_images


class CustomSprite(pygame.sprite.Sprite):
    def __init__(self, image, position):
        """
        Custom Sprite class that associates an image with a rectangle.
        
        Args:
            image (pygame.Surface): The image to display.
            position (tuple): The initial position of the sprite (center of the sprite).
        """
        super().__init__()

        self.image = image
        # Adjust the rectangle to be center-based
        self.rect = self.image.get_rect(center=position)
        
        '''
        # Print CustomSprite Debug Information
        print("=" * 100)
        print("CustomSprite of type <pygame.sprite.Sprite> contents created with CustomSprite class:")
        # Inspect self.rect
        if self.rect is not None:
            print("Rect attributes:")
            for key in ['x', 'y', 'width', 'height']:
                print(f"  {key}: {getattr(self.rect, key)}")
        else:
            print("self.rect is None.")
        
        # Inspect self.image
        if self.image is not None:
            print("\nImage attributes:")
            print(f"  Type of self.image: {type(self.image)}")
            print(f"  Size (width, height): {self.image.get_width()}, {self.image.get_height()}")
            print(f"  Pixel format: {self.image.get_bitsize()} bits per pixel")
            print(f"  Is locked: {'Yes' if self.image.get_locked() else 'No'}")
            print(f"  Color key: {self.image.get_colorkey()}")
            print(f"  Alpha transparency: {self.image.get_alpha()}")
        else:
            print("self.image is None.")
        print("-" * 100)
        
        '''
def _initialize_sprites(
    agent_positions, 
    goal_positions, 
    obstacles, 
    horizontal_walls, 
    vertical_walls, 
    entrances=None, 
    workstations=None, 
    storage_areas=None, 
    loading_docks=None, 
    sprites=None, 
    scale_x=1.0, 
    scale_y=1.0
):
    """
    Initialize sprite mappings for agents, goals, and environment objects.
    
    Parameters:
        agent_positions (dict): Dictionary of agent positions.
        goal_positions (dict): Dictionary of goal positions.
        obstacles (list): List of obstacle positions.
        horizontal_walls (list): List of horizontal wall positions.
        vertical_walls (list): List of vertical wall positions.
        entrances (list, optional): List of entrance positions.
        workstations (list, optional): List of workstation positions.
        storage_areas (list, optional): List of storage area positions.
        loading_docks (list, optional): List of loading dock positions.
        sprites (dict): Dictionary of sprite images for various objects.
        scale_x (float): Horizontal scaling factor.
        scale_y (float): Vertical scaling factor.
    
    Returns:
        dict: A dictionary mapping object categories to their respective sprites.
    """
    return {
        "agents": {
            agent: CustomSprite(
                image=sprites["agent"][f"agent_{i % len(sprites['agent'])}"],
                position=(
                    float(agent_position[0]) * scale_x,
                    float(agent_position[1]) * scale_y
                )
            )
            for i, (agent, agent_position) in enumerate(agent_positions.items())
            if isinstance(agent_position, (np.ndarray, tuple, list)) and len(agent_position) == 2
        },
        "goals": {
            agent: CustomSprite(
                image=sprites["goal"][f"goal_{i % len(sprites['goal'])}"],
                position=(
                    float(goal_positions[agent][0]) * scale_x,
                    float(goal_positions[agent][1]) * scale_y
                )
            )
            for i, agent in enumerate(agent_positions)
            if isinstance(goal_positions[agent], (tuple, list)) and len(goal_positions[agent]) == 2
        },
        "obstacles": [
            CustomSprite(
                image=sprites["obstacle"][f"obstacle_{i % len(sprites['obstacle'])}"],
                position=(
                    float(obstacle[0][0]) * scale_x,
                    float(obstacle[0][1]) * scale_y
                )
            )
            for i, obstacle in enumerate(obstacles)
            if isinstance(obstacle, (tuple, list)) and len(obstacle) == 2
        ],
        "walls": {
            "horizontal": [
                CustomSprite(
                    image=sprites["wall_horizontal"][f"wall_horizontal_{i % len(sprites['wall_horizontal'])}"],
                    position=(
                        float(wall[0][0]) * scale_x,
                        float(wall[0][1]) * scale_y
                    )
                )
                for i, wall in enumerate(horizontal_walls)
                if isinstance(wall, (tuple, list)) and len(wall) == 2
            ],
            "vertical": [
                CustomSprite(
                    image=sprites["wall_vertical"][f"wall_vertical_{i % len(sprites['wall_vertical'])}"],
                    position=(
                        float(wall[0][0]) * scale_x,
                        float(wall[0][1]) * scale_y
                    )
                )
                for i, wall in enumerate(vertical_walls)
                if isinstance(wall, (tuple, list)) and len(wall) == 2
            ]
        },
        "entrances": [
            CustomSprite(
                image=sprites["entrance"][f"entrance_{i % len(sprites['entrance'])}"],
                position=(
                    float(entrance[0][0]) * scale_x,
                    float(entrance[0][1]) * scale_y
                )
            )
            for i, entrance in enumerate(entrances or [])
            if isinstance(entrance, (tuple, list)) and len(entrance) == 2
        ],
        "workstations": [
            CustomSprite(
                image=sprites["workstation"][f"workstation_{i % len(sprites['workstation'])}"],
                position=(
                    float(workstation[0][0]) * scale_x,
                    float(workstation[0][1]) * scale_y
                )
            )
            for i, workstation in enumerate(workstations or [])
            if isinstance(workstation, (tuple, list)) and len(workstation) == 2
        ],
        "storage_areas": [
            CustomSprite(
                image=sprites["storage_area"][f"storage_area_{i % len(sprites['storage_area'])}"],
                position=(
                    float(storage_area[0][0]) * scale_x,
                    float(storage_area[0][1]) * scale_y
                )
            )
            for i, storage_area in enumerate(storage_areas or [])
            if isinstance(storage_area, (tuple, list)) and len(storage_area) == 2
        ],
        "loading_docks": [
            CustomSprite(
                image=sprites["loading_dock"][f"loading_dock_{i % len(sprites['loading_dock'])}"],
                position=(
                    float(loading_dock[0][0]) * scale_x,
                    float(loading_dock[0][1]) * scale_y
                )
            )
            for i, loading_dock in enumerate(loading_docks or [])
            if isinstance(loading_dock, (tuple, list)) and len(loading_dock) == 2
        ],
    }




def draw_agent_goal_rects(screen, agent_rects, goal_rects, unique_agent_goal_colors, 
                          terminated_agents, goal_positions, goal_size, scale_x, scale_y):
    """
    Draw rectangles around agents and their goals, with a unique color mapping each agent to its goal.
    Terminated agents are visually distinct.

    Args:
        screen (pygame.Surface): The screen surface to draw on.
        agent_rects (dict): Dictionary mapping agent IDs to their respective pygame.Rect.
        goal_rects (dict): Dictionary mapping agent IDs to their respective goal pygame.Rect.
        unique_agent_goal_colors (dict): Mapping of agent IDs to unique RGB colors.
        terminated_agents (set): Set of agent IDs that have terminated (reached their goals).
    """
    def _make_goal_rects(goal_positions, goal_size, scale_x, scale_y):
        """
        Create a dictionary of rects for goals to prevent agents from occupying other agents' goals.
    
        Args:
            goal_positions (dict): Dictionary with agent IDs as keys and (x, y) goal positions as values.
            goal_size (tuple): Size of each goal in logical units (width, height).
            scale_x (float): Scaling factor for x-axis.
            scale_y (float): Scaling factor for y-axis.
    
        Returns:
            dict: Dictionary of agent IDs mapped to their corresponding pygame.Rect objects.
    
        Usage:
            used by "setup_collision_boundaries(env_layout, goal_positions, goal_size, scale_x, scale_y)" function.
            used by "init" method inside the ContinuousPathfindingEnv class for "self.goal_rects".
            
        """
        rects = {}
       
        for agent_id, position in goal_positions.items():  # Iterate over the dictionary
            # Validate the goal position
            if not isinstance(position, (tuple, list, np.ndarray)) or len(position) != 2:
                raise ValueError(f"Invalid goal position for {agent_id}: {position}, expected a 2D tuple, list, or ndarray.")
            
            # Unpack position
            center_x, center_y = position
    
            # Scale dimensions and center position
            scaled_width = goal_size[0] * scale_x
            scaled_height = goal_size[1] * scale_y
            scaled_center_x = center_x * scale_x
            scaled_center_y = center_y * scale_y
    
            # Create a pygame.Rect with scaled dimensions and center position
            rect = pygame.Rect(0, 0, scaled_width, scaled_height)
            rect.center = (scaled_center_x, scaled_center_y)
    
            # Map agent ID to its rect
            rects[agent_id] = rect
        
        # Validate all rects before returning
        assert all(isinstance(rect, pygame.Rect) for rect in rects.values()), "All elements in rects must be pygame.Rect"
        
        return rects 

    simulation_goal_rects = _make_goal_rects(goal_positions, goal_size, scale_x, scale_y)
        
    
    for agent_id, agent_rect in agent_rects.items():
        
        simulation_goal_rect = simulation_goal_rects.get(agent_id)

        # Determine the color for the agent
        if agent_id in terminated_agents and simulation_goal_rect and agent_rect == simulation_goal_rect:
            # Distinct color/style for terminated agents (e.g., green and filled)
            agent_color = (255, 0, 0)  # Red for terminated agents
            agent_outline_width = 0    # Filled rectangle
        else:
            # Normal color/style for active agents
            agent_color = unique_agent_goal_colors.get(agent_id, (255, 255, 255))  # Default to white
            agent_outline_width = 2    # Outline only
            


        # Draw the rectangle around the agent
        pygame.draw.rect(screen, agent_color, agent_rect, agent_outline_width)

        # Create and draw the corresponding goal rectangle
        #simulation_goal_rect = simulation_goal_rects.get(agent_id)
        if simulation_goal_rect:  # Ensure goal_rect exists before trying to draw
            # Use the same color as the agent, but always outlined for the goal
            pygame.draw.rect(screen, agent_color, simulation_goal_rect, 2)  # Width 2 for outline
            
        """
        # Retrieve and draw the corresponding goal rectangle
        goal_rect = goal_rects.get(agent_id)
        if goal_rect:  # Ensure goal_rect exists before trying to draw
            # Use the same color as the agent, but always outlined for the goal
            pygame.draw.rect(screen, agent_color, goal_rect, 2)  # Width 2 for outline
            """


def generate_unique_color(existing_colors, min_distance=50):
    """Generate a unique color that's at least `min_distance` away from existing colors."""
    while True:
        color = tuple(random.randint(0, 255) for _ in range(3))
        # Check if this color is far enough from all existing colors
        if all(np.linalg.norm(np.array(color) - np.array(existing)) >= min_distance for existing in existing_colors):
            return color


#-------------------------------LIDAR SENSING FOR COLLISION BOUNDARY DETECTION -> HELPER UTILS-----------------------------------------


def compute_all_sensor_rays(agent_positions, agent_orientations, sensor_fov, num_sensor_rays, sensor_range):
    """
    Compute sensor rays for all agents in the environment.

    Args:
        agent_positions (dict): A dictionary of agent positions {agent_id: (x, y)}.
        agent_orientations (dict): A dictionary of agent orientations {agent_id: orientation}.
        sensor_fov (float): Field of view of the sensor in radians.
        num_sensor_rays (int): Number of rays per agent within the field of view.
        sensor_range (float): Maximum range of the sensor.
        collision_boundaries (dict): Dictionary of collision categories to avoid and their rects.

    Returns:
        dict: A dictionary with agent IDs as keys and lists of sensor rays as values.


    Usage:
        -> used by "compute_lidar" Function inside env_helpers.py.
        -> used by "reset" method inside the ContinuousPathfindingEnv class.
        -> used by "step" method inside the ContinuousPathfindingEnv class.
           
    """
    all_sensor_rays = {}
    for agent_id, position in agent_positions.items():
        orientation = agent_orientations[agent_id]

        # Compute sensor rays for this agent
        half_fov = sensor_fov / 2
        
        ray_angles = np.linspace(
            orientation - half_fov,
            orientation + half_fov,
            num_sensor_rays,
            endpoint=False  # Excludes the last angle to avoid duplication
        )
        

        sensor_rays = []
        for angle in ray_angles:
            # Compute ray endpoints using center-based position
            end_x = position[0] + sensor_range * np.cos(angle)
            end_y = position[1] + sensor_range * np.sin(angle)
            sensor_rays.append(((position[0], position[1]), (end_x, end_y)))

        all_sensor_rays[agent_id] = sensor_rays

    return all_sensor_rays




def compute_lidar(sensor_rays, collision_boundaries, sensor_range, scale_x, scale_y):
    """
    Generates simulated LIDAR-sensor data by computing LIDAR distances for each agent based on 
    precomputed sensor rays and collision boundaries.
    Integrates rendering collision logic for efficient use.

    Args:
        sensor_rays (dict): Precomputed sensor rays for each agent {agent_id: [(start, end), ...]}.
        collision_boundaries (dict): Dictionary of collision categories and their rects.
        sensor_range (float): Maximum range of the sensor.
        scale_x (float): Scale factor for x-coordinates.
        scale_y (float): Scale factor for y-coordinates.

    Returns:
        dict: A dictionary of LIDAR distances and collision information for each agent.
              Example:
              {agent_id: {
                  "distances": [float, ...],  # LIDAR range values in logical units
                  "collision_points": [tuple, ...],  # Computed collision points in scaled units
                  "collision_flags": [bool, ...]  # Collision status for visualization
              }}
    Usage:
        -> used by "_render_rgb_array" Function inside env_helpers.py.
        -> used by "reset" method inside the ContinuousPathfindingEnv class.
        -> used by "step" method inside the ContinuousPathfindingEnv class as shown below:       
            # Examlpe of a Simple collision avoidance algorithm for angle, distance in lidar_data implemented in the step method:
            lidar_data = self.simulate_lidar(self.agent_pos, self.agent_angle)
            print("lidar_data:", lidar_data)    
            for angle, distance in lidar_data:
                if distance < 50:
                    self.agent_angle += 90  # Turn 90 degrees to avoid collision break
                    break
    """
    
    lidar_readings = {}

    for agent_id, rays in sensor_rays.items():
        distances = []
        collision_points = []
        collision_flags = []

        for ray_start, ray_end in rays:
            # Convert ray coordinates to numpy arrays in logical units
            ray_start_np = np.array([ray_start[0], ray_start[1]])
            ray_end_np = np.array([ray_end[0], ray_end[1]])
            
            # Scale ray coordinates to visualization space
            scaled_start = np.array([ray_start[0] * scale_x, ray_start[1] * scale_y])
            scaled_end = np.array([ray_end[0] * scale_x, ray_end[1] * scale_y])

            # Default assumption: no collision within the full range
            min_distance = sensor_range
            collision_detected = False
            collision_point = None

            for category, boundaries in collision_boundaries.items():
                if isinstance(boundaries, dict):  # Handle dictionary categories
                    for agent_id_in_category, rect in boundaries.items():
                        if agent_id_in_category == agent_id:  # Ignore own agent and its goal rectangle
                            continue
                        if rect.clipline(scaled_start, scaled_end):  # Check collision with these rects
                            collision_detected = True
                            # Extract intersection point in scaled coordinates
                            scaled_collision_point = rect.clipline(scaled_start, scaled_end)[0]
                            # Convert back to logical units
                            unscaled_collision_point_dict = (
                                scaled_collision_point[0] / scale_x,
                                scaled_collision_point[1] / scale_y
                            )
                            # Calculate distance in logical units
                            distance = np.linalg.norm(ray_start_np - np.array(unscaled_collision_point_dict))
                            if distance < min_distance:
                                min_distance = distance
                                collision_point = scaled_collision_point

                elif isinstance(boundaries, list):  # Handle regular list of rectangles
                    for rect in boundaries:
                        if rect.clipline(scaled_start, scaled_end):  # Collision detection
                            collision_detected = True
                            # Extract intersection point in scaled coordinates
                            scaled_collision_point = rect.clipline(scaled_start, scaled_end)[0]
                            # Convert back to logical units
                            unscaled_collision_point_list = (
                                scaled_collision_point[0] / scale_x,
                                scaled_collision_point[1] / scale_y
                            )
                            # Calculate distance in logical units
                            distance = np.linalg.norm(ray_start_np - np.array(unscaled_collision_point_list))
                            if distance < min_distance:
                                min_distance = distance
                                collision_point = scaled_collision_point

            # Update distances and collision information
            distances.append(min_distance)  # in logical units
            collision_points.append(collision_point if collision_detected else None)
            collision_flags.append(collision_detected)

        # Store results for the agent
        lidar_readings[agent_id] = {
            "distances": distances,
            "collision_points": collision_points,
            "collision_flags": collision_flags,
        }

    return lidar_readings


def avoid_collision_action(lidar_distances, safe_distance, max_speed, action_steering, action_speed, TURN_RATE=np.pi/9):
    """
    Determine collision avoidance actions based on LIDAR distances.
    
    Args:
        lidar_distances (list or np.array): List of LIDAR distances in logical units.
        safe_distance (float): Threshold distance to trigger evasive maneuvers.
        max_speed (float): Maximum possible speed set for the agent for forward acceleration.
        action_steering (float): Input steering angle from the agent's action.
        action_speed (float): Input speed from the agent's action.
        TURN_RATE (float): Maximum steering rate (radians per second).
        
        
    Returns:
        list: [steering, speed] for continuous control.
    """

    BRAKE_FORCE = 0.50*max_speed  # BRAKE_FORCE (float): Deceleration rate to slow down.
    MIN_BRAKE_FORCE = 0.25*max_speed  # MIN_BRAKE_FORCE (float): Minimum possible Deceleration rate to slow down.
    num_rays = len(lidar_distances)  # Number of LIDAR rays
    angle_step = 2 * np.pi / num_rays  # Angle between LIDAR rays (e.g., 360 rays would be 1 degree apart)
    
    # Default steering and speed values
    steering = action_steering
#    steering = np.random.uniform(-np.pi, np.pi)

    #speed = max_speed  # Default speed (full speed)
    speed = action_speed  # Default speed (full speed)

    """
    print("D"*100)
    print(f"DEBUG OUTPUT for avoid_collision_action:")
    print(f"Initial LIDAR Distances: {lidar_distances}")
    print(f"Safe Distance: {safe_distance}")
    print(f"Max Speed: {max_speed}")
        """
    # Find the closest obstacle and its angle
    closest_distance = np.min(lidar_distances)
    closest_index = np.argmin(lidar_distances)

    #print(f"Closest Distance: {closest_distance} (at index {closest_index})")

    
    if closest_distance < safe_distance:
        # If the closest obstacle is within safe distance, adjust speed and steering
        #print(f"Obstacle detected within safe distance! Taking action...")

        # Calculate the relative angle of the closest obstacle
        relative_angle = closest_index * angle_step - np.pi  # Center the agent's forward direction at 0 radians
        #print(f"Relative Angle to Obstacle: {relative_angle} radians")

        # Handle case where obstacle is directly in front (relative_angle = 0)
        if relative_angle == 0:
            # If obstacle is directly in front, steer slightly left or right
            #steering = np.random.choice([-(TURN_RATE*1), (TURN_RATE*1)])  # Randomly choose to steer left or right
            steering = np.random.choice([-(TURN_RATE*1), (TURN_RATE*1)])  # Randomly choose to steer left or right

            #print(f"Obstacle directly ahead. Steering adjusted: {steering} radians")
        else:
            # Adjust steering to turn away from the obstacle
            steering = np.clip(relative_angle, -(TURN_RATE*1), (TURN_RATE*1))
            #print(f"Adjusted Steering: {steering} radians (clipped to max turn rate)")

        # Apply brake force based on how close the obstacle is (adjust brake force based on distance)
        speed = BRAKE_FORCE * (closest_distance / safe_distance)
        speed = max(MIN_BRAKE_FORCE, speed)  # Ensure speed doesn't go below a very small value
        #print(f"Speed reduced to: {speed} (due to close obstacle)")
    
    else:
        #print("No obstacles within safe distance. Continuing at full speed.")
        # Only reset to max speed if the agent wasn't already braking
        if speed < max_speed:
            speed = max_speed  # Full speed ahead
    
    #print(f"Final Steering: {steering}, Final Speed: {speed}")
    
    # Optionally: since there are other obstacles, steer towards the "clearest" path
    # or use more advanced strategies like a weighted average of multiple obstacles, etc.
    
    return [steering, speed]






def field_forces(agent_id, lidar_data, agent_position, goal_position, agent_size, scale_x, scale_y, safe_distance):
    """
    Calculate potential fields forces for collision avoidance using lidar data.
    - Attractive force towards goal
    - Repulsive force from obstacles
    
    Args:
    - agent_id (str): The ID of the agent.
    - lidar_data (dict): The lidar readings for the agent.
    - agent_position (np.array): The current position of the agent.
    - goal_position (np.array): The goal position for the agent.
    - agent_size (tuple): The size of the agent (width, height).
    
    Returns:
    - force_x, force_y (float): The total force applied on the agent.
    """
    force_x, force_y = 0, 0

    # 1. Attractive force towards the goal
    goal_direction = np.array(goal_position) - np.array(agent_position)
    distance_to_goal = np.linalg.norm(goal_direction)
    
    if distance_to_goal > 0:
        goal_direction /= distance_to_goal  # Normalize to unit vector
        force_goal = goal_direction * distance_to_goal  # Attractive force
        force_x += force_goal[0]*0.1*0
        force_y += force_goal[1]*0.1*0

    # 2. Repulsive force from obstacles based on lidar data
    for i in range(len(lidar_data["distances"])):
        distance = lidar_data["distances"][i]
        collision_flag = lidar_data["collision_flags"][i]
        scaled_collision_point = lidar_data["collision_points"][i]


        if collision_flag and distance < safe_distance:  # Only consider nearby obstacles
            # Calculate repulsion strength based on the inverse of the square of the distance
            repulsion_strength = 1.0 / (distance ** 2)
            
            # Convert back to logical units
            unscaled_collision_point = (
                scaled_collision_point[0] / scale_x,
                scaled_collision_point[1] / scale_y
            )

            # Get the direction of the obstacle (angle or orientation of lidar beam)
            angle = np.arctan2(unscaled_collision_point[1] - agent_position[1], 
                               unscaled_collision_point[0] - agent_position[0])
            
            # Compute repulsion vector
            repulsion_x = repulsion_strength * np.cos(angle) * 0.0000008*1
            repulsion_y = repulsion_strength * np.sin(angle) * 0.0000008*1
            
            # Add repulsion to the total force
            force_x -= repulsion_x
            force_y -= repulsion_y

    return force_x, force_y





def calculate_safe_dist(half_width, half_height):
    """
    Calculate a safe distance based on agent's half-width and half-height.
    
    Args:
        half_width (float): Half the agent's width.
        half_height (float): Half the agent's height.
    
    Returns:
        float: A calculated safe distance value.
    """
    # Euclidean distance from center to edge in logical units
    return np.sqrt(half_width**2 + half_height**2)






#==================================OBSERVATION COMPUTATION UTILS===============================================


def compute_agents_relative_positions(agent_id, agent_positions):
    """
    Compute the relative positions of all other agents with respect to a given agent.

    Args:
        agent_id (str): The ID of the reference agent.
        agent_positions (dict): A dictionary mapping agent IDs to their positions.

    Returns:
        np.ndarray: A flattened array of relative positions (x, y) for all other agents.

    Usage:
        used by "_compute_agent_observation" function.
        
    """
    reference_position = agent_positions[agent_id]
    relative_positions = []

    for other_agent, position in agent_positions.items():
        if other_agent != agent_id:
            # Compute relative position
            relative_position = np.array(position) - np.array(reference_position)
            relative_positions.append(relative_position)

    # Flatten the relative positions to a 1D array (N * 2,)
    return np.concatenate(relative_positions)



def _generate_collision_category_map(field_size, collision_boundaries, scale_x, scale_y):
    """
    Generate a spatial map highlighting collision categories to avoid based on Pygame sprites.

    Args:
        field_size (tuple): The size of the environment's field (width, height) in logical units.
        collision_boundaries (dict): A dictionary containing categories with Pygame rects as values.
            Example:
                {
                    "obstacles": [<rect(...)>, ...],
                    "entrances": [<rect(...)>, ...],
                    "walls": [<rect(...)>, ...],
                    "goal_boundaries": {"agent_0": <rect(...)>, ...},
                    "agent_boundaries": {"agent_0": <rect(...)>, ...},
                }
        scale_x (float): Scaling factor for the x-axis (pixels to logical units).
        scale_y (float): Scaling factor for the y-axis (pixels to logical units).

    Returns:
        np.ndarray: A 2D map with 1's representing the collision categories to avoid, 0's elsewhere.

    Usage:
        used by "_compute_agent_observation" function.
        
    """
    # Initialize a zeroed map with the field size in logical units
    collision_map = np.zeros((int(field_size[0]), int(field_size[1])), dtype=np.float32)

    # Iterate over each category in collision_boundaries
    for category, rects in collision_boundaries.items():
        # Handle dictionary-based categories (goal_boundaries, agent_boundaries)
        if category in {"goal_boundaries", "agent_boundaries"}:
            rects = rects.values()  # Extract the rects from the dictionary values

        for rect in rects:
            # Convert Pygame rect's bounding box to logical grid coordinates
            start_x = int((rect.left) // scale_x)
            start_y = int((rect.top) // scale_y)
            end_x = int((rect.right) // scale_x)
            end_y = int((rect.bottom) // scale_y)

            # Ensure the bounding box lies within the logical grid boundaries
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(int(field_size[0]), end_x)
            end_y = min(int(field_size[1]), end_y)

            # Mark the collision map with 1's in the corresponding grid cells
            collision_map[start_y:end_y, start_x:end_x] = 1

    return collision_map




def _compute_collision_category_distances(agent_position, collision_boundaries, sensor_range, scale_x, scale_y):
    """
    Compute the distance from the agent to the nearest object in each collision category.

    Args:
        agent_position (tuple): The (x, y) position of the agent in logical units.
        collision_boundaries (dict): Dictionary of collision categories with Pygame rects as values.
            Example:
                {
                    "obstacles": [<rect(...)>, ...],
                    "entrances": [<rect(...)>, ...],
                    "workstations": [<rect(...)>, ...],
                    "walls": [<rect(...)>, ...],
                    "storage_areas": [<rect(...)>, ...],
                    "loading_docks": [<rect(...)>, ...],
                    "goal_boundaries": {"agent_0": <rect(...)>, "agent_1": <rect(...)>, ...},
                    "agent_boundaries": {"agent_0": <rect(...)>, "agent_1": <rect(...)>, ...},
                }
        sensor_range (float): Maximum sensing range of the agent.
        scale_x (float): Scaling factor for the x-axis (pixels to logical units).
        scale_y (float): Scaling factor for the y-axis (pixels to logical units).

    Returns:
        np.ndarray: Array of distances to the nearest object in each collision category.

    Usage:
        used by "_compute_agent_observation" function.
        
    """
    distances = []

    # Iterate over each collision category
    for category, rects in collision_boundaries.items():
        min_distance = sensor_range  # Initialize with the maximum sensor range

        # Handle dictionary-based categories (goal_boundaries, agent_boundaries)
        if category in {"goal_boundaries", "agent_boundaries"}:
            rects = rects.values()  # Extract the rects from the dictionary values

        # Iterate over all rects in the category
        for rect in rects:
            # Use the rect's center directly (assuming center-based convention)
            rect_center = (
                rect.centerx / scale_x,
                rect.centery / scale_y
            )

            # Calculate Euclidean distance from the agent to the rect center
            distance = np.linalg.norm(np.array(agent_position) - np.array(rect_center))

            # Update the minimum distance for this category if closer
            if distance < min_distance:
                min_distance = distance

        # Add the minimum distance for this category to the list
        distances.append(min_distance)

    # Convert distances to a numpy array
    return np.array(distances, dtype=np.float32)


def _get_action_mask(agent_id, agent_positions, agent_goals, atol=0.5):
    """
    Generate an action mask for the given agent.

    Args:
        agent_id (str): The ID of the agent.
        agent_positions (dict): Dictionary of agent positions where keys are agent IDs.
        agent_goals (dict): Dictionary of agent goals where keys are agent IDs.
        atol (float): Absolute tolerance to determine if the agent is at its goal.

    Returns:
        list[int]: A binary list where 1 indicates the action is valid, and 0 indicates it's invalid.

    Usage:
        used by "_compute_agent_observation" function.
        

        
    # Debugging output
    print(f"[DEBUG] Agent ID: {agent_id}")
    print(f"[DEBUG] Agent Position: {agent_positions[agent_id]}")
    print(f"[DEBUG] Agent Goal: {agent_goals[agent_id]}")
    print(f"[DEBUG] Tolerance (atol): {atol}")
    
    """

    if np.allclose(agent_positions[agent_id], agent_goals[agent_id], atol=atol):
        # If the agent has reached its goal, only "stay idle" is valid
        action_mask = [1, 0] # Example: [stay_idle, move]
    else:
        action_mask = [1, 1] # All actions valid


    # Log the resulting action mask
    #print(f"[DEBUG] Computed Action Mask: {action_mask}")
    return action_mask


from collections import OrderedDict

def sort_nested_dict(d):
    """
    Recursively sort the keys of a dictionary, including nested dictionaries.
    
    Args:
        d (dict): The dictionary to sort.
        
    Returns:
        OrderedDict: The sorted dictionary.
    """
    if isinstance(d, dict):
        return OrderedDict((key, sort_nested_dict(value)) for key, value in sorted(d.items()))
    else:
        return d





def _compute_agent_observation(
    agent_id,
    agent_positions,
    agent_orientations,
    agent_speeds,
    agent_goals,
    obstacles,
    workstations,
    entrances,
    storage_areas,
    loading_docks,
    walls,
    field_size,
    sensor_fov,
    num_sensor_rays,
    sensor_range,
    lidar_readings,
    collision_boundaries,
    scale_x,
    scale_y,
    agents_relative_positions=None,
    other_collision_categories_to_avoid_map=None,
    time_step=None,
    max_timesteps=None,
    status=None,
    action_mask=None,
):
    # Extract agent-specific data
    agent_position = agent_positions[agent_id]
    agent_orientation = agent_orientations[agent_id]
    agent_speed = agent_speeds[agent_id]
    agent_goal = agent_goals[agent_id]

    # Ensure sensor range is at least 1 for stability
    sensor_range = max(1, sensor_range)

    # Generate other_collision_categories_to_avoid_map if None
    if other_collision_categories_to_avoid_map is None:
        other_collision_categories_to_avoid_map = _generate_collision_category_map(
            field_size, collision_boundaries, scale_x, scale_y
        ).flatten()  # Flatten to ensure compatibility

    # Compute other agents' relative positions if None
    if agents_relative_positions is None:
        agents_relative_positions = compute_agents_relative_positions(agent_id, agent_positions).flatten()

    # Compute action mask dynamically
    if action_mask is None:
        action_mask = _get_action_mask(agent_id)

    # Compute distances for all categories to be avoided
    collision_distances = _compute_collision_category_distances(
        agent_position, collision_boundaries, sensor_range, scale_x, scale_y
    )

    # Ensure lidar readings have the correct shape
    if lidar_readings is None or len(lidar_readings) != num_sensor_rays:
        lidar_readings = np.zeros(num_sensor_rays, dtype=np.float32)

    # Sensor proximity map placeholder logic
    sensor_proximity_map = np.zeros((sensor_range, sensor_range), dtype=np.float32)



    # Handle empty list checks in common data categories
    common_dict = {}
    for name, data in [('obstacles', obstacles), 
                       ('workstations', workstations),
                       ('entrances', entrances), 
                       ('storage_areas', storage_areas),
                       ('loading_docks', loading_docks), 
                       ('walls', walls)]:
        if len(data) > 0:
            common_dict[name] = np.array(data, dtype=np.float32).flatten()  # Ensure correct shape (N*2,)
        else:
            continue

    # Construct the observation dictionary with reshaped data
    observation = {
        "common": common_dict,
        "position": np.array(agent_position, dtype=np.float32),  # Shape (2,)
        "velocity": np.array(agent_speed, dtype=np.float32),  # Shape (1,)
        "goal": np.array(agent_goal, dtype=np.float32),  # Shape (2,)
        "orientation": np.array([agent_orientation], dtype=np.float32),  # Shape (1,)
        "local_sensor_observations": np.array(lidar_readings, dtype=np.float32),  # Shape (num_sensor_rays,)
        #"sensor_proximity_map": np.array(sensor_proximity_map, dtype=np.float32).flatten(),  # Flatten to 1D
        "other_agents_relative_positions": agents_relative_positions,  # Shape (N,)
        #"other_collision_categories_to_avoid_map": np.array(other_collision_categories_to_avoid_map, dtype=np.float32),  # Shape (N,)
        "other_collision_categories_to_avoid_distances": np.array(collision_distances, dtype=np.float32),  # Shape (N,)
        #"agent_proximities": np.zeros((len(agents_relative_positions),), dtype=np.float32),  # Placeholder, Shape (N,)
        #"time_step": np.array([time_step if time_step is not None else 0], dtype=np.float32),  # Shape (1,)
        #"status": np.array([status if status is not None else 0], dtype=np.int32),  # Shape (1,)
        #"action_mask": np.array(action_mask, dtype=np.float32),  # Shape (2,)
    }

    # DEBUG: Log the observation for debugging
    # print(f"Observation for agent {agent_id}: {observation}")

    return observation




def _flatten_observation(observation):
    """
    Flatten the observation dictionary into a single 1D array.
    Handles nested dictionaries by recursively flattening them.
    """
    flattened_obs = []

    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            # Flatten the ndarray (Box type observation)
            flattened_obs.append(value.flatten())
        elif isinstance(value, dict):
            # Recursively flatten nested dictionaries (e.g., 'common' dictionary)
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    flattened_obs.append(sub_value.flatten())  # Flatten ndarray inside the dictionary
                else:
                    # Handle other types inside the dictionary (if needed)
                    flattened_obs.append(np.array(sub_value).flatten())
        elif isinstance(value, list):
            # Flatten lists (if necessary)
            flattened_obs.append(np.array(value).flatten())
        elif isinstance(value, int) or isinstance(value, float):
            # Handle single scalar values (e.g., status, time_step)
            flattened_obs.append(np.array(value).flatten())
        else:
            # For any unhandled types, just convert to a numpy array
            flattened_obs.append(np.array(value).flatten())

    return np.concatenate(flattened_obs, dtype=np.float32)

def flatten_observation(obs_values):
    """Flatten nested observations into a PyTorch tensor."""
    flat_obs = []
    for key, value in obs_values.items():
        if isinstance(value, dict):
            flat_obs.extend(flatten_observation(value))
        elif hasattr(value, "flatten"):  # NumPy array or similar
            flat_obs.extend(value.flatten())
        else:
            flat_obs.append(value)
    return torch.tensor(flat_obs, dtype=torch.float32)  # Convert to PyTorch tensor






#----------------------------------COLLISION DETECTION HELPER UTILS---------------------------------------------------


    
def create_category_rects(category_items, scale_x, scale_y):
    """
    Create pygame.Rect objects for a given category of items with center-based coordinates in line with layout position.

    Args:
        category_items (list): List of dictionary items containing `position` and `dimensions`.
        scale_x (float): Scaling factor for the x-axis.
        scale_y (float): Scaling factor for the y-axis.

    Returns:
        list: List of pygame.Rect objects representing the scaled boundaries for the items.

    Usage:
        used by "setup_collision_boundaries(env_layout, goal_positions, goal_size, scale_x, scale_y)" function.
        
    """
    rects = []
    for item in category_items:
        center_x, center_y = item["position"]  # center-based coordinates
        width, height = item["dimensions"]

        # Scale dimensions and positions
        scaled_width = width * scale_x
        scaled_height = height * scale_y
        scaled_center_x = center_x * scale_x
        scaled_center_y = center_y * scale_y

        # Create a pygame.Rect with scaled dimensions and center position
        rect = pygame.Rect(0, 0, scaled_width, scaled_height)
        rect.center = (scaled_center_x, scaled_center_y)

        rects.append(rect)

    return rects



def _create_agent_rects(agent_positions, agent_size, scale_x, scale_y):
    """
    Create a dictionary of rects for agents to handle collisions based on current positions.

    Args:
        agent_positions (dict): Dictionary with agent IDs as keys and (x, y) positions as values.
        agent_size (tuple): Size of each agent in logical units (width, height).
        scale_x (float): Scaling factor for x-axis.
        scale_y (float): Scaling factor for y-axis.

    Returns:
        dict: Dictionary of agent IDs mapped to their corresponding pygame.Rect objects.

    Usage:
        -> used by "init" method inside the ContinuousPathfindingEnv class.
        -> used by "init" method inside the ContinuousPathfindingEnv class for "self.agent_rects"
        
    """
    rects = {}
    for agent_id, position in agent_positions.items():  # Iterate over the dictionary
        # Ensure the position is a valid 2D array
        if not isinstance(position, (tuple, list, np.ndarray)) or len(position) != 2:
            raise ValueError(f"Invalid position for {agent_id}: {position}, expected a 2D tuple, list, or ndarray.")

        # Proceed only if the position is valid
        center_x, center_y = position

        # Scale dimensions and center position
        scaled_width = agent_size[0] * scale_x
        scaled_height = agent_size[1] * scale_y
        scaled_center_x = center_x * scale_x
        scaled_center_y = center_y * scale_y

        # Create a pygame.Rect with scaled dimensions and center position
        rect = pygame.Rect(0, 0, scaled_width, scaled_height)
        rect.center = (scaled_center_x, scaled_center_y)

        # Map agent ID to its rect
        rects[agent_id] = rect

    # Validate all rects before returning
    assert all(isinstance(rect, pygame.Rect) for rect in rects.values()), "All elements in rects must be pygame.Rect"

    return rects

def update_agent_boundaries(agent_positions, agent_size, scale_x, scale_y): # 
    """
    Update boundaries for all agents.

    Usage:
        not yet used inside the ENV class step method function.
      
    """
    agent_boundaries = {}
    for agent_id, agent_pos in agent_positions.items():
        rect = _create_agent_rects(agent_pos, agent_size, scale_x, scale_y)
        agent_boundaries[agent_id] = rect
    return agent_boundaries

def _create_goal_rects(goal_positions, goal_size, scale_x, scale_y):
    """
    Create a dictionary of rects for goals to prevent agents from occupying other agents' goals.

    Args:
        goal_positions (dict): Dictionary with agent IDs as keys and (x, y) goal positions as values.
        goal_size (tuple): Size of each goal in logical units (width, height).
        scale_x (float): Scaling factor for x-axis.
        scale_y (float): Scaling factor for y-axis.

    Returns:
        dict: Dictionary of agent IDs mapped to their corresponding pygame.Rect objects.

    Usage:
        used by "setup_collision_boundaries(env_layout, goal_positions, goal_size, scale_x, scale_y)" function.
        used by "init" method inside the ContinuousPathfindingEnv class for "self.goal_rects".
        
    """
    rects = {}
   
    for agent_id, position in goal_positions.items():  # Iterate over the dictionary
        # Validate the goal position
        if not isinstance(position, (tuple, list, np.ndarray)) or len(position) != 2:
            raise ValueError(f"Invalid goal position for {agent_id}: {position}, expected a 2D tuple, list, or ndarray.")
        
        # Unpack position
        center_x, center_y = position

        # Scale dimensions and center position
        scaled_width = goal_size[0] * scale_x
        scaled_height = goal_size[1] * scale_y
        scaled_center_x = center_x * scale_x
        scaled_center_y = center_y * scale_y

        # Create a pygame.Rect with scaled dimensions and center position
        rect = pygame.Rect(0, 0, scaled_width, scaled_height)
        rect.center = (scaled_center_x, scaled_center_y)

        # Map agent ID to its rect
        rects[agent_id] = rect
    
    # Validate all rects before returning
    assert all(isinstance(rect, pygame.Rect) for rect in rects.values()), "All elements in rects must be pygame.Rect"
    
    return {} # return rects if goals are to be avoided by the agents 
    

def setup_collision_boundaries(env_layout, goal_positions, goal_size, scale_x, scale_y):
    """
    Set up collision boundaries for all categories in the environment.

    Args:
        env_layout (dict): The layout information from `get_env_layout`.
        goal_positions (dict): A dictionary mapping agent IDs to their goal positions.
        goal_size (tuple): The dimensions of each goal as (width, height).
        scale_x (float): Scaling factor for the x-axis.
        scale_y (float): Scaling factor for the y-axis.

    Returns:
        dict: A dictionary mapping category names to their respective pygame.Rect objects.

    Usage:
        -> used by the "setup_collision_boundaries" method inside the ContinuousPathfindingEnv class.        
        -> used by the "init" method inside the ContinuousPathfindingEnv class 
           for self.collision_boundaries = self.setup_collision_boundaries().        
    """
    category_rects = {}

    # List of all categories with position and dimensions
    categories = ["obstacles", "entrances", 
                  "workstations", "walls", 
                  "storage_areas", "loading_docks", 
                  "goal_boundaries", 
                  "agent_boundaries"] # dynamic agent_boundaries will be created inside the step method

    for category in categories:
        if category in env_layout:
            # Create rects for static environment elements
            category_rects[category] = create_category_rects(env_layout[category], scale_x, scale_y)
        elif category == "goal_boundaries":
            # Create rects for goal boundaries
            category_rects[category] = _create_goal_rects(goal_positions, goal_size, scale_x, scale_y)

    return category_rects


    
def _is_position_valid_XM(agent, agent_rect, collision_boundaries, agent_rects, goal_rects):
    """
    Check if the agent's new position is valid by detecting collisions with obstacles or other categories,
    ignoring its own rectangle ONLY! (excluding mine).

    Args:
        agent (str): The agent's ID.
        agent_rect (pygame.Rect): The agent's rectangle for collision testing.
        collision_boundaries (dict): Dictionary of collision boundaries by category.
        agent_rects (dict): Dictionary of other agents' rects.
        goal_rects (dict): Dictionary of goal rects.

    Returns:
        bool: True if the position is valid, False if there is a collision.

    Usage:
        -> used by the "" function.
        -> used by the the reset method and its helper method.
        
    """
    # Validate input types
    assert isinstance(agent_rect, pygame.Rect), f"agent_rect is not a pygame.Rect: {agent_rect}"
    assert all(isinstance(rect, pygame.Rect) for rect in agent_rects.values()), "All elements in agent_rects must be pygame.Rect"
    assert all(isinstance(rect, pygame.Rect) for rect in goal_rects.values()), "All elements in goal_rects must be pygame.Rect"

    """
    # Check an agent's boundary area to see if is collides with other collision category boundary areas, while ignoring 
    the agent's boundary area.
    """
    # Iterate through all categories in the collision_boundaries
    for category, category_data in collision_boundaries.items():
        
        # Case 1: If category data is a list of rectangles (e.g., 'obstacles', 'workstations', 'walls')
        if isinstance(category_data, list):
            for boundary_rect in category_data:
                if agent_rect.colliderect(boundary_rect):
                    print(f"Collision detected with {category} boundary at {boundary_rect}!.")
                    print("$"*100)
                    return False
        
        # Case 2: If category data is a dictionary of rectangles (e.g., 'goal_boundaries', 'agent_boundaries')
        elif isinstance(category_data, dict):
            for other_agent, boundary_rect in category_data.items():
                # Skip checking the agent's own rectangle in agent-boundaries
                if category == 'agent_boundaries' and agent == other_agent:
                    continue
                
                # Check for collisions with other agent's goal areas or agent boundaries
                if agent_rect.colliderect(boundary_rect):
                    print(f"Collision detected with {category} ({other_agent}) at {boundary_rect}!.")
                    print("$"*100)
                    return False
    
    # Check for collisions with other agents' positions (from agent_rects)
    for other_agent, other_rect in agent_rects.items():
        if agent != other_agent:  # Skip checking self
            if agent_rect.colliderect(other_rect):
                print(f"Collision detected with another agent! for {other_agent}'s position at {other_rect}.")
                print("$"*100)
                return False

    # Check for collisions with other agents' goal rects (from goal_rects)
    for other_agent, goal_rect in goal_rects.items():
        if agent_rect.colliderect(goal_rect):
            print("Debug _is_position_valid Function:")
            print(f"Collision detected with another agent's goal! for {other_agent}'s goal area at {goal_rect}.")
            print("$"*100)
            return False

    # If no collisions are detected, the position is valid
    return True


    
def _is_position_valid_XG(agent, agent_rect, collision_boundaries, agent_rects, goal_rects):
    """
    Check if the agent's new position is valid by detecting collisions with obstacles or other categories,
    ignoring its own goal's rectangle ONLY! (excluding my goal).

    Args:
        agent (str): The agent's ID.
        agent_rect (pygame.Rect): The agent's rectangle for collision testing.
        collision_boundaries (dict): Dictionary of collision boundaries by category.
        agent_rects (dict): Dictionary of other agents' rects.
        goal_rects (dict): Dictionary of goal rects.

    Returns:
        bool: True if the position is valid, False if there is a collision.

    Usage:
        -> used by the "" function.
        -> used by the the reset method and its helper method.
        
    """
    # Validate input types
    assert isinstance(agent_rect, pygame.Rect), f"agent_rect is not a pygame.Rect: {agent_rect}"
    assert all(isinstance(rect, pygame.Rect) for rect in agent_rects.values()), "All elements in agent_rects must be pygame.Rect"
    assert all(isinstance(rect, pygame.Rect) for rect in goal_rects.values()), "All elements in goal_rects must be pygame.Rect"

    """
    # Check an agent's boundary area to see if is collides with other collision category boundary areas, while ignoring 
    the agent's boundary area.
    """
    # Iterate through all categories in the collision_boundaries
    for category, category_data in collision_boundaries.items():
        
        # Case 1: If category data is a list of rectangles (e.g., 'obstacles', 'workstations', 'walls')
        if isinstance(category_data, list):
            for boundary_rect in category_data:
                if agent_rect.colliderect(boundary_rect):
                    print(f"Collision detected with {category} boundary at {boundary_rect}!.")
                    print("$"*100)
                    return False
        
        # Case 2: If category data is a dictionary of rectangles (e.g., 'goal_boundaries', 'agent_boundaries')
        elif isinstance(category_data, dict):
            for other_agent, boundary_rect in category_data.items():
                # Skip checking the agent's own rectangle in agent-boundaries
                if category == 'goal_boundaries' and agent == other_agent:
                    continue
                
                # Check for collisions with other agent's goal areas or agent boundaries
                if agent_rect.colliderect(boundary_rect):
                    print(f"Collision detected with {category} ({other_agent}) at {boundary_rect}!.")
                    print("$"*100)
                    return False
    
    # Check for collisions with other agents' positions (from agent_rects)
    for other_agent, other_rect in agent_rects.items():
        if agent_rect.colliderect(other_rect):
            print(f"Collision detected with another agent! for {other_agent}'s position at {other_rect}.")
            print("$"*100)
            return False

    # Check for collisions with other agents' goal rects (from goal_rects)
    for other_agent, goal_rect in goal_rects.items():
        if agent != other_agent:  # Skip checking its goal rect
            if agent_rect.colliderect(goal_rect):
                print("Debug _is_position_valid Function:")
                print(f"Collision detected with another agent's goal! for {other_agent}'s goal area at {goal_rect}.")
                print("$"*100)
                return False

    # If no collisions are detected, the position is valid
    return True


def _is_position_valid_XMG(agent, agent_rect, collision_boundaries, agent_rects, goal_rects):
    """
    Check if the agent's new position is valid by detecting collisions with obstacles or other categories,
    ignoring its own rectangle ONLY! (excluding mine and that of my goal).

    Args:
        agent (str): The agent's ID.
        agent_rect (pygame.Rect): The agent's rectangle for collision testing.
        collision_boundaries (dict): Dictionary of collision boundaries by category.
        agent_rects (dict): Dictionary of other agents' rects.
        goal_rects (dict): Dictionary of goal rects.

    Returns:
        bool: True if the position is valid, False if there is a collision.

    Usage:
        -> used by the "_calculate_reward" function.
        -> used by the the step method ....
        
    """
    # Validate input types
    assert isinstance(agent_rect, pygame.Rect), f"agent_rect is not a pygame.Rect: {agent_rect}"
    assert all(isinstance(rect, pygame.Rect) for rect in agent_rects.values()), "All elements in agent_rects must be pygame.Rect"
    assert all(isinstance(rect, pygame.Rect) for rect in goal_rects.values()), "All elements in goal_rects must be pygame.Rect"

    """
    # Check an agent's boundary area to see if is collides with other collision category boundary areas, while ignoring 
    the agent's boundary area.
    """
    # Iterate through all categories in the collision_boundaries
    for category, category_data in collision_boundaries.items():
        
        # Case 1: If category data is a list of rectangles (e.g., 'obstacles', 'workstations', 'walls')
        if isinstance(category_data, list):
            for boundary_rect in category_data:
                if agent_rect.colliderect(boundary_rect):
                    #print(f"Collision detected with {category} boundary at {boundary_rect}!.")
                    #print("$"*100)
                    return False
        
        # Case 2: If category data is a dictionary of rectangles (e.g., 'goal_boundaries', 'agent_boundaries')
        elif isinstance(category_data, dict):
            for other_agent, boundary_rect in category_data.items():
                # Skip checking the agent's own rectangle in agent-boundaries
                if agent == other_agent:  # (excluding mine and that of my goal)
                    continue
                
                # Check for collisions with other agent's goal areas or agent boundaries
                if agent_rect.colliderect(boundary_rect):
                    #print(f"Collision detected with {category} ({other_agent}) at {boundary_rect}!.")
                    #print("$"*100)
                    return False
    
    # Check for collisions with other agents' positions (from agent_rects)
    for other_agent, other_rect in agent_rects.items():
        if agent != other_agent:  # Skip checking self
            if agent_rect.colliderect(other_rect):
                #print(f"Collision detected with another agent! for {other_agent}'s position at {other_rect}.")
                #print("$"*100)
                return False

    # Check for collisions with other agents' goal rects (from goal_rects)
    for other_agent, goal_rect in goal_rects.items():
        if agent != other_agent:  # Skip checking that agent's goal rectangle collision
            if agent_rect.colliderect(goal_rect):
                #print("Debug _is_position_valid Function:")
                #print(f"Collision detected with another agent's goal! for {other_agent}'s goal area at {goal_rect}.")
                #print("$"*100)
                return False

    # If no collisions are detected, the position is valid
    return True
    
def _is_position_valid_XMAG(agent, agent_rect, collision_boundaries, agent_rects, goal_rects):
    """
    Check if the agent's new position is valid by detecting collisions with obstacles or other categories,
    ignoring its own rectangle and all goals! (excluding mine, that of my goal, and that of other goals).

    Args:
        agent (str): The agent's ID.
        agent_rect (pygame.Rect): The agent's rectangle for collision testing.
        collision_boundaries (dict): Dictionary of collision boundaries by category.
        agent_rects (dict): Dictionary of other agents' rects.
        goal_rects (dict): Dictionary of goal rects.

    Returns:
        bool: True if the position is valid, False if there is a collision.

    Usage:
        -> used by the "_calculate_reward" function.
        -> used by the the step method ....
        
    """
    # Validate input types
    assert isinstance(agent_rect, pygame.Rect), f"agent_rect is not a pygame.Rect: {agent_rect}"
    assert all(isinstance(rect, pygame.Rect) for rect in agent_rects.values()), "All elements in agent_rects must be pygame.Rect"
    assert all(isinstance(rect, pygame.Rect) for rect in goal_rects.values()), "All elements in goal_rects must be pygame.Rect"

    """
    # Check an agent's boundary area to see if is collides with other collision category boundary areas, while ignoring 
    the agent's boundary area.
    """
    # Iterate through all categories in the collision_boundaries
    for category, category_data in collision_boundaries.items():
        
        # Case 1: If category data is a list of rectangles (e.g., 'obstacles', 'workstations', 'walls')
        if isinstance(category_data, list):
            for boundary_rect in category_data:
                if agent_rect.colliderect(boundary_rect):
                    #print(f"Collision detected with {category} boundary at {boundary_rect}!.")
                    #print("$"*100)
                    return False
        
        # Case 2: If category data is a dictionary of rectangles (e.g., 'goal_boundaries', 'agent_boundaries')
        elif isinstance(category_data, dict):
            for other_agent, boundary_rect in category_data.items():
                # Skip checking the agent's own rectangle in agent-boundaries
                if agent == other_agent:  # (excluding mine and that of my goal)
                    continue
                
                # Check for collisions with other agent's goal areas or agent boundaries
                if agent_rect.colliderect(boundary_rect):
                    #print(f"Collision detected with {category} ({other_agent}) at {boundary_rect}!.")
                    #print("$"*100)
                    return False
    
    # Check for collisions with other agents' positions (from agent_rects)
    for other_agent, other_rect in agent_rects.items():
        if agent != other_agent:  # Skip checking self
            if agent_rect.colliderect(other_rect):
                #print(f"Collision detected with another agent! for {other_agent}'s position at {other_rect}.")
                #print("$"*100)
                return False
                
    """
    # Check for collisions with other agents' goal rects (from goal_rects)
    for other_agent, goal_rect in goal_rects.items():
        if agent != other_agent:  # Skip checking that agent's goal rectangle collision
            if agent_rect.colliderect(goal_rect):
                #print("Debug _is_position_valid Function:")
                #print(f"Collision detected with another agent's goal! for {other_agent}'s goal area at {goal_rect}.")
                #print("$"*100)
                return False
                """

    # If no collisions are detected, the position is valid
    return True

#---------------------------------------REWARD UTILS------------------------------------------------



def _calculate_reward(agent, distance_to_goal, previous_distances, agent_rects, goal_rects, collision_boundaries, penalty_no_progress, penalty_collision, penalty_proximity, penalty_proximity_obstacle, reward_scale_goal_proximity, goal_radius, reward_goal_reached, SAFE_DISTANCE, lidar_readings):
    """
    Calculate the reward for an agent based on its current state.

    Args:
        agent (str): The ID of the agent.
        distance_to_goal (float): The current distance to the goal.
        previous_distances (dict): Dictionary tracking previous distances for each agent.
        agent_rects (dict): Dictionary of agent rectangles for collision detection.
        goal_rects (dict): Dictionary of goal rectangles for collision detection.
        collision_boundaries (dict): Dictionary of collision boundaries by category.
        penalty_no_progress (float): Penalty for moving away from the goal.
        penalty_collision (float): Penalty for collisions.
        penalty_proximity (float): Penalty for getting too close to other agents.
        penalty_proximity_obstacle (float): Penalty for getting too close to obstacles.
        reward_scale_goal_proximity (float): Scaling factor for goal proximity reward.
        goal_radius (float): The radius of the goal area.
        reward_goal_reached (float): Reward for reaching the goal.
        SAFE_DISTANCE (float): safety distance threshold beyond which collision risk is very high.
        lidar_readings (dict): dictionary containing computed LIDAR sensor readings or data (like dictances, etc).

    Returns:
        float: The calculated reward for the agent.
    """
    reward = 0.0

    # 1. Positive reward for moving closer to the goal
    previous_distance = previous_distances.get(agent, None)
    if previous_distance is None or np.isinf(previous_distance) or np.isnan(previous_distance):
        previous_distance = distance_to_goal  # Avoids inf in first step

    if distance_to_goal < previous_distance:
        reward += (previous_distance - distance_to_goal) * reward_scale_goal_proximity
    else:
        reward -= penalty_no_progress  # Penalty for moving further or not progressing
        
    # Convert agent_rects to a dictionary if it's not already
    if isinstance(agent_rects, list):
        agent_rects = {f"agent_{i}": rect for i, rect in enumerate(agent_rects)}
        

    # 2. Negative reward or Penalty for collisions with other elements or invalid moves
    agent_rect = agent_rects.get(agent)
    if agent_rect and not _is_position_valid_XMG(agent, agent_rect, collision_boundaries, agent_rects, goal_rects):        
        reward -= penalty_collision
        print("~"*100)
        print("Debug Reward Function:")
        print(f"Agent {agent} collided!")
        print("~"*100)

    # 3. Gently Penalize using sensor rays for proximity to an object ->if agent is too close to any other categories:
    for agent_id, lidar in lidar_readings.items():
        if any(dist <= SAFE_DISTANCE for dist in lidar["distances"]):  # Threshold for collision
            reward -= penalty_proximity_obstacle  # Small negative reward for collisions

    
    """""
    # 4. Encourage smooth motion (penalize large velocity changes)
    #    Assuming `observations` contains previous and current velocities:
    velocity_change_penalty = np.abs(observations[agent_id]['velocity_change']).sum()
    reward -= velocity_change_penalty * 2

    # Update agent's reward
    rewards[agent_id] = reward
    
    # 4. Ensure agents receive small negative rewards for proximity to obstacles
    for dist in lidar["distances"]:
        reward[agent_id] -= 1 / (dist + 1e-5)  # Penalty inversely proportional to distance
        

    # 3. Shaping reward: Penalize proximity to obstacles or other agents
    for other_agent, other_rect in agent_rects.items():
        if other_agent != agent:
            proximity = agent_rect.colliderect(other_rect)
            if proximity:
                reward -= penalty_proximity
                print(f"Agent {agent} is too close to Agent {other_agent}!")

    # 4. Additional penalties for getting too close to obstacles
    for category, rects in collision_boundaries.items():
        if category in ["goal_boundaries", "agent_boundaries"]:
        continue  # Skip these categories since their logic is already handled by agent_rects and goal_rects
        # Handle lists of rectangles
        for rect in rects:
            if agent_rect.colliderect(rect):
                reward -= penalty_proximity_obstacle
                print(f"Agent {agent} is too close to {category}!")

    """""

    
    # 5. Positive reward or Bonus reward for reaching the goal
    if distance_to_goal <= goal_radius:
        reward += reward_goal_reached

    return reward




#---------------------------------------Utility Functions for PYTORCH Management------------------------------------------------

def convert_to_torch(batch, device="cuda"):
    """Convert NumPy arrays in a batch to PyTorch tensors on the specified device."""
    torch_batch = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            torch_batch[key] = torch.tensor(value, dtype=torch.float, device=device)
        else:
            torch_batch[key] = value
    return torch_batch

import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.fc = nn.Linear(obs_space.shape[0], 64)
        self.action_fc = nn.Linear(64, action_space.n)
        self.value_fc = nn.Linear(64, 1)

    def forward(self, obs):
        x = torch.relu(self.fc(obs))
        action_logits = self.action_fc(x)
        value = self.value_fc(x)
        return action_logits, value  # Return both action logits and value function predictions


#---------------------------------------Utility Functions for Checkpoint Management------------------------------------------------


def create_checkpoint_dir(base_path="checkpoints", session_name="session_01"):
    session_path = os.path.join(base_path, session_name)
    os.makedirs(session_path, exist_ok=True)
    return session_path



def save_checkpoint(model, optimizer, epoch, cumulative_reward, session_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "cumulative_reward": cumulative_reward,
    }
    checkpoint_path = os.path.join(session_path, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    cumulative_reward = checkpoint["cumulative_reward"]
    print(f"Checkpoint loaded: {checkpoint_path} (Epoch {epoch}, Reward: {cumulative_reward})")
    return epoch, cumulative_reward




#---------------------------------------DEBUGING UTILS------------------------------------------------


def debug_collision_boundaries(collision_boundaries):
    """
    Print the contents of collision boundaries with centers for each category.

    Args:
        collision_boundaries (dict): Dictionary containing collision boundaries for all categories.

    Usage:
        used by "init" method inside the main Environment class. 
    """
    print("=" * 100)
    print("Contents stored in self.collision_boundaries for each category:")

    for category, boundaries in collision_boundaries.items():
        print(f"{category}:")
        
        if isinstance(boundaries, dict):  # For goal_boundaries (mapped by agent IDs)
            for agent_id, rect in boundaries.items():
                center = rect.center if hasattr(rect, "center") else "N/A"
                print(f"  {agent_id}: <{rect}>, Center: {center}")
        elif isinstance(boundaries, list):  # For other categories (list of pygame.Rect)
            for idx, rect in enumerate(boundaries):
                center = rect.center if hasattr(rect, "center") else "N/A"
                print(f"  Rectangle {idx}: <{rect}>, Center: {center}")
        else:
            print("  Unknown data type for boundaries")
    
    print("=" * 100)



def debug_all_observations(obs):
    """
    Debug and print observations in a structured format for all agents,
    handling cases where the input is tuple-wrapped.


    Usage:
    used by "_compute_agent_observation" Function inside Env-helpers.py.
    
    """
    # If the observation is a tuple, extract its first element
    if isinstance(obs, tuple):
        print("[DEBUG] Observation is wrapped in a tuple. Extracting the first element.")
        obs = obs[0]

    print("\n" + "=" * 80)
    print("[DEBUG] Agent Observations")
    print("=" * 80)

    # Process the extracted dictionary
    for agent_id, agent_obs in obs.items():
        print(f"\n[DEBUG] Observation for {agent_id}:\n")
        
        if isinstance(agent_obs, dict):
            for key, value in agent_obs.items():
                print(f"  {key}:")
                
                # Handle arrays or lists with shapes
                if isinstance(value, (list, np.ndarray)):
                    array_value = np.array(value)
                    print(f"    Array shape: {array_value.shape}")
                    print(f"    {array_value}\n")
                
                # Handle None values
                elif value is None:
                    print("    None\n")
                
                # Handle nested dictionaries
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}:")
                        if isinstance(sub_value, (list, np.ndarray)):
                            array_value = np.array(sub_value)
                            print(f"      Array shape: {array_value.shape}")
                            print(f"      {array_value}\n")
                        elif sub_value is None:
                            print("      None\n")
                        else:
                            print(f"      {sub_value}\n")
                
                # Handle other data types
                else:
                    print(f"    {value}\n")
        else:
            print(f"  Observation is not a dictionary! Type: {type(agent_obs).__name__}\n")

    print("=" * 80 + "\n")


# Debugging: Log environment's observations in detail
def log_observations(observations, method_name):
    """
    Logs detailed observations for each agent.
    
    :param observations: Dictionary of observations keyed by agent_id.
    :param method_name: String, either 'reset' or 'step', indicating the source of the observation.
    """
    for agent_id, obs in observations.items():
        logger.debug(f"{'-'*50} DEBUGGING OBSERVATIONS {'-'*50}")
        logger.debug(f"Source: {method_name} method | Agent ID: {agent_id}")
        logger.debug(f"Observation Structure: {type(obs)}")
        
        if isinstance(obs, dict):
            # Log each key-value pair in the observation
            for key, value in obs.items():
                logger.debug(f"  {key}: {type(value)}")
                if isinstance(value, (np.ndarray, list)):
                    logger.debug(f"    Shape: {np.shape(value)} | Values: {np.array(value)}")
                else:
                    logger.debug(f"    Value: {value}")
        else:
            logger.debug(f"  Observation Value: {obs}")
        
        logger.debug(f"{'-'*100}")



# Debugging: Log environment's observation spaces in detail
def log_observation_spaces(observation_spaces):
    """
    Logs detailed information about the observation spaces for each agent.
    
    :param observation_spaces: Dictionary of observation spaces keyed by agent_id, or a single observation space.
    """
    if isinstance(observation_spaces, dict):
        logger.debug(f"{'-'*50} DEBUGGING OBSERVATION SPACES {'-'*50}")
        for agent_id, space in observation_spaces.items():
            logger.debug(f"Agent ID: {agent_id}")
            _log_space_details(space)
    else:
        logger.debug(f"{'-'*50} DEBUGGING SINGLE OBSERVATION SPACE {'-'*50}")
        _log_space_details(observation_spaces)

def _log_space_details(space):
    """
    Logs detailed information about a single gym space.
    
    :param space: A `gym.spaces.Space` object.
    """
    logger.debug(f"Space Type: {type(space).__name__}")
    
    if isinstance(space, gym.spaces.Box):
        logger.debug(f"  Box Shape: {space.shape}")
        logger.debug(f"  Box Low: {space.low}")
        logger.debug(f"  Box High: {space.high}")
    elif isinstance(space, gym.spaces.Discrete):
        logger.debug(f"  Discrete N: {space.n}")
    elif isinstance(space, gym.spaces.Dict):
        logger.debug("  Dict Observation Space:")
        for key, subspace in space.spaces.items():
            logger.debug(f"    Key: {key}")
            _log_space_details(subspace)
    elif isinstance(space, gym.spaces.Tuple):
        logger.debug("  Tuple Observation Space:")
        for idx, subspace in enumerate(space.spaces):
            logger.debug(f"    Index: {idx}")
            _log_space_details(subspace)
    else:
        logger.debug(f"  Space Details: {space}")
    logger.debug(f"{'-'*100}")



#--------------------------------------- TO-DO SECTION UTILS------------------------------------------------


def _has_reached_goal(self, agent):
    # Check if an agent has reached its goal (within a threshold)
    x, y = self.agents_positions[agent]
    goal_x, goal_y = self.agents_goals[agent]
    threshold = 0.5  # Set a small threshold for goal reaching
    return np.sqrt((x - goal_x)**2 + (y - goal_y)**2) < threshold


        

