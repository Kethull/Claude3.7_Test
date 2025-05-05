"""
Prey agent module for the predator-prey simulation.

This module defines the Prey agent class, which gets energy by staying still
and runs from predators.
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
from agents.base import Agent


class Prey(Agent):
    """
    Prey agent that gains energy by staying still and tries to avoid predators.
    
    Attributes:
        energy_gain_rate (float): Energy gained per timestep when staying still.
    """
    
    def __init__(self, position: np.ndarray, energy: float = 100.0):
        """
        Initialize a prey agent.
        
        Args:
            position (np.ndarray): Initial position.
            energy (float): Initial energy level.
        """
        super().__init__(position, energy)
        self.type = "prey"
        
        # Prey-specific attributes
        self.energy_gain_rate = 1.0  # Energy gained per timestep when staying still
        self.reproduction_threshold = 120.0  # Energy needed to reproduce
        self.reproduction_cost = 50.0  # Energy lost during reproduction
        self.reproduction_cooldown = 20  # Timesteps between reproduction attempts
        
        # Movement cost is defined in the World.move_agent method
        
        # Policy network (will be set by controller)
        self.policy = None
    
    def act(self, observation: Dict[str, Any]) -> int:
        """
        Decide on an action based on the current observation.
        
        Uses reinforcement learning policy if available, or random actions if not.
        No predefined behaviors - all strategies must be learned through RL.
        
        Args:
            observation (Dict[str, Any]): Observation of the environment.
            
        Returns:
            int: Action index (0: stay, 1: up, 2: down, 3: right, 4: left)
        """
        # If we have a policy, use it
        if self.policy is not None:
            # Convert observation to tensor format expected by policy
            obs_tensor = self._prepare_observation(observation)
            action = self.policy.get_deterministic_action(obs_tensor).item()
            return action
        
        # No policy yet, use completely random actions
        return np.random.randint(0, 5)
    
    def _prepare_observation(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Convert the observation dictionary to a flat vector for the policy.
        
        Args:
            observation (Dict[str, Any]): Raw observation.
            
        Returns:
            np.ndarray: Flat observation vector.
        """
        # Extract values from observation
        vision = observation["vision"]
        energy = observation["energy"]
        
        # Flatten vision rays
        vision_flat = []
        for ray in vision:
            # Normalize distance (0-1 range)
            norm_distance = ray["distance"] / 100.0  # Assuming max range of 100
            vision_flat.append(norm_distance)
            
            # One-hot encoding of type (none, prey, predator)
            if ray["type"] == "none":
                vision_flat.extend([1, 0, 0])
            elif ray["type"] == "prey":
                vision_flat.extend([0, 1, 0])
            elif ray["type"] == "predator":
                vision_flat.extend([0, 0, 1])
            else:
                vision_flat.extend([0, 0, 0])
        
        # Normalize energy (0-1 range)
        norm_energy = energy / 200.0  # Assuming max energy of 200
        
        # Combine all features
        obs_vector = np.array(vision_flat + [norm_energy], dtype=np.float32)
        return obs_vector
    
    def _heuristic_action(self, observation: Dict[str, Any]) -> int:
        """
        Simple heuristic policy for prey behavior.
        
        Args:
            observation (Dict[str, Any]): Observation of the environment.
            
        Returns:
            int: Action index.
        """
        # Check for nearby predators
        vision = observation["vision"]
        predator_detected = False
        predator_direction = np.zeros(2)
        
        for i, ray in enumerate(vision):
            if ray["type"] == "predator" and ray["distance"] < 50:
                predator_detected = True
                angle = 2 * np.pi * i / len(vision)
                direction = np.array([np.cos(angle), np.sin(angle)])
                # Closer predators have more influence
                weight = (50 - ray["distance"]) / 50
                predator_direction += direction * weight
        
        if predator_detected:
            # Run away from predators (opposite direction)
            if np.linalg.norm(predator_direction) > 0:
                flee_direction = -predator_direction / np.linalg.norm(predator_direction)
                
                # Map direction to action
                if abs(flee_direction[0]) > abs(flee_direction[1]):
                    # Horizontal movement
                    return 4 if flee_direction[0] < 0 else 3  # Left or Right
                else:
                    # Vertical movement
                    return 1 if flee_direction[1] < 0 else 2  # Up or Down
        
        # No predators nearby, stay still to gain energy or explore
        # 90% chance to stay still if below reproduction threshold
        if observation["energy"] < self.reproduction_threshold and np.random.random() < 0.9:
            return 0  # Stay still
        
        # Otherwise, random movement
        return np.random.randint(0, 5)
    
    def update(self, observation: Dict[str, Any]) -> None:
        """
        Update agent state after action.
        
        For prey, gain energy if staying still.
        
        Args:
            observation (Dict[str, Any]): Current observation.
        """
        super().update(observation)
        
        # Gain energy if staying still (action 0)
        last_action = observation.get("last_action", None)
        if last_action == 0:  # Stay action
            self.add_energy(self.energy_gain_rate)
    
    def get_color(self) -> Tuple[int, int, int]:
        """
        Get the color for rendering this prey agent.
        
        Returns:
            Tuple[int, int, int]: RGB color values.
        """
        # Green color for prey, intensity based on energy level
        energy_ratio = min(1.0, self.energy / 150.0)
        green = int(100 + energy_ratio * 155)
        return (0, green, 0)