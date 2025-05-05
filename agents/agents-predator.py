"""
Predator agent module for the predator-prey simulation.

This module defines the Predator agent class, which gains energy by
catching and eating prey agents.
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
from agents.base import Agent


class Predator(Agent):
    """
    Predator agent that hunts prey for energy.
    
    Attributes:
        energy_decay_rate (float): Energy lost per timestep.
    """
    
    def __init__(self, position: np.ndarray, energy: float = 150.0):
        """
        Initialize a predator agent.
        
        Args:
            position (np.ndarray): Initial position.
            energy (float): Initial energy level.
        """
        super().__init__(position, energy)
        self.type = "predator"
        
        # Predator-specific attributes
        self.energy_decay_rate = 0.5  # Additional energy lost per timestep
        self.reproduction_threshold = 200.0  # Energy needed to reproduce
        self.reproduction_cost = 100.0  # Energy lost during reproduction
        self.reproduction_cooldown = 30  # Timesteps between reproduction attempts
        self.initial_energy = 150.0  # Starting energy for offspring
        
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
        norm_energy = energy / 300.0  # Assuming max energy of 300
        
        # Combine all features
        obs_vector = np.array(vision_flat + [norm_energy], dtype=np.float32)
        return obs_vector
    
    def _heuristic_action(self, observation: Dict[str, Any]) -> int:
        """
        Simple heuristic policy for predator behavior.
        
        Args:
            observation (Dict[str, Any]): Observation of the environment.
            
        Returns:
            int: Action index.
        """
        # Check for nearby prey
        vision = observation["vision"]
        prey_detected = False
        prey_direction = np.zeros(2)
        
        for i, ray in enumerate(vision):
            if ray["type"] == "prey" and ray["distance"] < 50:
                prey_detected = True
                angle = 2 * np.pi * i / len(vision)
                direction = np.array([np.cos(angle), np.sin(angle)])
                # Closer prey have more influence
                weight = (50 - ray["distance"]) / 50
                prey_direction += direction * weight
        
        if prey_detected:
            # Move toward prey
            if np.linalg.norm(prey_direction) > 0:
                hunt_direction = prey_direction / np.linalg.norm(prey_direction)
                
                # Map direction to action
                if abs(hunt_direction[0]) > abs(hunt_direction[1]):
                    # Horizontal movement
                    return 3 if hunt_direction[0] > 0 else 4  # Right or Left
                else:
                    # Vertical movement
                    return 2 if hunt_direction[1] > 0 else 1  # Down or Up
        
        # No prey nearby, random exploration
        return np.random.randint(1, 5)  # Avoid staying still
    
    def update(self, observation: Dict[str, Any]) -> None:
        """
        Update agent state after action.
        
        For predator, lose energy over time.
        
        Args:
            observation (Dict[str, Any]): Current observation.
        """
        super().update(observation)
        
        # Predators lose energy over time regardless of movement
        self.apply_energy_cost(self.energy_decay_rate)
    
    def get_color(self) -> Tuple[int, int, int]:
        """
        Get the color for rendering this predator agent.
        
        Returns:
            Tuple[int, int, int]: RGB color values.
        """
        # Red color for predator, intensity based on energy level
        energy_ratio = min(1.0, self.energy / 200.0)
        red = int(100 + energy_ratio * 155)
        return (red, 0, 0)