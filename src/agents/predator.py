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
        # If we have a policy, use it
        if self.policy is not None:
            # Convert observation to tensor format expected by policy
            obs_tensor = self._prepare_observation(observation)
            action = self.policy.get_deterministic_action(obs_tensor).item()
            return action
        
        # No policy yet, use random actions with a unique seed for each agent
        # Use a hash of the agent's id modulo 2^32-1 to stay within valid range
        seed = (hash(self.id) + observation["timestamp"]) % (2**32 - 1)
        rng = np.random.RandomState(seed)  # Create a separate random number generator
        return rng.randint(0, 5)  # Use the agent-specific RNG
    
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
    
    def _handle_interactions(self, agent: Agent) -> None:
        """
        Handle interactions between agents, such as predators eating prey.
        
        Args:
            agent (Agent): The agent to process interactions for.
        """
        # Skip if the agent isn't a predator
        if agent.type != "predator":
            return
        
        # Find nearby agents
        nearby_agents = self._get_nearby_agents(agent, 5.0)
        
        # Check for prey to eat
        for other in nearby_agents:
            if other.type == "prey" and other.alive:
                # Predator eats prey
                energy_gain = min(other.energy, 50)  # Cap energy gain
                agent.energy += energy_gain
                
                # Add RL reward for successful hunt
                # This is stored in the agent for the RL system to collect
                if hasattr(agent, 'last_reward'):
                    agent.last_reward = 2.0  # Significant positive reward for hunting
                
                # Mark prey as dead and remove
                other.alive = False
                self.remove_agent(other)
                
                # Add negative reward to prey (will be collected if using episodic RL)
                if hasattr(other, 'last_reward'):
                    other.last_reward = -1.0  # Negative reward for being eaten
                
                break  # Only eat one prey per step

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