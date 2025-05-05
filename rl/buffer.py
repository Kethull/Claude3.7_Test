"""
Experience buffer for reinforcement learning.

This module provides a buffer for storing agent experiences
(observations, actions, rewards, etc.) for training.
"""
import numpy as np
from typing import List, Tuple


class ExperienceBuffer:
    """
    Buffer for storing and sampling experiences for reinforcement learning.
    
    Attributes:
        capacity (int): Maximum number of experiences to store.
        obs_dim (int): Dimension of observation space.
        size (int): Current number of experiences in buffer.
    """
    
    def __init__(self, obs_dim: int, capacity: int = 1000):
        """
        Initialize the experience buffer.
        
        Args:
            obs_dim (int): Dimension of observation space.
            capacity (int): Maximum number of experiences to store.
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.size = 0
        self.pos = 0  # Position for next insertion
        
        # Allocate memory for experience storage
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
    
    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, 
           done: bool, log_prob: float, value: float) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            obs (np.ndarray): Observation.
            action (int): Action taken.
            reward (float): Reward received.
            next_obs (np.ndarray): Next observation.
            done (bool): Whether episode is done.
            log_prob (float): Log probability of the action.
            value (float): Value prediction.
        """
        # Store experience in buffer
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        
        # Update position and size
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size (int): Number of experiences to sample.
            
        Returns:
            Tuple[np.ndarray, ...]: Batch of experiences (obs, actions, rewards, next_obs, dones, log_probs, values).
        """
        # Sample random indices
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        
        # Get batch
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
            self.log_probs[indices],
            self.values[indices]
        )
    
    def get_all(self) -> Tuple[np.ndarray, ...]:
        """
        Get all experiences in the buffer.
        
        Returns:
            Tuple[np.ndarray, ...]: All experiences (obs, actions, rewards, next_obs, dones, log_probs, values).
        """
        return (
            self.obs[:self.size],
            self.actions[:self.size],
            self.rewards[:self.size],
            self.next_obs[:self.size],
            self.dones[:self.size],
            self.log_probs[:self.size],
            self.values[:self.size]
        )
    
    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.size = 0
        self.pos = 0