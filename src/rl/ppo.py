"""
Proximal Policy Optimization (PPO) implementation.

This module provides a PPO trainer for updating agent policies based on
collected experiences.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from rl.policy import PPOPolicy


class PPOTrainer:
    """
    Trainer for Proximal Policy Optimization algorithm.
    
    PPO is an on-policy algorithm that clips the policy gradient to
    prevent large policy updates, improving stability.
    
    Attributes:
        policy (PPOPolicy): Policy network to train.
        lr (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Clipping parameter.
        value_coef (float): Value loss coefficient.
        entropy_coef (float): Entropy coefficient.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99,
                 epsilon: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01):
        """
        Initialize the PPO trainer.
        
        Args:
            obs_dim (int): Dimension of observation space.
            action_dim (int): Dimension of action space.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Clipping parameter.
            value_coef (float): Value loss coefficient.
            entropy_coef (float): Entropy coefficient.
        """
        self.policy = PPOPolicy(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor,
                          next_values: torch.Tensor, dones: torch.Tensor,
                          gae_lambda: float = 0.95) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards (torch.Tensor): Batch of rewards.
            values (torch.Tensor): Batch of value predictions.
            next_values (torch.Tensor): Batch of next value predictions.
            dones (torch.Tensor): Batch of done flags.
            gae_lambda (float): GAE lambda parameter.
            
        Returns:
            torch.Tensor: Computed advantages.
        """
        # Initialize advantages
        advantages = torch.zeros_like(rewards)
        
        # Compute GAE iteratively
        gae = 0
        for t in reversed(range(len(rewards))):
            # If done, use only immediate reward
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                # TD error
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                # Update GAE
                gae = delta + self.gamma * gae_lambda * gae
            
            advantages[t] = gae
        
        return advantages
    
    def update_policy(self, obs: torch.Tensor, actions: torch.Tensor,
                     old_log_probs: torch.Tensor, returns: torch.Tensor,
                     advantages: torch.Tensor, epochs: int = 4) -> Dict[str, float]:
        """
        Update the policy network using PPO.
        
        Args:
            obs (torch.Tensor): Batch of observations.
            actions (torch.Tensor): Batch of actions.
            old_log_probs (torch.Tensor): Batch of log probabilities from old policy.
            returns (torch.Tensor): Batch of returns (discounted rewards).
            advantages (torch.Tensor): Batch of advantages.
            epochs (int): Number of optimization epochs.
            
        Returns:
            Dict[str, float]: Training statistics.
        """
        # Move data to device if necessary
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64).to(self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32).to(self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training stats
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0
        }
        
        # Perform multiple epochs of training
        for _ in range(epochs):
            # Get new log probs, values, and entropy
            new_log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)
            
            # Calculate ratio between new and old policies
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Calculate approximate KL divergence for early stopping
            approx_kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
            
            # Calculate clipping fraction
            clip_fraction = ((ratio < 1.0 - self.epsilon) | (ratio > 1.0 + self.epsilon)).float().mean().item()
            
            # Update stats
            stats['policy_loss'] += policy_loss.item() / epochs
            stats['value_loss'] += value_loss.item() / epochs
            stats['entropy'] += entropy.item() / epochs
            stats['approx_kl'] += approx_kl / epochs
            stats['clip_fraction'] += clip_fraction / epochs
            
            # Early stopping based on KL divergence
            if approx_kl > 0.015:
                break
        
        return stats
    
    def train_batch(self, batch: Tuple[np.ndarray, ...]) -> Dict[str, float]:
        """
        Train on a batch of experiences.
        
        Args:
            batch (Tuple[np.ndarray, ...]): Batch of experiences
                (obs, actions, rewards, next_obs, dones, log_probs, values).
            
        Returns:
            Dict[str, float]: Training statistics.
        """
        # Unpack batch
        obs, actions, rewards, next_obs, dones, log_probs, values = batch
        
        # Convert to tensors
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32).to(self.device)
        values_tensor = torch.as_tensor(values, dtype=torch.float32).to(self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32).to(self.device)
        
        # Get next state values
        with torch.no_grad():
            _, next_values = self.policy(next_obs_tensor)
            next_values = next_values.squeeze(-1)
        
        # Compute advantages
        advantages = self.compute_advantages(
            rewards_tensor, values_tensor, next_values, dones_tensor
        )
        
        # Compute returns
        returns = advantages + values_tensor
        
        # Update policy
        stats = self.update_policy(
            obs, actions, log_probs, returns, advantages
        )
        
        return stats
    
    def save(self, path: str) -> None:
        """
        Save the policy to a file.
        
        Args:
            path (str): Path to save to.
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load the policy from a file.
        
        Args:
            path (str): Path to load from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Add missing import
import torch.nn.functional as F