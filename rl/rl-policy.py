"""
Policy network implementation for PPO reinforcement learning.

This module defines the neural network policy used by agents to select actions
based on observations of the environment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class PPOPolicy(nn.Module):
    """
    Policy network for Proximal Policy Optimization.
    
    This network has two heads:
    - Actor: outputs action probabilities
    - Critic: outputs state value
    
    Attributes:
        actor_mean (nn.Sequential): Actor network that outputs action logits.
        critic (nn.Sequential): Critic network that outputs state value.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initialize the policy network.
        
        Args:
            obs_dim (int): Dimension of observation space.
            action_dim (int): Dimension of action space.
            hidden_dim (int): Dimension of hidden layers.
        """
        super(PPOPolicy, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input observation tensor.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Action probabilities and state value.
        """
        # Convert input to tensor if it's not already
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Shared features
        features = self.shared(x)
        
        # Actor: action probabilities
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic: state value
        value = self.critic(features)
        
        return action_probs, value
    
    def get_action_and_value(self, x: torch.Tensor, 
                            action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value for a given observation.
        
        Args:
            x (torch.Tensor): Input observation tensor.
            action (torch.Tensor, optional): Action to compute log probability for.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Action, log probability, and value.
        """
        # Forward pass
        action_probs, value = self(x)
        
        # Sample from the action distribution
        dist = torch.distributions.Categorical(action_probs)
        
        # Get action (either sample or use provided)
        if action is None:
            action = dist.sample()
        
        # Compute log probability
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def get_deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the deterministic (most likely) action for a given observation.
        
        Args:
            x (torch.Tensor): Input observation tensor.
            
        Returns:
            torch.Tensor: Most likely action.
        """
        # Forward pass
        action_probs, _ = self(x)
        
        # Get most likely action
        action = torch.argmax(action_probs, dim=-1)
        
        return action
    
    def evaluate_actions(self, x: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for a batch of observations.
        
        Args:
            x (torch.Tensor): Batch of observations.
            actions (torch.Tensor): Batch of actions to evaluate.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Log probs, values, entropy.
        """
        # Forward pass
        action_probs, value = self(x)
        
        # Create distribution
        dist = torch.distributions.Categorical(action_probs)
        
        # Compute log probabilities
        log_probs = dist.log_prob(actions)
        
        # Compute entropy
        entropy = dist.entropy().mean()
        
        return log_probs, value.squeeze(), entropy
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for saving the policy.
        
        Returns:
            Dict[str, Any]: State dictionary.
        """
        return super().state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from a state dictionary.
        
        Args:
            state_dict (Dict[str, Any]): State dictionary to load.
        """
        super().load_state_dict(state_dict)