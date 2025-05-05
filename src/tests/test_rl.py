import pytest
import numpy as np
import torch
from rl.policy import PPOPolicy
from rl.buffer import ExperienceBuffer
from rl.ppo import PPOTrainer


class TestPPOPolicy:
    
    def test_policy_initialization(self):
        """Test that PPO policy model initializes with correct input/output dimensions."""
        obs_dim = 20  # Example: 8 vision rays with distance and type (16) + energy (1) + other (3)
        action_dim = 5  # up, down, left, right, stay
        
        policy = PPOPolicy(obs_dim, action_dim)
        
        # Check that the policy network exists
        assert hasattr(policy, 'actor')
        assert hasattr(policy, 'critic')
        
        # Create dummy observation
        obs = torch.randn(1, obs_dim)
        
        # Test forward pass
        action_probs, value = policy(obs)
        
        assert action_probs.shape == (1, action_dim)
        assert value.shape == (1, 1)
        
        # Test that action probs sum to 1
        assert torch.isclose(action_probs.sum(), torch.tensor(1.0), atol=1e-6)
    
    def test_policy_action_selection(self):
        """Test that policy can select actions based on observations."""
        obs_dim = 20
        action_dim = 5
        
        policy = PPOPolicy(obs_dim, action_dim)
        
        # Create dummy observation
        obs = torch.randn(1, obs_dim)
        
        # Get action
        action, log_prob, value = policy.get_action_and_value(obs)
        
        assert action.item() in range(action_dim)
        assert log_prob.shape == (1,)
        assert value.shape == (1, 1)
    
    def test_policy_deterministic_action(self):
        """Test that policy can select deterministic actions for evaluation."""
        obs_dim = 20
        action_dim = 5
        
        policy = PPOPolicy(obs_dim, action_dim)
        
        # Create dummy observation
        obs = torch.randn(1, obs_dim)
        
        # Get deterministic action
        action = policy.get_deterministic_action(obs)
        
        assert action.item() in range(action_dim)
        
        # Test multiple calls with same observation return same action
        action2 = policy.get_deterministic_action(obs)
        assert action.item() == action2.item()


class TestExperienceBuffer:
    
    def test_buffer_initialization(self):
        """Test that experience buffer initializes with correct capacity."""
        obs_dim = 20
        action_dim = 5
        capacity = 1000
        
        buffer = ExperienceBuffer(obs_dim, capacity)
        
        assert buffer.obs_dim == obs_dim
        assert buffer.capacity == capacity
        assert buffer.size == 0
    
    def test_buffer_add_experience(self):
        """Test that experiences can be added to the buffer."""
        obs_dim = 20
        capacity = 1000
        
        buffer = ExperienceBuffer(obs_dim, capacity)
        
        # Create dummy experience
        obs = np.random.randn(obs_dim)
        action = 2
        reward = 1.0
        next_obs = np.random.randn(obs_dim)
        done = False
        log_prob = -0.5
        value = 0.7
        
        buffer.add(obs, action, reward, next_obs, done, log_prob, value)
        
        assert buffer.size == 1
        assert np.array_equal(buffer.obs[0], obs)
        assert buffer.actions[0] == action
        assert buffer.rewards[0] == reward
        assert np.array_equal(buffer.next_obs[0], next_obs)
        assert buffer.dones[0] == done
        assert buffer.log_probs[0] == log_prob
        assert buffer.values[0] == value
    
    def test_buffer_overflow(self):
        """Test that buffer handles overflow by overwriting oldest experiences."""
        obs_dim = 20
        capacity = 2  # Small capacity for testing overflow
        
        buffer = ExperienceBuffer(obs_dim, capacity)
        
        # Add 3 experiences (exceeding capacity)
        for i in range(3):
            obs = np.ones(obs_dim) * i
            buffer.add(obs, i, i, obs, False, i, i)
        
        assert buffer.size == capacity
        
        # Check that oldest experience was overwritten
        assert np.array_equal(buffer.obs[0], np.ones(obs_dim) * 1)
        assert np.array_equal(buffer.obs[1], np.ones(obs_dim) * 2)
    
    def test_buffer_get_batch(self):
        """Test that random batches can be sampled from the buffer."""
        obs_dim = 20
        capacity = 100
        buffer = ExperienceBuffer(obs_dim, capacity)
        
        # Add some experiences
        for i in range(10):
            obs = np.ones(obs_dim) * i
            buffer.add(obs, i % 5, i, obs, i % 2 == 0, -0.5, 0.7)
        
        # Sample a batch
        batch_size = 5
        batch = buffer.get_batch(batch_size)
        
        assert len(batch) == 7  # obs, actions, rewards, next_obs, dones, log_probs, values
        assert batch[0].shape == (batch_size, obs_dim)  # obs
        assert batch[1].shape == (batch_size,)  # actions
        assert batch[2].shape == (batch_size,)  # rewards
        assert batch[3].shape == (batch_size, obs_dim)  # next_obs
        assert batch[4].shape == (batch_size,)  # dones
        assert batch[5].shape == (batch_size,)  # log_probs
        assert batch[6].shape == (batch_size,)  # values
    
    def test_buffer_clear(self):
        """Test that buffer can be cleared."""
        obs_dim = 20
        capacity = 100
        buffer = ExperienceBuffer(obs_dim, capacity)
        
        # Add some experiences
        for i in range(10):
            obs = np.ones(obs_dim) * i
            buffer.add(obs, i % 5, i, obs, i % 2 == 0, -0.5, 0.7)
        
        assert buffer.size == 10
        
        # Clear the buffer
        buffer.clear()
        
        assert buffer.size == 0


class TestPPOTrainer:
    
    def test_trainer_initialization(self):
        """Test that PPO trainer initializes with correct parameters."""
        obs_dim = 20
        action_dim = 5
        lr = 3e-4
        gamma = 0.99
        
        trainer = PPOTrainer(obs_dim, action_dim, lr=lr, gamma=gamma)
        
        assert hasattr(trainer, 'policy')
        assert hasattr(trainer, 'optimizer')
        assert trainer.lr == lr
        assert trainer.gamma == gamma
    
    def test_compute_advantages(self):
        """Test that advantages are correctly computed."""
        obs_dim = 20
        action_dim = 5
        trainer = PPOTrainer(obs_dim, action_dim)
        
        # Create dummy rewards and values
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        next_values = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0])  # Last state has zero value
        dones = torch.tensor([False, False, False, False, True])
        
        # Compute advantages
        advantages = trainer.compute_advantages(rewards, values, next_values, dones)
        
        assert advantages.shape == rewards.shape
        
        # Last advantage should be rewards - values (since done is True)
        assert torch.isclose(advantages[-1], rewards[-1] - values[-1])
    
    def test_update_policy(self):
        """Test that policy can be updated with a batch of experiences."""
        obs_dim = 20
        action_dim = 5
        batch_size = 32
        trainer = PPOTrainer(obs_dim, action_dim)
        
        # Create dummy batch
        obs = torch.randn(batch_size, obs_dim)
        actions = torch.randint(0, action_dim, (batch_size,))
        log_probs = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        
        # Initial policy parameters
        initial_params = [param.clone() for param in trainer.policy.parameters()]
        
        # Update policy
        stats = trainer.update_policy(obs, actions, log_probs, returns, advantages)
        
        # Check that policy was updated
        for i, param in enumerate(trainer.policy.parameters()):
            assert not torch.allclose(param, initial_params[i])
        
        # Check that stats were returned
        assert 'policy_loss' in stats
        assert 'value_loss' in stats
        assert 'entropy' in stats
