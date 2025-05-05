import pytest
import os
import numpy as np
import torch
import h5py
from utils.storage import SimulationStorage
from engine.world import World
from agents.prey import Prey
from agents.predator import Predator
from rl.policy import PPOPolicy


class TestSimulationStorage:
    
    @pytest.fixture
    def setup_world_and_agents(self):
        """Setup a small world with some agents for testing."""
        world = World(100, 100)
        
        # Add some prey
        for i in range(5):
            prey = Prey(np.array([20.0 * i, 30.0]), energy=100)
            world.add_agent(prey)
        
        # Add some predators
        for i in range(3):
            predator = Predator(np.array([20.0 * i, 60.0]), energy=150)
            world.add_agent(predator)
        
        # Create policies for agents
        obs_dim = 20
        action_dim = 5
        prey_policy = PPOPolicy(obs_dim, action_dim)
        predator_policy = PPOPolicy(obs_dim, action_dim)
        
        return world, prey_policy, predator_policy
    
    def test_storage_initialization(self):
        """Test that simulation storage initializes correctly."""
        filename = "test_sim.h5"
        
        # Remove file if it exists
        if os.path.exists(filename):
            os.remove(filename)
        
        storage = SimulationStorage(filename)
        
        assert storage.filename == filename
        assert not storage.is_open()
    
    def test_save_world_state(self, setup_world_and_agents):
        """Test saving world state to HDF5 file."""
        world, prey_policy, predator_policy = setup_world_and_agents
        filename = "test_sim_world.h5"
        
        # Remove file if it exists
        if os.path.exists(filename):
            os.remove(filename)
        
        storage = SimulationStorage(filename)
        storage.open('w')
        
        # Save world state
        storage.save_world_state(world, 0)
        
        # Close storage
        storage.close()
        
        # Check that file exists
        assert os.path.exists(filename)
        
        # Open file and check contents
        with h5py.File(filename, 'r') as f:
            assert 'world' in f
            assert 'timestep_0' in f['world']
            assert 'positions' in f['world']['timestep_0']
            assert 'energies' in f['world']['timestep_0']
            assert 'types' in f['world']['timestep_0']
            
            # Check agent count
            positions = f['world']['timestep_0']['positions'][:]
            assert len(positions) == len(world.agents)
        
        # Cleanup
        os.remove(filename)
    
    def test_save_policy_weights(self, setup_world_and_agents):
        """Test saving policy weights to HDF5 file."""
        world, prey_policy, predator_policy = setup_world_and_agents
        filename = "test_sim_policy.h5"
        
        # Remove file if it exists
        if os.path.exists(filename):
            os.remove(filename)
        
        storage = SimulationStorage(filename)
        storage.open('w')
        
        # Save policy weights
        storage.save_policy_weights(prey_policy, 'prey', 0)
        storage.save_policy_weights(predator_policy, 'predator', 0)
        
        # Close storage
        storage.close()
        
        # Check that file exists
        assert os.path.exists(filename)
        
        # Open file and check contents
        with h5py.File(filename, 'r') as f:
            assert 'policies' in f
            assert 'prey_0' in f['policies']
            assert 'predator_0' in f['policies']
            
            # Check that state_dict keys exist for prey policy
            prey_group = f['policies']['prey_0']
            for name, param in prey_policy.state_dict().items():
                assert name in prey_group
        
        # Cleanup
        os.remove(filename)
    
    def test_load_world_state(self, setup_world_and_agents):
        """Test loading world state from HDF5 file."""
        world, prey_policy, predator_policy = setup_world_and_agents
        filename = "test_sim_load.h5"
        
        # Remove file if it exists
        if os.path.exists(filename):
            os.remove(filename)
        
        # Save world state
        storage = SimulationStorage(filename)
        storage.open('w')
        storage.save_world_state(world, 0)
        storage.close()
        
        # Create new empty world
        new_world = World(100, 100)
        
        # Load world state
        storage.open('r')
        storage.load_world_state(new_world, 0)
        storage.close()
        
        # Check that agents were loaded
        assert len(new_world.agents) == len(world.agents)
        
        # Count prey and predators
        orig_prey_count = sum(1 for a in world.agents if a.type == 'prey')
        orig_pred_count = sum(1 for a in world.agents if a.type == 'predator')
        
        new_prey_count = sum(1 for a in new_world.agents if a.type == 'prey')
        new_pred_count = sum(1 for a in new_world.agents if a.type == 'predator')
        
        assert new_prey_count == orig_prey_count
        assert new_pred_count == orig_pred_count
        
        # Cleanup
        os.remove(filename)
    
    def test_load_policy_weights(self, setup_world_and_agents):
        """Test loading policy weights from HDF5 file."""
        world, prey_policy, predator_policy = setup_world_and_agents
        filename = "test_sim_load_policy.h5"
        
        # Remove file if it exists
        if os.path.exists(filename):
            os.remove(filename)
        
        # Save policy weights
        storage = SimulationStorage(filename)
        storage.open('w')
        storage.save_policy_weights(prey_policy, 'prey', 0)
        storage.close()
        
        # Create new policy with different weights
        obs_dim = 20
        action_dim = 5
        new_policy = PPOPolicy(obs_dim, action_dim)
        
        # Verify policies are different
        for (name1, param1), (name2, param2) in zip(
            prey_policy.state_dict().items(), new_policy.state_dict().items()
        ):
            assert not torch.allclose(param1, param2)
        
        # Load policy weights
        storage.open('r')
        storage.load_policy_weights(new_policy, 'prey', 0)
        storage.close()
        
        # Verify policies are now identical
        for (name1, param1), (name2, param2) in zip(
            prey_policy.state_dict().items(), new_policy.state_dict().items()
        ):
            assert torch.allclose(param1, param2)
        
        # Cleanup
        os.remove(filename)
    
    def test_export_stats_to_csv(self):
        """Test exporting simulation statistics to CSV file."""
        filename = "test_sim_stats.h5"
        csv_filename = "test_sim_stats.csv"
        
        # Remove files if they exist
        for f in [filename, csv_filename]:
            if os.path.exists(f):
                os.remove(f)
        
        storage = SimulationStorage(filename)
        storage.open('w')
        
        # Add some stats
        timesteps = 100
        for t in range(timesteps):
            stats = {
                'prey_count': 100 - t // 2,
                'predator_count': 50 + t // 4,
                'prey_reward': -0.1 * t,
                'predator_reward': 0.2 * t
            }
            storage.save_stats(stats, t)
        
        storage.close()
        
        # Export stats to CSV
        storage.export_stats_to_csv(csv_filename)
        
        # Check that CSV file exists
        assert os.path.exists(csv_filename)
        
        # Read CSV and check contents
        import pandas as pd
        df = pd.read_csv(csv_filename)
        
        assert len(df) == timesteps
        assert 'timestep' in df.columns
        assert 'prey_count' in df.columns
        assert 'predator_count' in df.columns
        assert 'prey_reward' in df.columns
        assert 'predator_reward' in df.columns
        
        # Cleanup
        for f in [filename, csv_filename]:
            if os.path.exists(f):
                os.remove(f)
