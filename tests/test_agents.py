import pytest
import numpy as np
from agents.base import Agent
from agents.prey import Prey
from agents.predator import Predator
from engine.world import World


class TestAgents:
    
    def test_base_agent_initialization(self):
        """Test that base agent initializes with correct position and energy."""
        position = np.array([50.0, 50.0])
        energy = 100
        agent = Agent(position, energy)
        assert np.array_equal(agent.position, position)
        assert agent.energy == energy
        assert agent.alive is True
    
    def test_agent_energy_depletion(self):
        """Test that agent energy depletes with actions and agent dies at zero energy."""
        agent = Agent(np.array([50.0, 50.0]), energy=10)
        
        # Apply energy cost
        agent.apply_energy_cost(5)
        assert agent.energy == 5
        assert agent.alive is True
        
        # Deplete remaining energy
        agent.apply_energy_cost(5)
        assert agent.energy == 0
        assert agent.alive is False
        
        # Further energy depletion should not go below zero
        agent.apply_energy_cost(5)
        assert agent.energy == 0
    
    def test_prey_initialization(self):
        """Test that prey agent initializes with correct values."""
        position = np.array([50.0, 50.0])
        energy = 100
        prey = Prey(position, energy)
        assert np.array_equal(prey.position, position)
        assert prey.energy == energy
        assert prey.type == "prey"
    
    def test_predator_initialization(self):
        """Test that predator agent initializes with correct values."""
        position = np.array([50.0, 50.0])
        energy = 100
        predator = Predator(position, energy)
        assert np.array_equal(predator.position, position)
        assert prey.energy == energy
        assert predator.type == "predator"
    
    def test_prey_energy_gain_when_still(self):
        """Test that prey gains energy when staying still."""
        world = World(100, 100)
        prey = Prey(np.array([50.0, 50.0]), energy=90)
        world.add_agent(prey)
        
        # Force prey to stay still
        prey.act = lambda obs: 0  # Stay action
        
        world.step()
        assert prey.energy > 90  # Energy should increase
    
    def test_predator_energy_gain_from_eating(self):
        """Test that predator gains energy when eating prey."""
        world = World(100, 100)
        predator = Predator(np.array([50.0, 50.0]), energy=90)
        prey = Prey(np.array([51.0, 50.0]), energy=100)  # Place prey very close
        
        world.add_agent(predator)
        world.add_agent(prey)
        
        initial_energy = predator.energy
        
        # Force predator to move right toward prey
        predator.act = lambda obs: 3  # Right action
        
        world.step()
        assert predator.energy > initial_energy  # Energy should increase from eating
        assert prey not in world.agents  # Prey should be consumed and removed
    
    def test_agent_reproduction(self):
        """Test that agents reproduce when energy exceeds threshold."""
        world = World(100, 100)
        
        # Create prey with energy just below reproduction threshold
        repro_threshold = 120
        prey = Prey(np.array([50.0, 50.0]), energy=repro_threshold - 5)
        prey.reproduction_threshold = repro_threshold
        world.add_agent(prey)
        
        # Force prey to stay still and gain energy
        prey.act = lambda obs: 0  # Stay action
        
        # Step until reproduction
        for _ in range(10):  # Should be enough steps to gain sufficient energy
            world.step()
            if len(world.agents) > 1:
                break
                
        assert len(world.agents) > 1  # New agent should be born
        
        # Verify the new agent is a prey with correct initial energy
        new_prey = [a for a in world.agents if a is not prey][0]
        assert new_prey.type == "prey"
        assert new_prey.energy == prey.initial_energy
