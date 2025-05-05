import pytest
import numpy as np
from engine.world import World
from agents.base import Agent


class DummyAgent(Agent):
    """Dummy agent implementation for testing."""
    
    def __init__(self, position, energy=100):
        super().__init__(position, energy)
        self.type = "dummy"
    
    def act(self, observation):
        return 0  # Always stay still
    
    def get_color(self):
        return (255, 255, 255)  # White color for dummy agents


class TestWorld:
    
    def test_world_initialization(self):
        """Test that world initializes with correct dimensions."""
        width, height = 100, 150
        world = World(width, height)
        assert world.width == width
        assert world.height == height
        assert len(world.agents) == 0
        assert world.timestep == 0
    
    def test_add_agent(self):
        """Test adding an agent to the world."""
        world = World(100, 100)
        agent = DummyAgent(np.array([50.0, 50.0]))
        world.add_agent(agent)
        assert len(world.agents) == 1
        assert agent in world.agents
    
    def test_remove_agent(self):
        """Test removing an agent from the world."""
        world = World(100, 100)
        agent = DummyAgent(np.array([50.0, 50.0]))
        world.add_agent(agent)
        assert len(world.agents) == 1
        
        world.remove_agent(agent)
        assert len(world.agents) == 0
    
    def test_wraparound_position(self):
        """Test that positions wrap around world boundaries."""
        world = World(100, 100)
        
        # Test wrapping beyond right edge
        wrapped_pos = world.wrap_position(np.array([120.0, 50.0]))
        assert wrapped_pos[0] == 20.0
        assert wrapped_pos[1] == 50.0
        
        # Test wrapping beyond left edge
        wrapped_pos = world.wrap_position(np.array([-20.0, 50.0]))
        assert wrapped_pos[0] == 80.0
        assert wrapped_pos[1] == 50.0
        
        # Test wrapping beyond top edge
        wrapped_pos = world.wrap_position(np.array([50.0, -20.0]))
        assert wrapped_pos[0] == 50.0
        assert wrapped_pos[1] == 80.0
        
        # Test wrapping beyond bottom edge
        wrapped_pos = world.wrap_position(np.array([50.0, 120.0]))
        assert wrapped_pos[0] == 50.0
        assert wrapped_pos[1] == 20.0
    
    def test_step_updates_timestep(self):
        """Test that stepping the world increments the timestep."""
        world = World(100, 100)
        initial_timestep = world.timestep
        world.step()
        assert world.timestep == initial_timestep + 1
    
    def test_step_moves_agents(self):
        """Test that stepping the world updates agent positions."""
        world = World(100, 100)
        agent = DummyAgent(np.array([50.0, 50.0]))
        world.add_agent(agent)
        
        # Override the agent's act method to always move right
        agent.act = lambda obs: 3  # Right action
        
        initial_pos = agent.position.copy()
        world.step()
        
        # Agent should have moved right
        assert not np.array_equal(agent.position, initial_pos)
        assert agent.position[0] > initial_pos[0]
    
    def test_agent_death_on_zero_energy(self):
        """Test that agents are removed when energy reaches zero."""
        world = World(100, 100)
        agent = DummyAgent(np.array([50.0, 50.0]), energy=1)
        world.add_agent(agent)
        
        # Energy drains with movement
        agent.act = lambda obs: 3  # Right action
        
        world.step()
        assert len(world.agents) == 0  # Agent should be dead and removed
    
    def test_get_agent_observations(self):
        """Test that agent observations are correctly generated."""
        world = World(100, 100)
        agent = DummyAgent(np.array([50.0, 50.0]))
        world.add_agent(agent)
        
        obs = world.get_observation(agent)
        assert obs is not None
        assert "vision" in obs
        assert "energy" in obs
        assert obs["energy"] == agent.energy
