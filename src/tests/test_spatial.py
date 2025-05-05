import pytest
import numpy as np
from engine.spatial import QuadTree, SpatialGrid
from agents.base import Agent


class TestAgentForSpatial(Agent):
    """Simple agent implementation for spatial tests."""
    def __init__(self, position, energy=100, agent_id=None):
        super().__init__(position, energy)
        self.id = agent_id if agent_id is not None else id(self)
    
    def act(self, observation):
        return 0
    
    def get_color(self):
        return (255, 255, 255)


class TestQuadTree:
    
    def test_quadtree_initialization(self):
        """Test that quadtree initializes with correct boundaries."""
        boundary = (0, 0, 100, 100)  # x, y, width, height
        capacity = 4
        qt = QuadTree(boundary, capacity)
        
        assert qt.boundary == boundary
        assert qt.capacity == capacity
        assert len(qt.agents) == 0
        assert qt.divided is False
    
    def test_quadtree_insertion(self):
        """Test that agents can be inserted into the quadtree."""
        qt = QuadTree((0, 0, 100, 100), 4)
        agent = TestAgentForSpatial(np.array([50.0, 50.0]))
        
        success = qt.insert(agent)
        assert success is True
        assert len(qt.agents) == 1
    
    def test_quadtree_subdivision(self):
        """Test that quadtree subdivides when capacity is exceeded."""
        qt = QuadTree((0, 0, 100, 100), 2)  # Capacity of 2
        
        # Add 3 agents to force subdivision
        for i in range(3):
            agent = TestAgentForSpatial(np.array([50.0 + i, 50.0 + i]))
            qt.insert(agent)
        
        assert qt.divided is True
        assert hasattr(qt, 'northwest')
        assert hasattr(qt, 'northeast')
        assert hasattr(qt, 'southwest')
        assert hasattr(qt, 'southeast')
    
    def test_quadtree_query_range(self):
        """Test that agents within a range can be queried."""
        qt = QuadTree((0, 0, 100, 100), 4)
        
        # Add some agents
        agent1 = TestAgentForSpatial(np.array([25.0, 25.0]), agent_id=1)
        agent2 = TestAgentForSpatial(np.array([75.0, 75.0]), agent_id=2)
        agent3 = TestAgentForSpatial(np.array([10.0, 10.0]), agent_id=3)
        
        qt.insert(agent1)
        qt.insert(agent2)
        qt.insert(agent3)
        
        # Query range around agent1
        range_rect = (20, 20, 10, 10)  # x, y, width, height
        found = qt.query_range(range_rect)
        
        assert len(found) == 1
        assert found[0].id == 1
        
        # Larger query range
        range_rect = (0, 0, 50, 50)
        found = qt.query_range(range_rect)
        
        assert len(found) == 2
        assert agent1 in found
        assert agent3 in found
        assert agent2 not in found
    
    def test_quadtree_clear(self):
        """Test that quadtree can be cleared of all agents."""
        qt = QuadTree((0, 0, 100, 100), 2)
        
        # Add enough agents to force subdivision
        for i in range(5):
            agent = TestAgentForSpatial(np.array([20.0 * i, 20.0 * i]))
            qt.insert(agent)
        
        assert qt.divided is True
        
        # Clear the quadtree
        qt.clear()
        
        assert qt.divided is False
        assert len(qt.agents) == 0


class TestSpatialGrid:
    
    def test_grid_initialization(self):
        """Test that spatial grid initializes with correct dimensions."""
        width, height = 100, 100
        cell_size = 10
        grid = SpatialGrid(width, height, cell_size)
        
        assert grid.width == width
        assert grid.height == height
        assert grid.cell_size == cell_size
        assert grid.cols == width // cell_size
        assert grid.rows == height // cell_size
    
    def test_grid_insertion(self):
        """Test that agents can be inserted into the spatial grid."""
        grid = SpatialGrid(100, 100, 10)
        agent = TestAgentForSpatial(np.array([50.0, 50.0]))
        
        grid.insert(agent)
        
        # Get the cell coordinates for the agent's position
        col, row = grid.get_cell_indices(agent.position)
        cell_agents = grid.grid[row][col]
        
        assert agent in cell_agents
    
    def test_grid_query_radius(self):
        """Test that agents within a radius can be queried."""
        grid = SpatialGrid(100, 100, 10)
        
        # Add some agents
        agent1 = TestAgentForSpatial(np.array([50.0, 50.0]), agent_id=1)
        agent2 = TestAgentForSpatial(np.array([52.0, 52.0]), agent_id=2)  # Close to agent1
        agent3 = TestAgentForSpatial(np.array([80.0, 80.0]), agent_id=3)  # Far from agent1
        
        grid.insert(agent1)
        grid.insert(agent2)
        grid.insert(agent3)
        
        # Query agents near agent1
        nearby = grid.query_radius(agent1.position, 5.0)
        
        assert len(nearby) == 2  # Should find agent1 and agent2
        assert agent1 in nearby
        assert agent2 in nearby
        assert agent3 not in nearby
        
        # Larger radius
        nearby = grid.query_radius(agent1.position, 50.0)
        assert len(nearby) == 3  # Should find all agents
    
    def test_grid_update(self):
        """Test that the grid can be updated with new agent positions."""
        grid = SpatialGrid(100, 100, 10)
        agent = TestAgentForSpatial(np.array([50.0, 50.0]))
        
        grid.insert(agent)
        
        # Move the agent
        old_position = agent.position.copy()
        agent.position = np.array([80.0, 80.0])
        
        grid.update(agent, old_position)
        
        # Check old cell
        old_col, old_row = grid.get_cell_indices(old_position)
        old_cell_agents = grid.grid[old_row][old_col]
        assert agent not in old_cell_agents
        
        # Check new cell
        new_col, new_row = grid.get_cell_indices(agent.position)
        new_cell_agents = grid.grid[new_row][new_col]
        assert agent in new_cell_agents
    
    def test_grid_remove(self):
        """Test that agents can be removed from the spatial grid."""
        grid = SpatialGrid(100, 100, 10)
        agent = TestAgentForSpatial(np.array([50.0, 50.0]))
        
        grid.insert(agent)
        
        # Check that agent is in the grid
        col, row = grid.get_cell_indices(agent.position)
        cell_agents = grid.grid[row][col]
        assert agent in cell_agents
        
        # Remove the agent
        grid.remove(agent)
        
        # Check that agent is no longer in the grid
        cell_agents = grid.grid[row][col]
        assert agent not in cell_agents
    
    def test_grid_clear(self):
        """Test that the grid can be cleared of all agents."""
        grid = SpatialGrid(100, 100, 10)
        
        # Add some agents
        for i in range(5):
            agent = TestAgentForSpatial(np.array([20.0 * i, 20.0 * i]))
            grid.insert(agent)
        
        # Clear the grid
        grid.clear()
        
        # Check that all cells are empty
        for row in grid.grid:
            for cell in row:
                assert len(cell) == 0
