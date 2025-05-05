"""
Spatial partitioning data structures for efficient proximity queries.

This module provides spatial data structures (QuadTree and SpatialGrid) that
allow for O(log n) or O(1) lookups of nearby agents, instead of the O(n) brute
force approach of checking all other agents.
"""
import numpy as np
from typing import List, Tuple, Any


class QuadTree:
    """
    QuadTree spatial partitioning structure for efficient proximity queries.
    
    A quadtree recursively subdivides space into quadrants, allowing for
    efficient range queries.
    
    Attributes:
        boundary (Tuple[float, float, float, float]): x, y, width, height of region
        capacity (int): Maximum agents before subdividing
        agents (List): Agents in this node (if not subdivided)
        divided (bool): Whether this node has been subdivided
    """
    
    def __init__(self, boundary: Tuple[float, float, float, float], capacity: int):
        """
        Initialize a QuadTree node.
        
        Args:
            boundary (Tuple[float, float, float, float]): x, y, width, height
            capacity (int): Maximum agents per leaf node
        """
        self.boundary = boundary
        self.capacity = capacity
        self.agents = []
        self.divided = False
        
        # Child nodes (initialized when needed)
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None
    
    def insert(self, agent) -> bool:
        """
        Insert an agent into the quadtree.
        
        Args:
            agent: The agent to insert.
            
        Returns:
            bool: True if insertion was successful.
        """
        # Check if agent is within boundary
        x, y, w, h = self.boundary
        if not (x <= agent.position[0] < x + w and y <= agent.position[1] < y + h):
            return False
        
        # If space available in this node and not divided, add here
        if len(self.agents) < self.capacity and not self.divided:
            self.agents.append(agent)
            return True
        
        # Otherwise, subdivide if needed and insert into children
        if not self.divided:
            self._subdivide()
        
        # Try to insert into each child
        return (self.northwest.insert(agent) or
                self.northeast.insert(agent) or
                self.southwest.insert(agent) or
                self.southeast.insert(agent))
    
    def _subdivide(self) -> None:
        """
        Subdivide this node into four equal quadrants.
        """
        x, y, w, h = self.boundary
        half_w = w / 2
        half_h = h / 2
        
        # Create four children
        nw_boundary = (x, y, half_w, half_h)
        self.northwest = QuadTree(nw_boundary, self.capacity)
        
        ne_boundary = (x + half_w, y, half_w, half_h)
        self.northeast = QuadTree(ne_boundary, self.capacity)
        
        sw_boundary = (x, y + half_h, half_w, half_h)
        self.southwest = QuadTree(sw_boundary, self.capacity)
        
        se_boundary = (x + half_w, y + half_h, half_w, half_h)
        self.southeast = QuadTree(se_boundary, self.capacity)
        
        # Move existing agents to children
        for agent in self.agents:
            self.northwest.insert(agent) or \
            self.northeast.insert(agent) or \
            self.southwest.insert(agent) or \
            self.southeast.insert(agent)
        
        self.agents = []
        self.divided = True
    
    def query_range(self, range_rect: Tuple[float, float, float, float]) -> List[Any]:
        """
        Find all agents within a rectangular range.
        
        Args:
            range_rect (Tuple[float, float, float, float]): x, y, width, height
            
        Returns:
            List[Any]: Agents within the range.
        """
        found_agents = []
        
        # If range doesn't intersect boundary, return empty list
        if not self._intersects(range_rect, self.boundary):
            return found_agents
        
        # Check agents in this node
        for agent in self.agents:
            rx, ry, rw, rh = range_rect
            if rx <= agent.position[0] < rx + rw and ry <= agent.position[1] < ry + rh:
                found_agents.append(agent)
        
        # If not divided, we're done
        if not self.divided:
            return found_agents
        
        # Otherwise, check children
        found_agents.extend(self.northwest.query_range(range_rect))
        found_agents.extend(self.northeast.query_range(range_rect))
        found_agents.extend(self.southwest.query_range(range_rect))
        found_agents.extend(self.southeast.query_range(range_rect))
        
        return found_agents
    
    def _intersects(self, rect1: Tuple[float, float, float, float], 
                   rect2: Tuple[float, float, float, float]) -> bool:
        """
        Check if two rectangles intersect.
        
        Args:
            rect1 (Tuple[float, float, float, float]): x, y, width, height
            rect2 (Tuple[float, float, float, float]): x, y, width, height
            
        Returns:
            bool: True if rectangles intersect.
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    def clear(self) -> None:
        """
        Clear all agents from the quadtree.
        """
        self.agents = []
        
        if self.divided:
            self.northwest.clear()
            self.northeast.clear()
            self.southwest.clear()
            self.southeast.clear()
            
            self.northwest = None
            self.northeast = None
            self.southwest = None
            self.southeast = None
            
            self.divided = False


class SpatialGrid:
    """
    Grid-based spatial partitioning structure for efficient proximity queries.
    
    A uniform grid divides space into equal-sized cells, allowing for O(1)
    lookups of nearby agents.
    
    Attributes:
        width (int): Width of the world.
        height (int): Height of the world.
        cell_size (int): Size of each grid cell.
        grid (List[List[List]]): 2D grid of agent lists.
    """
    
    def __init__(self, width: int, height: int, cell_size: int):
        """
        Initialize a spatial grid.
        
        Args:
            width (int): Width of the world.
            height (int): Height of the world.
            cell_size (int): Size of each grid cell.
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Calculate grid dimensions
        self.cols = width // cell_size
        self.rows = height // cell_size
        
        # Initialize grid cells
        self.grid = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
    
    def get_cell_indices(self, position: np.ndarray) -> Tuple[int, int]:
        """
        Get the grid cell indices for a position.
        
        Args:
            position (np.ndarray): The position to check.
            
        Returns:
            Tuple[int, int]: Column and row indices.
        """
        try:
            # Ensure position is within world bounds
            if not (0 <= position[0] < self.width and 0 <= position[1] < self.height):
                print(f"WARNING: Position {position} outside world bounds ({self.width}x{self.height})")
                # Wrap around position
                position = np.array([position[0] % self.width, position[1] % self.height])
                
            col = min(int(position[0] // self.cell_size), self.cols - 1)
            row = min(int(position[1] // self.cell_size), self.rows - 1)
            return col, row
        except Exception as e:
            print(f"ERROR in get_cell_indices: {e}, position: {position}")
            # Return safe default values
            return 0, 0
    
    def insert(self, agent) -> None:
        """
        Insert an agent into the spatial grid.
        
        Args:
            agent: The agent to insert.
        """
        try:
            col, row = self.get_cell_indices(agent.position)
            if col < 0 or col >= self.cols or row < 0 or row >= self.rows:
                print(f"ERROR: Invalid cell indices: ({col}, {row}) for position {agent.position}")
                return
            
            if agent not in self.grid[row][col]:
                self.grid[row][col].append(agent)
        except Exception as e:
            print(f"ERROR in spatial grid insert: {e}, agent position: {agent.position}, type: {agent.type}")
    
    def remove(self, agent) -> None:
        """
        Remove an agent from the spatial grid.
        
        Args:
            agent: The agent to remove.
        """
        col, row = self.get_cell_indices(agent.position)
        if agent in self.grid[row][col]:
            self.grid[row][col].remove(agent)
    
    def update(self, agent, old_position: np.ndarray) -> None:
        """
        Update an agent's position in the spatial grid.
        
        Args:
            agent: The agent to update.
            old_position (np.ndarray): The agent's previous position.
        """
        old_col, old_row = self.get_cell_indices(old_position)
        new_col, new_row = self.get_cell_indices(agent.position)
        
        # If cell changed, update grid
        if old_col != new_col or old_row != new_row:
            if agent in self.grid[old_row][old_col]:
                self.grid[old_row][old_col].remove(agent)
            self.grid[new_row][new_col].append(agent)
    
    def query_radius(self, position: np.ndarray, radius: float) -> List[Any]:
        """
        Find all agents within a radius of the given position.
        
        Args:
            position (np.ndarray): Center position.
            radius (float): Search radius.
            
        Returns:
            List[Any]: Agents within the radius.
        """
        nearby = []
        
        # Calculate cell range to check
        cell_radius = int(radius // self.cell_size) + 1
        center_col, center_row = self.get_cell_indices(position)
        
        # Check cells within the radius
        for r in range(max(0, center_row - cell_radius), 
                       min(self.rows, center_row + cell_radius + 1)):
            for c in range(max(0, center_col - cell_radius), 
                           min(self.cols, center_col + cell_radius + 1)):
                # Check agents in this cell
                for agent in self.grid[r][c]:
                    # Calculate distance with wraparound
                    dx = min(abs(position[0] - agent.position[0]), 
                             self.width - abs(position[0] - agent.position[0]))
                    dy = min(abs(position[1] - agent.position[1]), 
                             self.height - abs(position[1] - agent.position[1]))
                    
                    if dx*dx + dy*dy <= radius*radius:
                        nearby.append(agent)
        
        return nearby
    
    def clear(self) -> None:
        """
        Clear all agents from the spatial grid.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = []