import numpy as np
from typing import List, Tuple, Any

class OptimizedSpatialGrid:
    """
    Optimized grid-based spatial partitioning structure.
    """
    
    def __init__(self, width: int, height: int, cell_size: int):
        """Initialize a spatial grid with improved cell size calculation."""
        self.width = width
        self.height = height
        
        # Choose a better cell size based on average density
        # For predator-prey, agents are typically within 5-10 units of each other
        # Cell size should be close to interaction radius for best performance
        self.cell_size = max(10, min(cell_size, 25))  # Force reasonable range
        
        # Calculate grid dimensions
        self.cols = max(10, width // self.cell_size)
        self.rows = max(10, height // self.cell_size)
        self.cell_width = width / self.cols
        self.cell_height = height / self.rows
        
        # Initialize grid cells with empty lists
        # Use a flat array for better cache locality
        self.grid = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Keep a dictionary of agent positions for faster updates
        self.agent_positions = {}
    
    def get_cell_indices(self, position: np.ndarray) -> Tuple[int, int]:
        """Get the grid cell indices for a position with wraparound."""
        # Ensure position is within world bounds (wraparound)
        x = position[0] % self.width
        y = position[1] % self.height
                
        col = min(int(x / self.cell_width), self.cols - 1)
        row = min(int(y / self.cell_height), self.rows - 1)
        return col, row
    
    def insert(self, agent) -> None:
        """Insert an agent into the grid."""
        col, row = self.get_cell_indices(agent.position)
        self.grid[row][col].append(agent)
        
        # Track agent position for faster updates
        self.agent_positions[agent] = (agent.position.copy(), col, row)
    
    def remove(self, agent) -> None:
        """Remove an agent from the grid."""
        if agent in self.agent_positions:
            _, col, row = self.agent_positions[agent]
            if agent in self.grid[row][col]:
                self.grid[row][col].remove(agent)
            del self.agent_positions[agent]
    
    def update(self, agent, old_position: np.ndarray) -> None:
        """Update an agent's position in the spatial grid."""
        # Get new cell
        new_col, new_row = self.get_cell_indices(agent.position)
        
        # Get old cell
        if agent in self.agent_positions:
            _, old_col, old_row = self.agent_positions[agent]
            
            # If cell changed, update grid
            if old_col != new_col or old_row != new_row:
                self.grid[old_row][old_col].remove(agent)
                self.grid[new_row][new_col].append(agent)
                
                # Update tracking
                self.agent_positions[agent] = (agent.position.copy(), new_col, new_row)
            else:
                # Just update the tracked position
                self.agent_positions[agent] = (agent.position.copy(), old_col, old_row)
        else:
            # If agent not in grid yet, insert it
            self.grid[new_row][new_col].append(agent)
            self.agent_positions[agent] = (agent.position.copy(), new_col, new_row)
    
    def query_radius(self, position: np.ndarray, radius: float) -> List[Any]:
        """Find all agents within a radius with optimized cell search."""
        nearby = []
        
        # Calculate cell range to check
        cell_radius = int(radius / min(self.cell_width, self.cell_height)) + 1
        center_col, center_row = self.get_cell_indices(position)
        
        # Check cells within the radius
        for r in range(max(0, center_row - cell_radius), 
                       min(self.rows, center_row + cell_radius + 1)):
            for c in range(max(0, center_col - cell_radius), 
                           min(self.cols, center_col + cell_radius + 1)):
                
                # Quick distance check for cell corners before checking agents
                cell_center_x = (c + 0.5) * self.cell_width
                cell_center_y = (r + 0.5) * self.cell_height
                dx = min(abs(position[0] - cell_center_x), 
                         self.width - abs(position[0] - cell_center_x))
                dy = min(abs(position[1] - cell_center_y), 
                         self.height - abs(position[1] - cell_center_y))
                cell_dist_sq = dx*dx + dy*dy
                
                # Skip cell if outside radius (with buffer for cell size)
                max_cell_half_width = max(self.cell_width, self.cell_height) / 2
                if cell_dist_sq > (radius + max_cell_half_width)**2:
                    continue
                
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
        """Clear all agents from the spatial grid."""
        self.grid = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        self.agent_positions = {}