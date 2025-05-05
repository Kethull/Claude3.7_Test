"""
World module for predator-prey simulation.

This module defines the World class that manages the simulation environment,
including agents, physics, and interactions.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from agents.base import Agent


class World:
    """
    Represents the simulation world where agents interact.
    
    The world maintains a list of all agents, handles their movement and interactions,
    and enforces world boundaries through wraparound.
    
    Attributes:
        width (int): Width of the world.
        height (int): Height of the world.
        agents (List[Agent]): List of all agents in the world.
        timestep (int): Current simulation timestep.
        spatial_index: Spatial partitioning structure (optional).
    """
    
    def set_separation_params(self, radius: float = 10.0, strength: float = 0.5, passes: int = 3) -> None:
        """
        Set parameters for agent separation behavior.
        
        Args:
            radius (float): Distance within which separation forces apply.
            strength (float): Strength of the separation force (0.0-1.0 recommended).
            passes (int): Number of separation passes per step (more passes = stronger separation).
        """
        self.separation_radius = radius
        self.separation_strength = strength
        self.separation_passes = passes
        
        print(f"Set separation parameters: radius={radius}, strength={strength}, passes={passes}")
    
    def __init__(self, width: int, height: int):
        """
        Initialize the world with given dimensions.
        
        Args:
            width (int): Width of the world.
            height (int): Height of the world.
        """
        self.width = width
        self.height = height
        self.agents = []
        self.timestep = 0
        self.spatial_index = None  # Will be initialized if needed
        
        # Separation parameters (default values)
        self.separation_radius = 10.0
        self.separation_strength = 0.5
        self.separation_passes = 3
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the world.
        
        Args:
            agent (Agent): The agent to add.
        """
        self.agents.append(agent)
        if self.spatial_index is not None:
            try:
                self.spatial_index.insert(agent)
            except Exception as e:
                print(f"ERROR inserting agent into spatial index: {e}")
    
    def remove_agent(self, agent: Agent) -> None:
        """
        Remove an agent from the world.
        
        Args:
            agent (Agent): The agent to remove.
        """
        if agent in self.agents:
            self.agents.remove(agent)
            if self.spatial_index is not None:
                self.spatial_index.remove(agent)
    
    def wrap_position(self, position: np.ndarray) -> np.ndarray:
        """
        Wrap a position around the world boundaries.
        
        Args:
            position (np.ndarray): The position to wrap.
            
        Returns:
            np.ndarray: The wrapped position.
        """
        wrapped = position.copy()
        wrapped[0] = wrapped[0] % self.width
        wrapped[1] = wrapped[1] % self.height
        return wrapped
    
    def get_observation(self, agent: Agent) -> Dict[str, Any]:
        """
        Generate observation for a given agent.
        
        This includes vision rays, agent's energy, and any other
        relevant information for decision making.
        
        Args:
            agent (Agent): The agent to generate observation for.
            
        Returns:
            Dict[str, Any]: Observation dictionary.
        """
        # Basic observation including agent's energy
        observation = {
            "energy": agent.energy,
            "position": agent.position.copy(),
            "vision": self._get_vision_rays_optimized(agent, 4, 75.0),
            "timestamp": self.timestep
        }
        
        return observation
    
    # Optimized vision ray calculation
    def _get_vision_rays_optimized(self, agent: Agent, num_rays: int = 4, 
                        vision_range: float = 100.0) -> List[Dict[str, Any]]:
        """Optimized vision ray calculation using spatial partitioning."""
        rays = []
        
        # Use fewer rays (reduce from 8 to 4)
        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Find the closest agent in this direction using spatial index
            closest_agent = None
            closest_distance = vision_range
            
            # Compute ray endpoint
            ray_end = agent.position + direction * vision_range
            
            # Get candidates from spatial index (much faster than checking all agents)
            if self.spatial_index is not None:
                # Create a search box along the ray
                search_width = 10.0  # Width of search box
                perpendicular = np.array([-direction[1], direction[0]]) * search_width/2
                
                # Check cells that intersect with this ray path
                cells_to_check = self._get_cells_along_ray(agent.position, ray_end, search_width)
                candidates = []
                for cell in cells_to_check:
                    row, col = cell
                    if 0 <= row < self.spatial_index.rows and 0 <= col < self.spatial_index.cols:
                        candidates.extend(self.spatial_index.grid[row][col])
            else:
                candidates = self.agents
            
            # Process only agent candidates from spatial index
            for other in candidates:
                if other is agent:
                    continue
                
                # Calculate vector to other agent
                to_other = other.position - agent.position
                
                # Handle wraparound for distance calculation
                to_other[0] = (to_other[0] + self.width / 2) % self.width - self.width / 2
                to_other[1] = (to_other[1] + self.height / 2) % self.height - self.height / 2
                
                # Project onto ray direction
                projection = np.dot(to_other, direction)
                
                if projection <= 0:  # Behind the agent
                    continue
                
                # Calculate perpendicular distance to ray
                perp_dist = np.linalg.norm(to_other - projection * direction)
                
                # If within the ray's width and closer than current closest
                if perp_dist < 5.0 and projection < closest_distance:
                    closest_agent = other
                    closest_distance = projection
            
            # Record the ray result
            ray_result = {
                "distance": closest_distance if closest_agent else vision_range,
                "type": closest_agent.type if closest_agent else "none"
            }
            rays.append(ray_result)
    
        return rays

    def _get_cells_along_ray(self, start, end, width):
        """Get the grid cells that a ray passes through."""
        cells = []
        direction = end - start
        length = np.linalg.norm(direction)
        if length == 0:
            return []
        
        direction = direction / length
        perpendicular = np.array([-direction[1], direction[0]])
        
        # Sample points along the ray
        num_samples = max(10, int(length / (self.spatial_index.cell_size/2)))
        for i in range(num_samples):
            t = i / num_samples
            center = start + direction * length * t
            
            # Get the cell for this point
            col, row = self.spatial_index.get_cell_indices(center)
            cells.append((row, col))
        
        return list(set(cells))  # Remove duplicates
    
    """
    Modified step_optimized method to include separation logic.
    This should replace the existing step_optimized method in the World class in world.py
    """
    def step_optimized(self) -> None:
        """
        Optimized version of the world step function with separation logic.
        
        This batches operations and improves performance:
        1. Gather all actions first
        2. Apply movements in batch
        3. Apply separation forces to prevent overlapping
        4. Handle interactions in batch
        5. Handle reproduction at the end
        """
        self.timestep += 1
        
        # For tracking new agents from reproduction
        new_agents = []
        
        # STEP 1: Gather all observations and actions first
        agent_actions = {}
        old_positions = {}
        
        for agent in self.agents:
            if not agent.alive:
                continue
                
            # Get observation and decide action
            observation = self.get_observation(agent)
            action = agent.act(observation)
            
            # Save for batch processing
            agent_actions[agent] = (action, observation)
            old_positions[agent] = agent.position.copy()
        
        # STEP 2: Process all movements first
        for agent, (action, observation) in agent_actions.items():
            self._move_agent(agent, action)
        
        # STEP 3: Update spatial index with all new positions at once
        if self.spatial_index is not None:
            # Completely rebuild spatial index (faster than many individual updates)
            self.spatial_index.clear()
            for agent in self.agents:
                if agent.alive:
                    self.spatial_index.insert(agent)
        
        # STEP 4: Apply separation forces to prevent overlapping
        # Multiple passes for better results
        for _ in range(self.separation_passes):
            for agent in self.agents:
                if not agent.alive:
                    continue
                
                # Apply separation forces
                self.apply_separation(
                    agent, 
                    separation_radius=self.separation_radius,
                    separation_strength=self.separation_strength
                )
        
        # STEP 5: Process all predator-prey interactions
        predator_agents = [a for a in self.agents if a.alive and a.type == "predator"]
        prey_agents = {a: True for a in self.agents if a.alive and a.type == "prey"}
        
        for predator in predator_agents:
            if not predator.alive:  # Skip if killed by another predator
                continue
                
            # Find nearby agents more efficiently with spatial index
            nearby_agents = self._get_nearby_agents(predator, 5.0)
            
            # Find prey to eat
            for prey in nearby_agents:
                if prey.type == "prey" and prey in prey_agents and prey_agents[prey]:
                    # Predator eats prey
                    energy_gain = min(prey.energy, 50)  # Cap energy gain
                    predator.energy += energy_gain
                    prey.alive = False
                    prey_agents[prey] = False  # Mark as eaten
                    break  # Only eat one prey per step
        
        # STEP 6: Process reproduction and agent updates
        for agent, (action, observation) in agent_actions.items():
            if not agent.alive:
                continue
                
            # Add last action to observation for energy updates
            observation["last_action"] = action
                
            # Update agent internal state
            agent.update(observation)
            
            # Try reproduction
            offspring = agent.try_reproduce()
            if offspring:
                offspring.position = self.wrap_position(offspring.position)
                new_agents.append(offspring)
        
        # STEP 7: Remove eaten prey
        self.agents = [agent for agent in self.agents if agent.alive]
        
        # STEP 8: Add new agents from reproduction
        for offspring in new_agents:
            self.add_agent(offspring)
    
    def _move_agent(self, agent: Agent, action: int) -> None:
        """
        Move an agent based on its action.
        
        Args:
            agent (Agent): The agent to move.
            action (int): The action index.
        """
        # Define movement vectors for discrete actions
        # 0: Stay, 1: Up, 2: Down, 3: Right, 4: Left
        moves = {
            0: np.array([0.0, 0.0]),    # Stay
            1: np.array([0.0, -1.0]),   # Up (negative y)
            2: np.array([0.0, 1.0]),    # Down (positive y)
            3: np.array([1.0, 0.0]),    # Right (positive x)
            4: np.array([-1.0, 0.0])    # Left (negative x)
        }
        
        # Get movement vector
        move_vector = moves.get(action, np.array([0.0, 0.0]))
        
        # Apply movement with speed factor
        speed = 2.0  # Units per timestep
        agent.position += move_vector * speed
        
        # Apply energy cost if moved
        if action != 0:  # Not staying still
            agent.apply_energy_cost(1)
        
        # Wrap position around world boundaries
        agent.position = self.wrap_position(agent.position)
    
    def _handle_interactions(self, agent: Agent) -> None:
        """
        Handle interactions between agents, such as predators eating prey.
        
        Args:
            agent (Agent): The agent to process interactions for.
        """
        # Skip if the agent isn't a predator
        if agent.type != "predator":
            return
        
        # Find nearby agents
        nearby_agents = self._get_nearby_agents(agent, 5.0)
        
        # Check for prey to eat
        for other in nearby_agents:
            if other.type == "prey" and other.alive:
                # Predator eats prey
                energy_gain = min(other.energy, 50)  # Cap energy gain
                agent.energy += energy_gain
                other.alive = False
                self.remove_agent(other)
                break  # Only eat one prey per step
    
    def _get_nearby_agents(self, agent: Agent, radius: float) -> List[Agent]:
        """
        Get agents within a certain radius of the given agent.
        
        Args:
            agent (Agent): The central agent.
            radius (float): The search radius.
            
        Returns:
            List[Agent]: List of nearby agents.
        """
        if self.spatial_index is not None:
            return self.spatial_index.query_radius(agent.position, radius)
        
        # Fallback to brute force if no spatial index
        nearby = []
        for other in self.agents:
            if other is agent:
                continue
                
            # Calculate distance with wraparound
            dx = min(abs(agent.position[0] - other.position[0]), 
                     self.width - abs(agent.position[0] - other.position[0]))
            dy = min(abs(agent.position[1] - other.position[1]), 
                     self.height - abs(agent.position[1] - other.position[1]))
            
            if dx*dx + dy*dy <= radius*radius:
                nearby.append(other)
                
        return nearby
        
    def set_spatial_index(self, spatial_index) -> None:
        """
        Set a spatial partitioning structure for efficient neighbor queries.
        
        Args:
            spatial_index: The spatial index structure to use.
        """
        self.spatial_index = spatial_index
        print(f"Setting spatial index: {type(spatial_index).__name__}")
        
        # Insert all existing agents into the spatial index
        for agent in self.agents:
            try:
                spatial_index.insert(agent)
            except Exception as e:
                print(f"ERROR inserting agent into spatial index during setup: {e}")
        
        print(f"Inserted {len(self.agents)} agents into spatial index")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current world state.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics.
        """
        stats = {
            "timestep": self.timestep,
            "total_agents": len(self.agents),
            "prey_count": sum(1 for a in self.agents if a.type == "prey"),
            "predator_count": sum(1 for a in self.agents if a.type == "predator"),
            "avg_prey_energy": np.mean([a.energy for a in self.agents if a.type == "prey"]) if any(a.type == "prey" for a in self.agents) else 0,
            "avg_predator_energy": np.mean([a.energy for a in self.agents if a.type == "predator"]) if any(a.type == "predator" for a in self.agents) else 0,
        }
        return stats
    
    """
    Add separation logic to the World class.
    This should be added to the World class in world.py
    """
    """
    Updated apply_separation method to properly utilize the separation parameters.
    This should replace the apply_separation method we created earlier.
    """
    def apply_separation(self, agent, separation_radius=None, separation_strength=None):
        """
        Apply separation forces to keep agents from overlapping.
        
        Args:
            agent (Agent): Agent to apply separation to.
            separation_radius (float, optional): Radius within which separation forces apply.
            separation_strength (float, optional): Strength of the separation force.
            
        Returns:
            np.ndarray: New position after applying separation.
        """
        # Use instance parameters if not specified
        if separation_radius is None:
            separation_radius = self.separation_radius
        
        if separation_strength is None:
            separation_strength = self.separation_strength
        
        # Get nearby agents using spatial index if available
        nearby_agents = self._get_nearby_agents(agent, separation_radius)
        
        # Calculate separation force
        separation_force = agent.calculate_separation_force(
            nearby_agents, self, separation_radius, separation_strength
        )
        
        # Apply the force to the agent's position
        if np.any(separation_force):
            # Store old position for spatial index update
            old_position = agent.position.copy()
            
            # Apply separation force
            agent.position += separation_force
            
            # Wrap around world boundaries
            agent.position = self.wrap_position(agent.position)
            
            # Update spatial index if needed
            if self.spatial_index is not None:
                self.spatial_index.update(agent, old_position)
                
        return agent.position