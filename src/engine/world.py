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
    
    def __init__(self, width: int, height: int):
        """
        Initialize the world with given dimensions.
        
        Args:
            width (int): Width of the world.
            height (int): Height of the world.
        """
        self.width = width
        self.height = height
        self.agents: List[Agent] = []
        self.timestep = 0
        self.spatial_index = None  # Will be initialized if needed
    
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the world.
        
        Args:
            agent (Agent): The agent to add.
        """
        self.agents.append(agent)
        if self.spatial_index is not None:
            self.spatial_index.insert(agent)
    
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
            "vision": self._get_vision_rays(agent),
            "timestamp": self.timestep
        }
        
        return observation
    
    def _get_vision_rays(self, agent: Agent, num_rays: int = 8, 
                       vision_range: float = 100.0) -> List[Dict[str, Any]]:
        """
        Generate vision rays for an agent.
        
        Args:
            agent (Agent): The agent to generate vision rays for.
            num_rays (int): Number of rays to cast.
            vision_range (float): Maximum distance of vision.
            
        Returns:
            List[Dict[str, Any]]: List of vision ray results.
        """
        rays = []
        
        # Cast rays in different directions
        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Find the closest agent in this direction
            closest_agent = None
            closest_distance = vision_range
            
            for other in self.agents:
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
    
    def step(self) -> None:
        """
        Advance the simulation by one timestep.
        
        This function:
        1. Gets observations for each agent
        2. Has agents decide on actions
        3. Updates agent positions and energy
        4. Handles interactions (eating, reproduction)
        5. Removes dead agents
        """
        self.timestep += 1
        
        # For tracking new agents from reproduction
        new_agents = []
        
        # Process each agent
        for agent in list(self.agents):  # Make a copy for safe iteration
            if not agent.alive:
                self.remove_agent(agent)
                continue
            
            # Get observation and decide action
            observation = self.get_observation(agent)
            action = agent.act(observation)
            
            # Save old position for spatial index update
            old_position = agent.position.copy()
            
            # Update position based on action
            self._move_agent(agent, action)
            
            # Update spatial index if needed
            if self.spatial_index is not None:
                self.spatial_index.update(agent, old_position)
            
            # Handle interactions with other agents
            self._handle_interactions(agent)
            
            # Handle reproduction
            offspring = agent.try_reproduce()
            if offspring:
                offspring.position = self.wrap_position(offspring.position)
                new_agents.append(offspring)
        
        # Add new agents from reproduction
        for offspring in new_agents:
            self.add_agent(offspring)
        
        # Remove dead agents again to catch any that died during interactions
        self.agents = [agent for agent in self.agents if agent.alive]
    
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
        
        # Insert all existing agents into the spatial index
        for agent in self.agents:
            spatial_index.insert(agent)
    
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