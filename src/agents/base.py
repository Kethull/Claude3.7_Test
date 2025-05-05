"""
Base agent module for the predator-prey simulation.

This module defines the common Agent interface from which specific agent types
(prey, predator) inherit.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
import uuid


class Agent:
    """
    Base agent class for all entities in the simulation.
    
    The agent class handles:
    - Position and movement
    - Energy management
    - Interaction with the environment
    
    Attributes:
        position (np.ndarray): Current position in the world.
        energy (float): Current energy level.
        alive (bool): Whether the agent is alive.
        type (str): Type of agent (to be defined by subclasses).
    """
    
    def __init__(self, position: np.ndarray, energy: float = 100.0):
        """
        Initialize an agent.
        
        Args:
            position (np.ndarray): Initial position.
            energy (float): Initial energy level.
        """
        self.position = position.copy()
        self.energy = energy
        self.alive = True
        self.type = "generic"  # Overridden by subclasses
        self.id = str(uuid.uuid4())  # Unique identifier
        self.age = 0  # Age in timesteps
        
        # Configuration for reproduction
        self.reproduction_threshold = 150.0
        self.reproduction_cost = 50.0
        self.initial_energy = 100.0
        self.reproduction_cooldown = 10
        self.last_reproduction = 0
        
        # RL-related attributes
        self.policy = None
        self.last_observation = None
        self.last_action = None
        self.last_reward = 0.0
        self.cumulative_reward = 0.0
    
    def apply_energy_cost(self, cost: float) -> None:
        """
        Apply an energy cost to the agent.
        
        Args:
            cost (float): Amount of energy to deduct.
        """
        self.energy = max(0.0, self.energy - cost)
        
        # Check if agent dies from energy depletion
        if self.energy <= 0:
            self.alive = False
    
    def add_energy(self, amount: float) -> None:
        """
        Add energy to the agent.
        
        Args:
            amount (float): Amount of energy to add.
        """
        self.energy += amount
    
    def act(self, observation: Dict[str, Any]) -> int:
        """
        Decide on an action based on the current observation.
        
        This is an abstract method to be implemented by subclasses.
        
        Args:
            observation (Dict[str, Any]): Observation of the environment.
            
        Returns:
            int: Action index.
        """
        raise NotImplementedError("Subclasses must implement act()")
    
    def try_reproduce(self) -> Optional['Agent']:
        """
        Attempt to reproduce if conditions are met.
        
        Returns:
            Optional[Agent]: New offspring agent if reproduction occurs, None otherwise.
        """
        # Check if agent has enough energy and cooldown has passed
        if (self.energy >= self.reproduction_threshold and 
                self.age - self.last_reproduction >= self.reproduction_cooldown):
            
            # Create offspring (with some position offset)
            direction = np.random.rand(2) * 2 - 1  # Random direction
            direction = direction / np.linalg.norm(direction)  # Normalize
            offset = direction * 5.0  # 5 units away
            
            # Subclasses should override this with the correct agent type
            offspring = self.__class__(self.position + offset, self.initial_energy)
            
            # Apply reproduction cost
            self.energy -= self.reproduction_cost
            self.last_reproduction = self.age
            
            return offspring
        
        return None
    
    def update(self, observation: Dict[str, Any]) -> None:
        """
        Update agent state after action.
        
        Args:
            observation (Dict[str, Any]): Current observation.
        """
        self.age += 1
    
    def get_color(self) -> Tuple[int, int, int]:
        """
        Get the color for rendering this agent.
        
        Returns:
            Tuple[int, int, int]: RGB color values.
        """
        # Default color (gray)
        return (128, 128, 128)
    
    """
    Add separation logic to the base Agent class.
    This should be added to the Agent class in base.py
    """
    def calculate_separation_force(self, nearby_agents, world, separation_radius=10.0, separation_strength=1.0):
        """
        Calculate separation force to avoid overlapping with other agents.
        
        Args:
            nearby_agents (List[Agent]): List of nearby agents.
            world: The simulation world (for wraparound calculations).
            separation_radius (float): Radius within which separation applies.
            separation_strength (float): Strength of the separation force.
            
        Returns:
            np.ndarray: Separation force vector.
        """
        if not nearby_agents:
            return np.zeros(2)
        
        separation_force = np.zeros(2)
        
        for other in nearby_agents:
            if other is self:
                continue
            
            # Calculate displacement vector with wraparound
            dx = self.position[0] - other.position[0]
            dy = self.position[1] - other.position[1]
            
            # Handle wraparound for distance calculation
            if abs(dx) > world.width / 2:
                dx = dx - np.sign(dx) * world.width
            if abs(dy) > world.height / 2:
                dy = dy - np.sign(dy) * world.height
                
            # Calculate distance
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Apply separation force if too close
            if 0 < distance < separation_radius:
                # Get normalized direction vector away from other agent
                # Strength increases as agents get closer
                repulsion = (separation_radius - distance) / separation_radius
                
                # Normalize the displacement vector
                norm = np.sqrt(dx*dx + dy*dy)
                if norm > 0:
                    separation_force[0] += (dx / norm) * repulsion
                    separation_force[1] += (dy / norm) * repulsion
        
        # Normalize the separation force if it's not zero
        norm = np.linalg.norm(separation_force)
        if norm > 0:
            separation_force = separation_force / norm * separation_strength
            
        return separation_force