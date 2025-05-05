"""
Storage utilities for saving and loading simulation states.

This module provides functionality for saving and loading simulation states,
agent populations, and neural network policies using HDF5 format.
"""
import os
import h5py
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Optional, Tuple


class SimulationStorage:
    """
    Storage for saving and loading simulation data.
    
    This class handles:
    - Saving world states (agent positions, energy levels, types)
    - Saving neural network policy weights
    - Saving simulation statistics
    - Loading saved states for replay
    
    Attributes:
        filename (str): Path to the HDF5 file.
        file (h5py.File): Open HDF5 file handle.
    """
    
    def __init__(self, filename: str):
        """
        Initialize the storage.
        
        Args:
            filename (str): Path to the HDF5 file.
        """
        self.filename = filename
        self.file = None
    
    def open(self, mode: str = 'r') -> None:
        """
        Open the HDF5 file.
        
        Args:
            mode (str): File mode ('r' for read, 'w' for write, 'a' for append).
        """
        if self.file is not None:
            self.close()
        
        self.file = h5py.File(self.filename, mode)
    
    def close(self) -> None:
        """
        Close the HDF5 file.
        """
        if self.file is not None:
            self.file.close()
            self.file = None
    
    def is_open(self) -> bool:
        """
        Check if the file is open.
        
        Returns:
            bool: True if file is open, False otherwise.
        """
        return self.file is not None
    
    def save_world_state(self, world, timestep: int) -> None:
        """
        Save the state of the world at a specific timestep.
        
        Args:
            world: The world object to save.
            timestep (int): Current simulation timestep.
        """
        if not self.is_open():
            raise ValueError("File not open. Call open() first.")
        
        # Create group for this timestep
        timestep_group = self.file.require_group(f"world/timestep_{timestep}")
        
        # Collect agent data
        positions = np.array([agent.position for agent in world.agents])
        energies = np.array([agent.energy for agent in world.agents])
        types = np.array([agent.type for agent in world.agents])
        
        # Save data
        if len(world.agents) > 0:
            # Save or replace datasets
            if "positions" in timestep_group:
                del timestep_group["positions"]
            if "energies" in timestep_group:
                del timestep_group["energies"]
            if "types" in timestep_group:
                del timestep_group["types"]
            
            timestep_group.create_dataset("positions", data=positions)
            timestep_group.create_dataset("energies", data=energies)
            timestep_group.create_dataset("types", data=types, dtype=h5py.special_dtype(vlen=str))
        else:
            # Create empty datasets if no agents
            timestep_group.create_dataset("positions", shape=(0, 2))
            timestep_group.create_dataset("energies", shape=(0,))
            timestep_group.create_dataset("types", shape=(0,), dtype=h5py.special_dtype(vlen=str))
        
        # Save world metadata
        timestep_group.attrs["world_width"] = world.width
        timestep_group.attrs["world_height"] = world.height
        timestep_group.attrs["agent_count"] = len(world.agents)
    
    def save_policy_weights(self, policy, agent_type: str, timestep: int) -> None:
        """
        Save neural network policy weights.
        
        Args:
            policy: PyTorch policy network.
            agent_type (str): Type of agent (prey, predator).
            timestep (int): Current simulation timestep.
        """
        if not self.is_open():
            raise ValueError("File not open. Call open() first.")
        
        # Create group for policies
        policy_group = self.file.require_group(f"policies/{agent_type}_{timestep}")
        
        # Save each parameter tensor
        for name, param in policy.state_dict().items():
            # Convert parameter to numpy array
            param_np = param.detach().cpu().numpy()
            
            # Save or replace dataset
            if name in policy_group:
                del policy_group[name]
            
            policy_group.create_dataset(name, data=param_np)
    
    def save_stats(self, stats: Dict[str, Any], timestep: int) -> None:
        """
        Save simulation statistics for a timestep.
        
        Args:
            stats (Dict[str, Any]): Dictionary of statistics.
            timestep (int): Current simulation timestep.
        """
        if not self.is_open():
            raise ValueError("File not open. Call open() first.")
        
        # Create group for stats
        stats_group = self.file.require_group("stats")
        
        # Create dataset for this timestep if it doesn't exist
        if str(timestep) not in stats_group:
            stats_group.create_dataset(str(timestep), data=np.array([0]))
        
        # Save each statistic as an attribute
        for key, value in stats.items():
            stats_group[str(timestep)].attrs[key] = value
    
    def load_world_state(self, world, timestep: int) -> bool:
        """
        Load world state from storage.
        
        Args:
            world: World object to populate.
            timestep (int): Timestep to load.
            
        Returns:
            bool: True if successful, False if timestep not found.
        """
        if not self.is_open():
            raise ValueError("File not open. Call open() first.")
        
        # Check if this timestep exists
        timestep_path = f"world/timestep_{timestep}"
        if timestep_path not in self.file:
            return False
        
        # Get timestep group
        timestep_group = self.file[timestep_path]
        
        # Clear current agents
        world.agents = []
        
        # Load positions, energies, and types
        positions = timestep_group["positions"][:]
        energies = timestep_group["energies"][:]
        types = timestep_group["types"][:]
        
        # Set world dimensions if available
        if "world_width" in timestep_group.attrs:
            world.width = timestep_group.attrs["world_width"]
        if "world_height" in timestep_group.attrs:
            world.height = timestep_group.attrs["world_height"]
        
        # Create and add agents
        from agents.prey import Prey
        from agents.predator import Predator
        
        for i in range(len(positions)):
            if types[i] == "prey":
                agent = Prey(positions[i], energies[i])
            elif types[i] == "predator":
                agent = Predator(positions[i], energies[i])
            else:
                # Generic agent fallback
                from agents.base import Agent
                agent = Agent(positions[i], energies[i])
                agent.type = types[i]
            
            world.add_agent(agent)
        
        return True
    
    def load_policy_weights(self, policy, agent_type: str, timestep: int) -> bool:
        """
        Load policy weights from storage.
        
        Args:
            policy: PyTorch policy network to load weights into.
            agent_type (str): Type of agent (prey, predator).
            timestep (int): Timestep to load.
            
        Returns:
            bool: True if successful, False if policy not found.
        """
        if not self.is_open():
            raise ValueError("File not open. Call open() first.")
        
        # Check if this policy exists
        policy_path = f"policies/{agent_type}_{timestep}"
        if policy_path not in self.file:
            return False
        
        # Get policy group
        policy_group = self.file[policy_path]
        
        # Create state dict to load
        state_dict = {}
        
        # Load each parameter
        for name in policy_group:
            # Convert to torch tensor
            param = torch.tensor(policy_group[name][:])
            state_dict[name] = param
        
        # Load state dict into policy
        policy.load_state_dict(state_dict)
        
        return True
    
    def get_stats(self, timestep: int) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific timestep.
        
        Args:
            timestep (int): Timestep to get stats for.
            
        Returns:
            Optional[Dict[str, Any]]: Statistics dictionary, or None if not found.
        """
        if not self.is_open():
            raise ValueError("File not open. Call open() first.")
        
        # Check if stats exist for this timestep
        stats_path = f"stats/{timestep}"
        if stats_path not in self.file:
            return None
        
        # Get stats dataset
        stats_dataset = self.file[stats_path]
        
        # Convert attributes to dictionary
        stats = dict(stats_dataset.attrs)
        
        return stats
    
    def get_timesteps(self) -> List[int]:
        """
        Get all timesteps that have been saved.
        
        Returns:
            List[int]: List of timesteps.
        """
        if not self.is_open():
            raise ValueError("File not open. Call open() first.")
        
        if "world" not in self.file:
            return []
        
        # Extract timestep numbers from group names
        timesteps = []
        for name in self.file["world"]:
            if name.startswith("timestep_"):
                try:
                    timestep = int(name.split("_")[1])
                    timesteps.append(timestep)
                except (ValueError, IndexError):
                    continue
        
        return sorted(timesteps)
    
    def export_stats_to_csv(self, csv_filename: str) -> None:
        """
        Export simulation statistics to a CSV file.
        
        Args:
            csv_filename (str): Path to CSV file.
        """
        # Open file if needed
        if not self.is_open():
            self.open('r')
        
        try:
            # Check if stats exist
            if "stats" not in self.file:
                raise ValueError("No statistics found in HDF5 file.")
            
            stats_group = self.file["stats"]
            
            # Collect all stats
            all_stats = []
            for timestep in stats_group:
                try:
                    timestep_int = int(timestep)
                    stats = dict(stats_group[timestep].attrs)
                    stats["timestep"] = timestep_int
                    all_stats.append(stats)
                except ValueError:
                    continue  # Skip non-integer timesteps
            
            # Convert to DataFrame and sort by timestep
            df = pd.DataFrame(all_stats)
            if len(df) > 0:
                df = df.sort_values("timestep")
                
                # Save to CSV
                df.to_csv(csv_filename, index=False)
            else:
                print("No statistics found to export.")
        
        finally:
            # Close if we opened the file
            if not self.is_open():
                self.close()