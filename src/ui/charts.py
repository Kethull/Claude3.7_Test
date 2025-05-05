"""
Charts module for visualizing simulation statistics.

This module provides real-time charts for displaying population and 
other metrics over time using Matplotlib.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive mode
import matplotlib.pyplot as plt
import pygame
from typing import Dict, Any, List, Tuple, Optional
import io
import time


class SimulationCharts:
    """
    Real-time charts for displaying simulation statistics.
    
    This class handles:
    - Population vs. time chart
    - Energy vs. time chart
    - Converting matplotlib plots to Pygame surfaces
    
    Attributes:
        width (int): Width of the chart surface.
        height (int): Height of the chart surface.
        stats_history (List[Dict[str, Any]]): History of simulation statistics.
        max_history (int): Maximum number of timesteps to display.
    """
    
    def __init__(self, width: int = 500, height: int = 350, max_history: int = 500):
        """
        Initialize charts renderer.
        
        Args:
            width (int): Width of the chart surface.
            height (int): Height of the chart surface.
            max_history (int): Maximum number of timesteps to display.
        """
        self.width = width
        self.height = height
        self.max_history = max_history
        
        # Style configuration
        plt.style.use('dark_background')
        
        # Create figure with subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(width/100, height/100), dpi=100)
        self.fig.tight_layout(pad=3)
        
        # Configure population chart
        self.ax1.set_title('Population Over Time')
        self.ax1.set_xlabel('Timestep')
        self.ax1.set_ylabel('Count')
        self.ax1.grid(True, alpha=0.3)
        
        # Configure energy chart
        self.ax2.set_title('Average Energy Over Time')
        self.ax2.set_xlabel('Timestep')
        self.ax2.set_ylabel('Energy')
        self.ax2.grid(True, alpha=0.3)
        
        # Initialize empty lines
        self.prey_line, = self.ax1.plot([], [], 'g-', label='Prey')
        self.predator_line, = self.ax1.plot([], [], 'r-', label='Predators')
        self.prey_energy_line, = self.ax2.plot([], [], 'g-', label='Prey Energy')
        self.predator_energy_line, = self.ax2.plot([], [], 'r-', label='Predator Energy')
        
        # Add legends
        self.ax1.legend(loc='upper right')
        self.ax2.legend(loc='upper right')
        
        # Initial surface
        self.surface = pygame.Surface((width, height))
        
        # Chart update tracking
        self.last_update_time = 0
        self.update_interval = 0.5  # Update every 0.5 seconds max
        
    def update(self, stats_history: List[Dict[str, Any]]) -> None:
        """
        Update charts with new statistics.
        
        Args:
            stats_history (List[Dict[str, Any]]): History of simulation statistics.
        """
        # If no stats, return empty surface
        if not stats_history:
            return
        
        # Rate-limit updates to avoid excessive CPU usage
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Limit history to max length
        if len(stats_history) > self.max_history:
            stats_history = stats_history[-self.max_history:]
        
        # Extract data for charts
        timesteps = [stat.get('timestep', i) for i, stat in enumerate(stats_history)]
        prey_counts = [stat.get('prey_count', 0) for stat in stats_history]
        predator_counts = [stat.get('predator_count', 0) for stat in stats_history]
        prey_energy = [stat.get('avg_prey_energy', 0) for stat in stats_history]
        predator_energy = [stat.get('avg_predator_energy', 0) for stat in stats_history]
        
        # Update population chart
        self.prey_line.set_data(timesteps, prey_counts)
        self.predator_line.set_data(timesteps, predator_counts)
        
        # Update energy chart
        self.prey_energy_line.set_data(timesteps, prey_energy)
        self.predator_energy_line.set_data(timesteps, predator_energy)
        
        # Adjust y-axis limits with some padding
        pop_max = max(max(prey_counts, default=10), max(predator_counts, default=10)) * 1.1
        self.ax1.set_ylim(0, max(10, pop_max))
        
        energy_max = max(max(prey_energy, default=100), max(predator_energy, default=100)) * 1.1
        self.ax2.set_ylim(0, max(100, energy_max))
        
        # Auto-adjust x-axis
        if len(timesteps) > 1:
            self.ax1.set_xlim(min(timesteps), max(timesteps))
            self.ax2.set_xlim(min(timesteps), max(timesteps))
        else:
            # If only one timestep, create a small range
            self.ax1.set_xlim(min(timesteps) - 1, max(timesteps) + 1)
            self.ax2.set_xlim(min(timesteps) - 1, max(timesteps) + 1)
        
        # Redraw figure
        self.fig.canvas.draw()
        
        # Convert to pygame surface
        self._update_surface()
    
    def _update_surface(self) -> None:
        """
        Convert matplotlib figure to pygame surface.
        """
        # Save figure to in-memory buffer
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Load buffer as pygame image
        chart_img = pygame.image.load(buf, 'png')
        
        # Create new surface
        self.surface = pygame.Surface((self.width, self.height))
        
        # Blit image to surface
        self.surface.blit(chart_img, (0, 0))
        
        # Clean up
        buf.close()
    
    def get_surface(self) -> pygame.Surface:
        """
        Get the current charts as a pygame surface.
        
        Returns:
            pygame.Surface: Surface containing the charts.
        """
        return self.surface
    
    def save_charts(self, filename: str) -> None:
        """
        Save current charts to an image file.
        
        Args:
            filename (str): Path to save the file.
        """
        self.fig.savefig(filename, dpi=100)
    
    def display_window(self, stats_history: List[Dict[str, Any]]) -> None:
        """
        Display charts in a separate matplotlib window (for debugging).
        
        Args:
            stats_history (List[Dict[str, Any]]): History of simulation statistics.
        """
        # Use a clean figure
        plt.figure(figsize=(10, 8))
        
        # Plot data
        plt.subplot(2, 1, 1)
        timesteps = [stat.get('timestep', i) for i, stat in enumerate(stats_history)]
        prey_counts = [stat.get('prey_count', 0) for stat in stats_history]
        predator_counts = [stat.get('predator_count', 0) for stat in stats_history]
        
        plt.plot(timesteps, prey_counts, 'g-', label='Prey')
        plt.plot(timesteps, predator_counts, 'r-', label='Predators')
        plt.grid(True)
        plt.title('Population Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Count')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        prey_energy = [stat.get('avg_prey_energy', 0) for stat in stats_history]
        predator_energy = [stat.get('avg_predator_energy', 0) for stat in stats_history]
        
        plt.plot(timesteps, prey_energy, 'g-', label='Prey Energy')
        plt.plot(timesteps, predator_energy, 'r-', label='Predator Energy')
        plt.grid(True)
        plt.title('Average Energy Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Energy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()