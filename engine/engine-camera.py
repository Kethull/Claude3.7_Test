"""
Camera module for controlling the viewport.

This module provides camera controls for panning and zooming the view of
the simulation world.
"""
import numpy as np
from typing import Tuple, Optional, Any


class Camera:
    """
    Camera for controlling the viewport into the simulation world.
    
    The camera handles:
    - Panning (moving the view)
    - Zooming (changing the scale)
    - Following a selected agent
    
    Attributes:
        world_width (int): Width of the simulation world.
        world_height (int): Height of the simulation world.
        position (np.ndarray): Center position of the camera view.
        zoom (float): Zoom level (scale factor).
        target_agent: Optional agent to follow.
    """
    
    def __init__(self, world_width: int, world_height: int):
        """
        Initialize the camera.
        
        Args:
            world_width (int): Width of the simulation world.
            world_height (int): Height of the simulation world.
        """
        self.world_width = world_width
        self.world_height = world_height
        self.position = np.array([world_width / 2, world_height / 2], dtype=float)
        self.zoom = 1.0
        self.target_agent = None
        
        # Zoom limits
        self.min_zoom = 0.1
        self.max_zoom = 5.0
    
    def pan(self, dx: float, dy: float) -> None:
        """
        Pan the camera by the given amounts.
        
        Args:
            dx (float): Amount to pan horizontally.
            dy (float): Amount to pan vertically.
        """
        self.position[0] += dx / self.zoom
        self.position[1] += dy / self.zoom
        
        # Constrain to world boundaries, allowing some margin
        margin = 100
        self.position[0] = np.clip(self.position[0], -margin, self.world_width + margin)
        self.position[1] = np.clip(self.position[1], -margin, self.world_height + margin)
    
    def zoom_in(self, factor: float = 1.1) -> None:
        """
        Zoom in by the given factor.
        
        Args:
            factor (float): Factor to zoom in by.
        """
        self.zoom = min(self.zoom * factor, self.max_zoom)
    
    def zoom_out(self, factor: float = 1.1) -> None:
        """
        Zoom out by the given factor.
        
        Args:
            factor (float): Factor to zoom out by.
        """
        self.zoom = max(self.zoom / factor, self.min_zoom)
    
    def reset(self) -> None:
        """
        Reset the camera to the center of the world with default zoom.
        """
        self.position = np.array([self.world_width / 2, self.world_height / 2], dtype=float)
        self.zoom = 1.0
        self.target_agent = None
    
    def follow(self, agent) -> None:
        """
        Set the camera to follow an agent.
        
        Args:
            agent: The agent to follow.
        """
        self.target_agent = agent
    
    def stop_following(self) -> None:
        """
        Stop following the current target agent.
        """
        self.target_agent = None
    
    def update(self) -> None:
        """
        Update the camera position if following an agent.
        """
        if self.target_agent and self.target_agent.alive:
            # Smoothly move camera towards target
            target_pos = self.target_agent.position.copy()
            self.position = self.position * 0.9 + target_pos * 0.1
    
    def world_to_screen(self, world_pos: np.ndarray, screen_width: int, 
                       screen_height: int) -> np.ndarray:
        """
        Convert world coordinates to screen coordinates.
        
        Args:
            world_pos (np.ndarray): Position in world coordinates.
            screen_width (int): Width of the screen.
            screen_height (int): Height of the screen.
            
        Returns:
            np.ndarray: Position in screen coordinates.
        """
        # Calculate screen center
        screen_center = np.array([screen_width / 2, screen_height / 2])
        
        # Calculate offset from camera position
        offset = world_pos - self.position
        
        # Handle wraparound for smooth scrolling
        if offset[0] > self.world_width / 2:
            offset[0] -= self.world_width
        elif offset[0] < -self.world_width / 2:
            offset[0] += self.world_width
            
        if offset[1] > self.world_height / 2:
            offset[1] -= self.world_height
        elif offset[1] < -self.world_height / 2:
            offset[1] += self.world_height
        
        # Apply zoom and add to screen center
        screen_pos = screen_center + offset * self.zoom
        return screen_pos
    
    def screen_to_world(self, screen_pos: np.ndarray, screen_width: int, 
                       screen_height: int) -> np.ndarray:
        """
        Convert screen coordinates to world coordinates.
        
        Args:
            screen_pos (np.ndarray): Position in screen coordinates.
            screen_width (int): Width of the screen.
            screen_height (int): Height of the screen.
            
        Returns:
            np.ndarray: Position in world coordinates.
        """
        # Calculate screen center
        screen_center = np.array([screen_width / 2, screen_height / 2])
        
        # Calculate offset from screen center
        offset = (screen_pos - screen_center) / self.zoom
        
        # Add offset to camera position
        world_pos = self.position + offset
        
        # Wrap around world boundaries
        world_pos[0] = world_pos[0] % self.world_width
        world_pos[1] = world_pos[1] % self.world_height
        
        return world_pos