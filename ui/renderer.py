"""
Pygame renderer for the predator-prey simulation.

This module handles rendering the simulation world, agents, and UI elements.
"""
import pygame
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import pygame.font


class Renderer:
    """
    Renderer for the predator-prey simulation using Pygame.
    
    This class handles:
    - Initializing the display
    - Rendering the world grid
    - Rendering agents
    - Rendering UI elements (stats, controls)
    - Handling user input
    
    Attributes:
        width (int): Width of the display.
        height (int): Height of the display.
        screen (pygame.Surface): Pygame display surface.
        camera: Camera for viewport control.
    """
    
    def __init__(self, width: int = 1280, height: int = 720):
        """
        Initialize the renderer.
        
        Args:
            width (int): Width of the display.
            height (int): Height of the display.
        """
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Display settings
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Predator-Prey Simulation")
        
        # Fonts
        self.font_small = pygame.font.SysFont("Arial", 14)
        self.font_medium = pygame.font.SysFont("Arial", 18)
        self.font_large = pygame.font.SysFont("Arial", 24)
        
        # Colors
        self.bg_color = (10, 10, 10)  # Dark background
        self.grid_color = (30, 30, 30)  # Darker grid lines
        self.text_color = (220, 220, 220)  # Light text
        self.ui_bg_color = (40, 40, 40, 180)  # Semi-transparent UI background
        
        # FPS control
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.frame_time = 0
        
        # Camera (to be set by main app)
        self.camera = None
        
        # UI state
        self.show_ui = True
        self.show_grid = True
        self.show_stats = True
        self.show_help = False
        self.is_paused = False
        self.sim_speed = 1.0  # Simulation speed multiplier
        
        # Selected agent
        self.selected_agent = None
        
        # Performance metrics
        self.render_time = 0
        self.sim_time = 0
        
        # Store for chart data
        self.stats_history = []
    
    def set_camera(self, camera) -> None:
        """
        Set the camera for viewport control.
        
        Args:
            camera: Camera object for viewport transformations.
        """
        self.camera = camera
    
    def handle_events(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle pygame events (keyboard, mouse).
        
        Returns:
            Tuple[bool, Dict[str, Any]]: (running flag, actions dict)
        """
        running = True
        actions = {}
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Key presses
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                    actions["pause_toggle"] = self.is_paused
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_u:
                    self.show_ui = not self.show_ui
                elif event.key == pygame.K_s:
                    self.show_stats = not self.show_stats
                elif event.key == pygame.K_r:
                    # Reset camera
                    if self.camera:
                        self.camera.reset()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    # Speed up simulation
                    self.sim_speed = min(8.0, self.sim_speed * 1.25)
                    actions["speed"] = self.sim_speed
                elif event.key == pygame.K_MINUS:
                    # Slow down simulation
                    self.sim_speed = max(0.25, self.sim_speed / 1.25)
                    actions["speed"] = self.sim_speed
                elif event.key == pygame.K_c:
                    # Clear selection
                    self.selected_agent = None
                elif event.key == pygame.K_f:
                    # Toggle follow selected agent
                    if self.selected_agent and self.camera:
                        actions["follow"] = self.selected_agent
            
            # Mouse events
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Select agent
                    mouse_pos = pygame.mouse.get_pos()
                    actions["select"] = self._screen_to_world(mouse_pos)
                elif event.button == 3:  # Right click
                    # Pan starts
                    actions["pan_start"] = pygame.mouse.get_pos()
                elif event.button == 4:  # Scroll up
                    # Zoom in
                    if self.camera:
                        self.camera.zoom_in()
                elif event.button == 5:  # Scroll down
                    # Zoom out
                    if self.camera:
                        self.camera.zoom_out()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:  # Right click released
                    # Pan ends
                    actions["pan_end"] = True
            
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[2]:  # Right mouse held
                    # Pan camera
                    dx, dy = event.rel
                    if self.camera:
                        self.camera.pan(-dx, -dy)
        
        return running, actions
    
    def render(self, world, stats: Dict[str, Any]) -> None:
        """
        Render the simulation world and UI.
        
        Args:
            world: World object to render.
            stats (Dict[str, Any]): Current simulation statistics.
        """
        start_time = time.time()
        
        # Clear screen
        self.screen.fill(self.bg_color)
        
        # Update camera if following an agent
        if self.camera:
            self.camera.update()
        
        # Render world grid
        if self.show_grid:
            self._render_grid(world)
        
        # Render agents
        self._render_agents(world)
        
        # Render UI
        if self.show_ui:
            self._render_ui(stats)
        
        # Render help
        if self.show_help:
            self._render_help()
        
        # Update display
        pygame.display.flip()
        
        # Track render time
        self.render_time = time.time() - start_time
        
        # Store stats for charts
        self.stats_history.append(stats)
        if len(self.stats_history) > 1000:
            # Keep only the most recent stats
            self.stats_history = self.stats_history[-1000:]
        
        # Cap framerate
        self.frame_time = self.clock.tick(self.fps) / 1000.0
    
    def _render_grid(self, world) -> None:
        """
        Render the world grid.
        
        Args:
            world: World object containing dimensions.
        """
        if not self.camera:
            return
        
        # Grid spacing in world units
        grid_spacing = 50
        
        # Calculate grid boundaries based on screen and camera
        left = int(self.camera.position[0] - self.width / (2 * self.camera.zoom))
        right = int(self.camera.position[0] + self.width / (2 * self.camera.zoom))
        top = int(self.camera.position[1] - self.height / (2 * self.camera.zoom))
        bottom = int(self.camera.position[1] + self.height / (2 * self.camera.zoom))
        
        # Adjust for grid spacing
        left = (left // grid_spacing) * grid_spacing
        right = ((right // grid_spacing) + 1) * grid_spacing
        top = (top // grid_spacing) * grid_spacing
        bottom = ((bottom // grid_spacing) + 1) * grid_spacing
        
        # Draw vertical grid lines
        for x in range(left, right + 1, grid_spacing):
            start_pos = self.camera.world_to_screen(np.array([x, top]), self.width, self.height)
            end_pos = self.camera.world_to_screen(np.array([x, bottom]), self.width, self.height)
            pygame.draw.line(self.screen, self.grid_color, start_pos, end_pos, 1)
        
        # Draw horizontal grid lines
        for y in range(top, bottom + 1, grid_spacing):
            start_pos = self.camera.world_to_screen(np.array([left, y]), self.width, self.height)
            end_pos = self.camera.world_to_screen(np.array([right, y]), self.width, self.height)
            pygame.draw.line(self.screen, self.grid_color, start_pos, end_pos, 1)
    
    def _render_agents(self, world) -> None:
        """
        Render all agents in the world.
        
        Args:
            world: World object containing agents.
        """
        if not self.camera:
            return
        
        # Get screen boundaries in world coordinates
        screen_left = self.camera.position[0] - self.width / (2 * self.camera.zoom)
        screen_right = self.camera.position[0] + self.width / (2 * self.camera.zoom)
        screen_top = self.camera.position[1] - self.height / (2 * self.camera.zoom)
        screen_bottom = self.camera.position[1] + self.height / (2 * self.camera.zoom)
        
        # Buffer for wraparound rendering
        buffer = 50
        
        # Render each agent
        for agent in world.agents:
            # Skip agents far outside the viewport
            if (agent.position[0] < screen_left - buffer or 
                agent.position[0] > screen_right + buffer or
                agent.position[1] < screen_top - buffer or
                agent.position[1] > screen_bottom + buffer):
                continue
            
            # Convert world position to screen position
            screen_pos = self.camera.world_to_screen(agent.position, self.width, self.height)
            
            # Calculate radius based on zoom level
            radius = max(3, int(4 * self.camera.zoom))
            
            # Get agent color
            color = agent.get_color()
            
            # Draw agent
            pygame.draw.circle(self.screen, color, screen_pos.astype(int), radius)
            
            # Highlight selected agent
            if agent == self.selected_agent:
                highlight_radius = radius + 3
                pygame.draw.circle(self.screen, (255, 255, 255), screen_pos.astype(int), highlight_radius, 1)
                
                # Draw energy bar above agent
                bar_width = 20
                bar_height = 4
                max_energy = 200  # Assumed maximum energy
                energy_ratio = agent.energy / max_energy
                
                # Bar background
                bar_bg_rect = pygame.Rect(
                    screen_pos[0] - bar_width // 2,
                    screen_pos[1] - radius - bar_height - 5,
                    bar_width,
                    bar_height
                )
                pygame.draw.rect(self.screen, (60, 60, 60), bar_bg_rect)
                
                # Energy fill
                energy_width = int(bar_width * energy_ratio)
                energy_rect = pygame.Rect(
                    screen_pos[0] - bar_width // 2,
                    screen_pos[1] - radius - bar_height - 5,
                    energy_width,
                    bar_height
                )
                
                # Color based on agent type
                if agent.type == "prey":
                    energy_color = (0, 200, 0)
                elif agent.type == "predator":
                    energy_color = (200, 0, 0)
                else:
                    energy_color = (200, 200, 0)
                
                pygame.draw.rect(self.screen, energy_color, energy_rect)
    
    def _render_ui(self, stats: Dict[str, Any]) -> None:
        """
        Render UI elements including statistics.
        
        Args:
            stats (Dict[str, Any]): Current simulation statistics.
        """
        # Background for UI panel
        panel_width = 250
        panel_height = self.height
        panel_rect = pygame.Rect(self.width - panel_width, 0, panel_width, panel_height)
        
        # Create a transparent surface for the panel
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill(self.ui_bg_color)
        self.screen.blit(panel_surface, panel_rect)
        
        # Draw title
        title_text = self.font_large.render("Predator-Prey Simulation", True, self.text_color)
        self.screen.blit(title_text, (self.width - panel_width + 10, 10))
        
        # Draw stats
        y_pos = 50
        line_height = 20
        
        # Format and display stats
        if self.show_stats:
            stat_lines = [
                f"Timestep: {stats.get('timestep', 0)}",
                f"Total Agents: {stats.get('total_agents', 0)}",
                f"Prey: {stats.get('prey_count', 0)}",
                f"Predators: {stats.get('predator_count', 0)}",
                f"Avg Prey Energy: {stats.get('avg_prey_energy', 0):.1f}",
                f"Avg Predator Energy: {stats.get('avg_predator_energy', 0):.1f}",
                "",
                f"Speed: {self.sim_speed:.2f}x",
                f"FPS: {int(1.0 / max(0.001, self.frame_time))}",
                f"Render Time: {self.render_time * 1000:.1f} ms",
                f"Sim Time: {self.sim_time * 1000:.1f} ms"
            ]
            
            for line in stat_lines:
                text = self.font_medium.render(line, True, self.text_color)
                self.screen.blit(text, (self.width - panel_width + 10, y_pos))
                y_pos += line_height
        
        # Draw selected agent info
        if self.selected_agent:
            y_pos += 20
            agent_info = [
                "Selected Agent:",
                f"Type: {self.selected_agent.type}",
                f"Energy: {self.selected_agent.energy:.1f}",
                f"Position: ({self.selected_agent.position[0]:.1f}, {self.selected_agent.position[1]:.1f})",
                f"Age: {self.selected_agent.age}"
            ]
            
            for line in agent_info:
                text = self.font_medium.render(line, True, self.text_color)
                self.screen.blit(text, (self.width - panel_width + 10, y_pos))
                y_pos += line_height
        
        # Draw simulation status
        y_pos = self.height - 50
        if self.is_paused:
            status_text = self.font_large.render("PAUSED", True, (255, 100, 100))
            self.screen.blit(status_text, (self.width - panel_width + 10, y_pos))
    
    def _render_help(self) -> None:
        """
        Render help information overlay.
        """
        # Background for help panel
        help_width = 400
        help_height = 400
        x_pos = (self.width - help_width) // 2
        y_pos = (self.height - help_height) // 2
        
        help_rect = pygame.Rect(x_pos, y_pos, help_width, help_height)
        
        # Create a transparent surface for the panel
        help_surface = pygame.Surface((help_width, help_height), pygame.SRCALPHA)
        help_surface.fill((20, 20, 20, 230))  # Dark with alpha
        self.screen.blit(help_surface, help_rect)
        
        # Draw title
        title_text = self.font_large.render("Controls", True, self.text_color)
        self.screen.blit(title_text, (x_pos + 10, y_pos + 10))
        
        # Draw help text
        controls = [
            "ESC - Quit",
            "SPACE - Pause/Resume",
            "G - Toggle Grid",
            "H - Toggle Help",
            "U - Toggle UI",
            "S - Toggle Stats",
            "R - Reset Camera",
            "+/- - Adjust Simulation Speed",
            "Left Click - Select Agent",
            "Right Click + Drag - Pan Camera",
            "Mouse Wheel - Zoom In/Out",
            "C - Clear Selection",
            "F - Follow Selected Agent"
        ]
        
        line_height = 24
        text_y = y_pos + 50
        
        for control in controls:
            text = self.font_medium.render(control, True, self.text_color)
            self.screen.blit(text, (x_pos + 20, text_y))
            text_y += line_height
    
    def _screen_to_world(self, screen_pos: Tuple[int, int]) -> np.ndarray:
        """
        Convert screen coordinates to world coordinates.
        
        Args:
            screen_pos (Tuple[int, int]): Screen coordinates.
            
        Returns:
            np.ndarray: World coordinates.
        """
        if not self.camera:
            return np.array(screen_pos)
        
        return self.camera.screen_to_world(
            np.array(screen_pos), self.width, self.height
        )
    
    def select_agent_at_position(self, world, position: np.ndarray, radius: float = 10.0) -> Optional[Any]:
        """
        Select an agent near the given position.
        
        Args:
            world: World containing agents.
            position (np.ndarray): Position to search around.
            radius (float): Search radius.
            
        Returns:
            Optional[Any]: Selected agent or None if none found.
        """
        # Find agents within radius
        agents_in_range = []
        
        for agent in world.agents:
            # Calculate distance with wraparound
            dx = min(abs(position[0] - agent.position[0]), 
                    world.width - abs(position[0] - agent.position[0]))
            dy = min(abs(position[1] - agent.position[1]), 
                    world.height - abs(position[1] - agent.position[1]))
            
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance <= radius / self.camera.zoom:
                agents_in_range.append((agent, distance))
        
        # Return closest agent if any found
        if agents_in_range:
            agents_in_range.sort(key=lambda x: x[1])
            self.selected_agent = agents_in_range[0][0]
            return self.selected_agent
        
        return None
    
    def set_simulation_time(self, time_taken: float) -> None:
        """
        Set the simulation step time for performance monitoring.
        
        Args:
            time_taken (float): Time taken for simulation step.
        """
        self.sim_time = time_taken
    
    def cleanup(self) -> None:
        """
        Clean up pygame resources.
        """
        pygame.quit()