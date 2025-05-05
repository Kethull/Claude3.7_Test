"""
Main entry point for predator-prey simulation.

This module provides command-line options for running the simulation
and handles the main simulation loop.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import pygame
from typing import Dict, Any, List, Tuple, Optional
import datetime

# Import local modules
from engine.world import World
from engine.camera import Camera
from engine.spatial import OptimizedSpatialGrid
from agents.prey import Prey
from agents.predator import Predator
from rl.policy import PPOPolicy
from rl.buffer import ExperienceBuffer
from rl.ppo import PPOTrainer
from ui.renderer import Renderer
from ui.charts import SimulationCharts
from utils.storage import SimulationStorage


# Default simulation parameters
DEFAULT_CONFIG = {
    "WORLD_WIDTH": 1900,
    "WORLD_HEIGHT": 900,
    "VISION_RAYS": 4,
    "VISION_RANGE": 100,
    "INITIAL_PREY": 50,
    "INITIAL_PREDATORS": 50,
    "PREY_REPRO_THRESHOLD": 120,
    "PRED_REPRO_THRESHOLD": 200,
    "LEARNING_RATE": 3e-4,
    "GAMMA": 0.99,
    "BATCH_SIZE": 128,
    "TRAIN_INTERVAL": 10,  # Timesteps between training
    "SAVE_INTERVAL": 5000,   # Timesteps between saving
    "REPLAY_INTERVAL": 1,    # Timesteps between replay saves
    "DISPLAY_WIDTH": 1900,
    "DISPLAY_HEIGHT": 900,
    "SPATIAL_CELL_SIZE": 30  # Cell size for spatial partitioning
}

# Add this to main.py
class LogLevel:
    NONE = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4

log_level = LogLevel.ERROR  # Only show errors by default


class SimulationApp:
    log_level = LogLevel.ERROR  # Only show errors by default
    """
    Main application class for the predator-prey simulation.
    
    This class:
    - Initializes the world, agents, and UI
    - Runs the main simulation loop
    - Handles saving and loading
    - Coordinates reinforcement learning
    
    Attributes:
        config (Dict[str, Any]): Simulation configuration parameters.
        world (World): Simulation world.
        camera (Camera): Camera for viewport control.
        renderer (Renderer): Renderer for visualization.
        charts (SimulationCharts): Charts for statistics visualization.
        is_training (bool): Whether to train policies.
        is_replay (bool): Whether in replay mode.
    """
    
    def __init__(self, config: Dict[str, Any], replay_file: Optional[str] = None):
        """
        Initialize the simulation application.
        
        Args:
            config (Dict[str, Any]): Simulation configuration.
            replay_file (Optional[str]): Path to replay file, if replaying.
        """
        self.config = config
        
        # Initialize components
        self.world = World(config["WORLD_WIDTH"], config["WORLD_HEIGHT"])
        self.camera = Camera(config["WORLD_WIDTH"], config["WORLD_HEIGHT"])
        self.renderer = Renderer(config["DISPLAY_WIDTH"], config["DISPLAY_HEIGHT"])
        self.charts = SimulationCharts()
        
        print(f"Initialized world with dimensions: {config['WORLD_WIDTH']}x{config['WORLD_HEIGHT']}")
        print(f"Initialized renderer with dimensions: {config['DISPLAY_WIDTH']}x{config['DISPLAY_HEIGHT']}")
        
        # Connect components
        self.renderer.set_camera(self.camera)
        print("Camera set in renderer")
        
        # Set up spatial partitioning
        cell_size = config["SPATIAL_CELL_SIZE"]
        self.spatial_grid = OptimizedSpatialGrid(
            config["WORLD_WIDTH"], 
            config["WORLD_HEIGHT"],
            cell_size
        )
        self.world.set_spatial_index(self.spatial_grid)
        
        # Learning components
        self.is_training = True  # Enable training by default for blank-slate learning
        obs_dim = config["VISION_RAYS"] * 4 + 1  # Distance + type (one-hot: 3) + energy
        action_dim = 5  # stay, up, down, right, left
        
        self.prey_policy = PPOPolicy(obs_dim, action_dim)
        self.predator_policy = PPOPolicy(obs_dim, action_dim)
        
        self.prey_trainer = PPOTrainer(
            obs_dim, action_dim, 
            lr=config["LEARNING_RATE"], 
            gamma=config["GAMMA"]
        )
        self.predator_trainer = PPOTrainer(
            obs_dim, action_dim, 
            lr=config["LEARNING_RATE"], 
            gamma=config["GAMMA"]
        )
        
        self.prey_buffer = ExperienceBuffer(obs_dim, capacity=10000)
        self.predator_buffer = ExperienceBuffer(obs_dim, capacity=10000)
        
        # Replay mode
        self.is_replay = replay_file is not None
        self.replay_file = replay_file
        self.replay_storage = None
        self.replay_timesteps = []
        self.current_replay_index = 0
        
        # Stats
        self.stats = {}
        self.stats_history = []
        
        # Storage for saving simulation
        self.storage_filename = None
        if not self.is_replay:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.storage_filename = f"simulation_{timestamp}.h5"
            self.storage = SimulationStorage(self.storage_filename)
        else:
            self.storage = SimulationStorage(replay_file)
            self._setup_replay()
    
    def initialize_world(self) -> None:
        """
        Initialize the world with agents.
        """
        if self.is_replay:
            # In replay mode, load from file
            return
        
        # Create initial prey
        print(f"Creating {self.config['INITIAL_PREY']} prey agents...")
        for i in range(self.config["INITIAL_PREY"]):
            pos = np.random.rand(2) * np.array([self.config["WORLD_WIDTH"], self.config["WORLD_HEIGHT"]])
            prey = Prey(pos, energy=np.random.uniform(80, 120))
            prey.policy = self.prey_policy
            prey.reproduction_threshold = self.config["PREY_REPRO_THRESHOLD"]
            self.world.add_agent(prey)
        
        # Create initial predators
        print(f"Creating {self.config['INITIAL_PREDATORS']} predator agents...")
        for i in range(self.config["INITIAL_PREDATORS"]):
            pos = np.random.rand(2) * np.array([self.config["WORLD_WIDTH"], self.config["WORLD_HEIGHT"]])
            predator = Predator(pos, energy=np.random.uniform(130, 170))
            predator.policy = self.predator_policy
            predator.reproduction_threshold = self.config["PRED_REPRO_THRESHOLD"]
            self.world.add_agent(predator)
        
        print(f"World initialized with {len(self.world.agents)} total agents")
    
    def _setup_replay(self) -> None:
        """
        Set up replay mode from saved file.
        """
        try:
            self.storage.open('r')
            self.replay_timesteps = self.storage.get_timesteps()
            self.storage.close()
            
            if not self.replay_timesteps:
                print("Error: No timesteps found in replay file.")
                sys.exit(1)
            
            print(f"Loaded {len(self.replay_timesteps)} timesteps for replay.")
            
            # Load initial state
            self.storage.open('r')
            self.storage.load_world_state(self.world, self.replay_timesteps[0])
            
            # Load policies if available
            try:
                self.storage.load_policy_weights(self.prey_policy, 'prey', self.replay_timesteps[0])
                self.storage.load_policy_weights(self.predator_policy, 'predator', self.replay_timesteps[0])
            except Exception as e:
                print(f"Warning: Could not load policies: {e}")
            
            self.storage.close()
            
        except Exception as e:
            print(f"Error setting up replay: {e}")
            sys.exit(1)
    
    # Replace the run method in SimulationApp class
    def run_optimized(self) -> None:
        """
        Run the optimized main simulation loop.
        """
        log(LogLevel.INFO, "\n=== Starting Simulation ===")
        
        # Initialize pygame and world
        if not pygame.get_init():
            pygame.init()
        
        self.initialize_world()
        
        # Main loop variables
        running = True
        paused = False
        sim_speed = 1.0
        render_interval = 0.05  # Render every 50ms (20 FPS target)
        last_render_time = 0
        
        # Main simulation loop
        while running:
            current_time = time.time()
            
            # Handle events at full speed
            running, actions = self.renderer.handle_events()
            
            # Process actions
            if "pause_toggle" in actions:
                paused = actions["pause_toggle"]
            
            if "speed" in actions:
                sim_speed = actions["speed"]
            
            if "select" in actions:
                pos = actions["select"]
                self.renderer.select_agent_at_position(self.world, pos)
            
            if "follow" in actions:
                self.camera.follow(actions["follow"])
            
            # Simulation step (decoupled from rendering)
            if not paused:
                # Limit rendering to target FPS, but run simulation at full speed
                render_now = current_time - last_render_time >= render_interval
                
                # Simulation time measurement
                sim_start_time = time.time()
                
                if self.is_replay:
                    # Replay mode logic (unchanged)
                    if self.current_replay_index < len(self.replay_timesteps):
                        timestep = self.replay_timesteps[self.current_replay_index]
                        self.storage.open('r')
                        self.storage.load_world_state(self.world, timestep)
                        stats = self.storage.get_stats(timestep)
                        self.storage.close()
                        self.stats = stats if stats else {}
                        self.current_replay_index += 1
                    else:
                        paused = True
                        log(LogLevel.INFO, "End of replay reached.")
                else:
                    # Normal simulation mode - run multiple steps based on speed
                    steps = max(1, int(sim_speed))
                    for _ in range(steps):
                        self._simulation_step()
                
                # Track simulation time
                sim_time = time.time() - sim_start_time
                self.renderer.set_simulation_time(sim_time)
                
                # Render only if enough time has passed since last render
                if render_now:
                    # Update charts and render (moved outside simulation steps)
                    self.charts.update(self.stats_history)
                    self.renderer.render(self.world, self.stats)
                    last_render_time = current_time
                    
                    # Cap framerate to avoid burning CPU on rendering
                    elapsed = time.time() - current_time
                    if elapsed < render_interval:
                        time.sleep(render_interval - elapsed)
            else:
                # When paused, render at a low rate to save CPU
                time.sleep(0.1)  # 10 FPS when paused
                self.renderer.render(self.world, self.stats)
        
        # Cleanup when done
        self.cleanup()
    
    def _simulation_step(self) -> None:
        """
        Perform a single simulation step.
        """
        # Step the world
        self.world.step_optimized()
        
        # Collect experience for RL training
        if self.is_training:
            self._collect_experience()
        
        # Train policies periodically
        if self.is_training and self.world.timestep % self.config["TRAIN_INTERVAL"] == 0:
            self._train_policies()
        
        # Update stats
        self.stats = self.world.get_stats()
        self.stats_history.append(self.stats.copy())
        if len(self.stats_history) > 1000:
            self.stats_history = self.stats_history[-1000:]
        
        # Save simulation state periodically
        if self.storage_filename and self.world.timestep % self.config["SAVE_INTERVAL"] == 0:
            self._save_simulation()
    
    def _collect_experience(self) -> None:
        """
        Collect experience from agents for reinforcement learning.
        """
        # For each agent
        for agent in self.world.agents:
            if not agent.alive:
                continue
            
            # Get observation and convert to tensor format
            observation = self.world.get_observation(agent)
            obs_vector = agent._prepare_observation(observation)
            
            # Determine action using current policy
            if agent.type == "prey" and agent.policy is not None:
                # Get prey action and value
                action_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0)
                action, log_prob, value = agent.policy.get_action_and_value(action_tensor)
                
                # Execute action (happens in world.step())
                # Store action for reward calculation in next step
                observation["last_action"] = action.item()
                
                # Calculate reward
                reward = 0.0
                
                # Base survival reward
                reward += 0.1  # Small positive reward for staying alive
                
                # Energy-based reward components
                if agent.energy > agent.reproduction_threshold:
                    reward += 1.0  # Bonus for reaching reproduction threshold
                
                # Energy gain reward (if staying still)
                if observation.get("last_action", None) == 0:  # Stay action
                    reward += 0.2  # Reward for energy-positive behavior
                
                # Store experience in buffer
                # We'll need next observation when agent acts again next time, so this is partial
                self.prey_buffer.add(
                    obs_vector,
                    action.item(),
                    reward,
                    obs_vector.copy(),  # Will be updated with next step
                    False,  # Will be set properly in next step
                    log_prob.item(),
                    value.item()
                )
                
            elif agent.type == "predator" and agent.policy is not None:
                # Get predator action and value
                action_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0)
                action, log_prob, value = agent.policy.get_action_and_value(action_tensor)
                
                # Store action for reward calculation in next step
                observation["last_action"] = action.item()
                
                # Calculate reward
                reward = 0.0
                
                # Base survival reward
                reward += 0.1  
                
                # Energy-based reward
                if agent.energy > agent.reproduction_threshold:
                    reward += 1.5  # Larger bonus than prey for reproduction threshold
                
                # Hunt reward (handled in world step when eating prey)
                # We'll add +2.0 reward for successful hunts in the world._handle_interactions method
                
                # Exploration penalty for staying still
                # if observation.get("last_action", None) == 0:  # Stay action
                #     reward -= 0.1  # Small penalty for not hunting
                
                # Store experience in buffer
                self.predator_buffer.add(
                    obs_vector,
                    action.item(),
                    reward,
                    obs_vector.copy(),  # Will be updated with next step
                    False,  # Will be set properly in next step
                    log_prob.item(),
                    value.item()
                )
                
        # Update experience records for agents that died or performed actions in previous steps
        # This would track completed experiences with actual next states and done flags
        # Implementation would require tracking agent IDs and previous observations
    
    def _train_policies(self) -> None:
        """
        Train policies using collected experience.
        """
        # Train prey policy
        if self.prey_buffer.size > self.config["BATCH_SIZE"]:
            batch = self.prey_buffer.get_batch(self.config["BATCH_SIZE"])
            prey_stats = self.prey_trainer.train_batch(batch)
            print(f"Prey training - value loss: {prey_stats['value_loss']:.4f}, policy loss: {prey_stats['policy_loss']:.4f}")
        
        # Train predator policy
        if self.predator_buffer.size > self.config["BATCH_SIZE"]:
            batch = self.predator_buffer.get_batch(self.config["BATCH_SIZE"])
            pred_stats = self.predator_trainer.train_batch(batch)
            print(f"Predator training - value loss: {pred_stats['value_loss']:.4f}, policy loss: {pred_stats['policy_loss']:.4f}")
    
    def _save_simulation(self) -> None:
        """
        Save the current simulation state to storage.
        """
        try:
            # Open storage
            self.storage.open('a')
            
            # Save world state
            self.storage.save_world_state(self.world, self.world.timestep)
            
            # Save policies
            self.storage.save_policy_weights(self.prey_policy, 'prey', self.world.timestep)
            self.storage.save_policy_weights(self.predator_policy, 'predator', self.world.timestep)
            
            # Save stats
            self.storage.save_stats(self.stats, self.world.timestep)
            
            # Close storage
            self.storage.close()
            
            print(f"Saved simulation state at timestep {self.world.timestep}")
            
        except Exception as e:
            print(f"Error saving simulation: {e}")
    
    def export_stats(self, filename: str) -> None:
        """
        Export simulation statistics to CSV.
        
        Args:
            filename (str): Path to CSV file.
        """
        if self.storage_filename:
            try:
                self.storage.export_stats_to_csv(filename)
                print(f"Exported statistics to {filename}")
            except Exception as e:
                print(f"Error exporting statistics: {e}")
    
    def cleanup(self) -> None:
        """
        Clean up resources.
        """
        self.renderer.cleanup()
        if self.storage_filename:
            self.storage.close()
        
        # Export final stats
        if not self.is_replay and self.storage_filename:
            csv_name = self.storage_filename.replace('.h5', '.csv')
            self.export_stats(csv_name)


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Predator-Prey Simulation with RL")
    
    parser.add_argument("--width", type=int, default=DEFAULT_CONFIG["WORLD_WIDTH"],
                        help="Width of the world")
    parser.add_argument("--height", type=int, default=DEFAULT_CONFIG["WORLD_HEIGHT"],
                        help="Height of the world")
    parser.add_argument("--prey", type=int, default=DEFAULT_CONFIG["INITIAL_PREY"],
                        help="Initial number of prey")
    parser.add_argument("--predators", type=int, default=DEFAULT_CONFIG["INITIAL_PREDATORS"],
                        help="Initial number of predators")
    parser.add_argument("--train", action="store_true",
                        help="Enable reinforcement learning training")
    parser.add_argument("--replay", type=str, default=None,
                        help="Replay from the specified HDF5 file")
    parser.add_argument("--load", type=str, default=None,
                        help="Load existing policies from file")
    parser.add_argument("--display-width", type=int, default=DEFAULT_CONFIG["DISPLAY_WIDTH"],
                        help="Width of the display window")
    parser.add_argument("--display-height", type=int, default=DEFAULT_CONFIG["DISPLAY_HEIGHT"],
                        help="Height of the display window")
    
    return parser.parse_args()


def log(level, message):
    """Conditionally print log messages based on current log level."""
    if level <= log_level:
        print(message)


def main():
    """
    Main function to run the simulation.
    """
    print("Starting predator-prey simulation...")
    
    # Check pygame initialization
    print("Checking pygame initialization...")
    if not pygame.get_init():
        print("Initializing pygame...")
        pygame.init()
    else:
        print("Pygame already initialized")
    
    args = parse_args()
    
    # Configure simulation
    config = DEFAULT_CONFIG.copy()
    config["WORLD_WIDTH"] = args.width
    config["WORLD_HEIGHT"] = args.height
    config["INITIAL_PREY"] = args.prey
    config["INITIAL_PREDATORS"] = args.predators
    config["DISPLAY_WIDTH"] = args.display_width
    config["DISPLAY_HEIGHT"] = args.display_height
    
    # Create and run application
    app = SimulationApp(config, replay_file=args.replay)
    app.is_training = args.train
    
    # Load policies if specified
    if args.load and not args.replay:
        try:
            storage = SimulationStorage(args.load)
            storage.open('r')
            
            # Load policies
            storage.load_policy_weights(app.prey_policy, 'prey', 0)
            storage.load_policy_weights(app.predator_policy, 'predator', 0)
            
            storage.close()
            print(f"Loaded policies from {args.load}")
        except Exception as e:
            print(f"Error loading policies: {e}")
    
    # Run simulation
    app.run_optimized()


if __name__ == "__main__":
    main()