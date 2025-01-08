from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import numpy as np
from datetime import datetime
import time

@dataclass
class TrainerConfig:
    batch_size: int
    learning_rate: float
    gamma: float
    device: str
    num_episodes: int
    frame_skip: int
    stack_size: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    model_save_freq: int
    validation_episodes: int
    grad_clip_norm: float
    memory_capacity: int
    tau: float
    n_step: int
    use_best_model_callback: bool
    best_model_smoothing_window: int

class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_survival_times = []
        self.damage_taken = []
        self.damage_inflicted = []
        self.ram_usage = []
        self.gpu_memory_usage = []
        self.action_diversity = []
        self.losses = []
        self.q_values = []
        self.grad_norms = []
        
    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if hasattr(self, key):
                getattr(self, key).append(value)

class Trainer:
    def __init__(self, config: TrainerConfig, resource_manager, logger):
        self.config = config
        self.resource_manager = resource_manager
        self.logger = logger
        self.device = torch.device(config.device)
        self.metrics = TrainingMetrics()
        self.best_reward = float('-inf')
        self.early_stopping_counter = 0
        self.early_stopping_patience = 5
        self.training_start_time = None
        
    def train_episode(self, agent, environment, episode_num: int) -> Dict[str, Any]:
        """Run a single training episode with resource monitoring"""
        with self.resource_manager.monitor_resources():
            try:
                # Initialize episode
                if self.training_start_time is None:
                    self.training_start_time = time.time()
                
                environment.new_episode()
                state = self._get_initial_state(environment)
                
                # Episode tracking
                total_reward = 0
                step_count = 0
                episode_start_time = time.time()
                damage_taken = 0
                damage_inflicted = 0
                
                # Run episode steps
                while not environment.is_episode_finished():
                    # Select and perform action
                    action_idx = agent.select_action(state)
                    reward = environment.make_action(action_idx, self.config.frame_skip)
                    
                    # Get next state
                    if not environment.is_episode_finished():
                        next_state = self._process_state(environment)
                        damage_step = self._get_damage_info(environment)
                        damage_taken += damage_step[0]
                        damage_inflicted += damage_step[1]
                    else:
                        next_state = None
                    
                    # Store transition and learn
                    if next_state is not None:
                        agent.memory.push(state, action_idx, reward, next_state, False)
                        loss_info = agent.learn()
                    
                    state = next_state
                    total_reward += reward
                    step_count += 1
                
                # Episode metrics
                episode_metrics = {
                    'episode': episode_num,
                    'reward': total_reward,
                    'steps': step_count,
                    'survival_time': time.time() - episode_start_time,
                    'damage_taken': damage_taken,
                    'damage_inflicted': damage_inflicted,
                    'ram_usage': self.resource_manager.get_ram_usage(),
                    'gpu_memory': self.resource_manager.get_gpu_memory() if self.resource_manager.gpu_available else 0
                }
                
                self.metrics.update(episode_metrics)
                
                # Update agent's exploration rate
                agent.update_epsilon()
                
                return episode_metrics
                
            except Exception as e:
                self.logger.error(f"Error in training episode {episode_num}: {e}")
                return None

    def validate_episode(self, agent, environment, episode_num: int) -> Dict[str, Any]:
        """Run a single validation episode"""
        with self.resource_manager.monitor_resources():
            try:
                # Store training state
                original_epsilon = agent.epsilon
                agent.epsilon = 0.05  # Minimal exploration during validation
                
                # Run validation episode
                environment.new_episode()
                state = self._get_initial_state(environment)
                
                total_reward = 0
                step_count = 0
                start_time = time.time()
                
                while not environment.is_episode_finished():
                    action_idx = agent.select_action(state)
                    reward = environment.make_action(action_idx, self.config.frame_skip)
                    
                    if not environment.is_episode_finished():
                        next_state = self._process_state(environment)
                        state = next_state
                    
                    total_reward += reward
                    step_count += 1
                
                # Validation metrics
                validation_metrics = {
                    'episode': episode_num,
                    'reward': total_reward,
                    'steps': step_count,
                    'duration': time.time() - start_time
                }
                
                # Check for best model
                if total_reward > self.best_reward:
                    self.best_reward = total_reward
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                # Restore training state
                agent.epsilon = original_epsilon
                
                return validation_metrics
                
            except Exception as e:
                self.logger.error(f"Error in validation episode {episode_num}: {e}")
                return None
            
    def _get_initial_state(self, environment):
        """Process and return initial state"""
        state = environment.get_state()
        if state is None:
            return np.zeros(self.config.stack_size)
        return self._process_state(environment)
    
    def _process_state(self, environment):
        """Process environment state"""
        state = environment.get_state()
        if state is not None:
            # Add state processing logic here
            return state
        return np.zeros(self.config.stack_size)
    
    def _get_damage_info(self, environment):
        """Get damage information from environment"""
        state = environment.get_state()
        if state is not None and hasattr(state, 'game_variables'):
            return state.game_variables[0], state.game_variables[1]
        return 0, 0
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early"""
        return self.early_stopping_counter >= self.early_stopping_patience
