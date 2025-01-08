import os
import json
import yaml
import shutil
import time
import random
import logging
import glob
import psutil
import gc
import math  # Add math import
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import imageio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from vizdoom import DoomGame, ScreenFormat, ScreenResolution, Mode  # Import DoomGame

# --- Create config.yaml if it doesn't exist ---
CONFIG_PATH = "config.yaml"  # Path to your config file

default_config = {
    "drive_model_dir": "/content/drive/My Drive/ViZDoomModels",
    "drive_wad_dir": "/content/drive/My Drive/ViZDoomWADs",
    "local_wad_dir": "/content/ViZDoomWADs",
    "video_dir": "/content/drive/My Drive/ViZDoomRecordings",
    "video_filename": "game_recording.mp4",
    "record_lmp": False,
    "record_video": True,
    "video_fps": 30,
    "lmp_dir": "lmp_recordings",
    "scenario_name": "defend_the_center.cfg",
    "stack_size": 4,
    "num_episodes": 10,
    "frame_skip_training": 4,
    "frame_skip_recording": 1,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.995,
    "model_save_freq": 10,
    "batch_size": 64,
    "memory_capacity": 10000,
    "learning_rate": 0.0001,
    "tau": 0.005,
    "n_step": 3,
    "grad_clip_norm": 1.0,
    "epsilon_decay_rate_step": 1000,
    "use_best_model_callback": True,
    "best_model_smoothing_window": 10,
    "validation_episodes": 5,
}

if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(default_config, f)

# --- Configuration Loading ---
def load_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

CONFIG = load_config()

# --- Logging Setup ---
def setup_logger(log_dir, timestamp):
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logger = logging.getLogger(__name__)

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

# --- Google Drive Setup ---
DRIVE_MODEL_DIR = CONFIG["drive_model_dir"]
DRIVE_WAD_DIR = CONFIG["drive_wad_dir"]
LOCAL_WAD_DIR = CONFIG["local_wad_dir"]
VIDEO_DIR = CONFIG["video_dir"]
VIDEO_FILENAME = CONFIG["video_filename"]
VIDEO_PATH = os.path.join(VIDEO_DIR, VIDEO_FILENAME)

# --- Configuration Variables ---
RECORD_LMP = CONFIG["record_lmp"]
RECORD_VIDEO = CONFIG["record_video"]
VIDEO_FPS = CONFIG["video_fps"]
LMP_DIR = CONFIG["lmp_dir"]
SCENARIO_NAME = CONFIG["scenario_name"]
STACK_SIZE = CONFIG["stack_size"]
NUM_EPISODES = CONFIG["num_episodes"]
FRAME_SKIP_TRAINING = CONFIG["frame_skip_training"]
FRAME_SKIP_RECORDING = CONFIG["frame_skip_recording"]

# --- Training Parameters ---
GAMMA = CONFIG["gamma"]
EPSILON_START = CONFIG["epsilon_start"]
EPSILON_END = CONFIG["epsilon_end"]
EPSILON_DECAY = CONFIG["epsilon_decay"]
MODEL_SAVE_FREQ = CONFIG["model_save_freq"]
BATCH_SIZE = CONFIG["batch_size"]
MEMORY_CAPACITY = CONFIG["memory_capacity"]
LEARNING_RATE = CONFIG["learning_rate"]
TAU = CONFIG["tau"]
N_STEP = CONFIG["n_step"]
GRAD_CLIP_NORM = CONFIG["grad_clip_norm"]
EPSILON_DECAY_RATE_STEP = CONFIG["epsilon_decay_rate_step"]
USE_BEST_MODEL_CALLBACK = CONFIG["use_best_model_callback"]
BEST_MODEL_SMOOTHING_WINDOW = CONFIG["best_model_smoothing_window"]
VALIDATION_EPISODES = CONFIG["validation_episodes"]

# --- Setup Google Drive ---
def setup_google_drive(logger):
    drive_mounted = os.path.exists('/content/drive/My Drive')
    if not drive_mounted:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("Google Drive mounted successfully.")
        except Exception as e:
            logger.error(f"Error mounting Google Drive: {e}")
            return False

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
    os.makedirs(DRIVE_WAD_DIR, exist_ok=True)
    os.makedirs(LOCAL_WAD_DIR, exist_ok=True)
    logger.info("Google Drive directories created successfully.")
    return True

def copy_scenarios_to_drive(logger):
    local_scenarios_dir = "/usr/local/lib/python3.10/dist-packages/vizdoom/scenarios"
    wad_files = glob.glob(os.path.join(local_scenarios_dir, "*.wad"))
    for wad_file in wad_files:
        dest_file = os.path.join(DRIVE_WAD_DIR, os.path.basename(wad_file))
        if os.path.exists(dest_file):
            logger.info(f"Skipped existing: {os.path.basename(wad_file)}")
        else:
            try:
                shutil.copy(wad_file, dest_file)
                logger.info(f"Copied: {os.path.basename(wad_file)}")
            except Exception as e:
                logger.error(f"Error copying: {os.path.basename(wad_file)}, error: {e}")
    return True

# --- User Input for Recording ---
def get_record_choices(logger):
    logger.info(f"Record LMP: {RECORD_LMP}, Record Video: {RECORD_VIDEO}")
    return RECORD_LMP, RECORD_VIDEO

# --- User Input for Training Episodes ---
def get_num_episodes(logger):
    logger.info(f"Number of Episodes: {NUM_EPISODES}")
    return NUM_EPISODES

# --- User Input for FRAME_SKIP ---
def get_frame_skips(logger):
    FRAME_SKIP = FRAME_SKIP_TRAINING if not RECORD_VIDEO else FRAME_SKIP_RECORDING
    logger.info(f"Frame Skip Training: {FRAME_SKIP_TRAINING}, Frame Skip Recording: {FRAME_SKIP_RECORDING}")
    return FRAME_SKIP_TRAINING, FRAME_SKIP_RECORDING, FRAME_SKIP

def get_wad_choice(logger):
    print("Choose a WAD file to use:")
    print("1. Use current scenario WAD (defend_the_center.cfg with vizdoom assets)")
    print("2. Use original Doom WAD from Google Drive (ViZDoomWADs folder with defend_the_center.cfg)")
    wad_choice = input("Enter your choice (1 or 2): ")
    logger.info(f"WAD Choice: {wad_choice}")
    return wad_choice

def setup_scenario(wad_choice, logger):
    global SCENARIO_NAME
    SCENARIO_NAME = "defend_the_center.cfg"

    if wad_choice == "1":
        try:
            import vizdoom
            vizdoom_path = os.path.dirname(vizdoom.__file__)
            SCENARIO_PATH = os.path.join(vizdoom_path, "scenarios", SCENARIO_NAME)
            wad_path = None
            logger.info(f"Using default scenario path: {SCENARIO_PATH}")
            return SCENARIO_PATH, wad_path
        except Exception as e:
            logger.error(f"Error loading default scenario: {e}")
            # Fallback: Try to use a path relative to the current file
            SCENARIO_PATH = os.path.join("scenarios", SCENARIO_NAME)
            logger.warning(f"Trying fallback scenario path: {SCENARIO_PATH}")
            return SCENARIO_PATH, None

    elif wad_choice == "2":
        wad_files = glob.glob(os.path.join(DRIVE_WAD_DIR, "*.wad"))
        if not wad_files:
            logger.warning("No WAD files found in ViZDoomWADs. Using default scenario.")
            return setup_scenario("1", logger)
        print("Available WAD files in ViZDoomWADs:")
        for i, file in enumerate(wad_files):
            print(f"{i+1}. {os.path.basename(file)}")
        wad_file_choice = input("Enter the number of the WAD file to use: ")
        try:
            wad_file_choice = int(wad_file_choice)
            if 1 <= wad_file_choice <= len(wad_files):
                wad_path = wad_files[wad_file_choice - 1]
                local_wad_path = os.path.join(LOCAL_WAD_DIR, os.path.basename(wad_path))
                if not os.path.exists(local_wad_path):
                    print(f"Copying {os.path.basename(wad_path)} from Google Drive to local...")
                    try:
                        shutil.copy(wad_path, local_wad_path)
                        logger.info(f"Copied {os.path.basename(wad_path)} from Google Drive to local.")
                    except Exception as e:
                        logger.error(f"Error copying wad file to local: {e}")
                        return setup_scenario("1", logger)
                else:
                    print(f"{os.path.basename(wad_path)} already exists locally. Using local copy.")
                    logger.info(f"{os.path.basename(wad_path)} already exists locally.")
                wad_path = local_wad_path
                import vizdoom
                vizdoom_path = os.path.dirname(vizdoom.__file__)
                SCENARIO_PATH = os.path.join(vizdoom_path, "scenarios", SCENARIO_NAME)
                logger.info(f"Using WAD file: {wad_path}, scenario path: {SCENARIO_PATH}")
                return SCENARIO_PATH, wad_path
            else:
                logger.warning("Invalid WAD file choice. Using default scenario.")
                return setup_scenario("1", logger)
        except ValueError:
            logger.warning("Invalid input. Using default scenario.")
            return setup_scenario("1", logger)
    else:
        logger.warning("Invalid choice. Using default scenario.")
        return setup_scenario("1", logger)

# --- DQN ---
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.get_conv_output(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_conv_output(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        return F.linear(x,
                       self.weight_mu + self.weight_sigma * self.weight_epsilon,
                       self.bias_mu + self.bias_sigma * self.bias_epsilon)

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions, use_noisy=True):
        super(DuelingDQN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_output(input_shape)

        # Value stream
        if use_noisy:
            self.value_fc = NoisyLinear(conv_out_size, 512)
            self.value = NoisyLinear(512, 1)
        else:
            self.value_fc = nn.Linear(conv_out_size, 512)
            self.value = nn.Linear(512, 1)

        # Advantage stream
        if use_noisy:
            self.advantage_fc = NoisyLinear(conv_out_size, 512)
            self.advantage = NoisyLinear(512, num_actions)
        else:
            self.advantage_fc = nn.Linear(conv_out_size, 512)
            self.advantage = nn.Linear(512, num_actions)

        self.use_noisy = use_noisy
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        value = F.relu(self.value_fc(x))
        value = self.value(value)

        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)

        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        if self.use_noisy:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()

# --- Prioritized Experience Replay Memory ---
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        beta = min(1.0, self.beta_start + (self.frame * (1 - self.beta_start) / self.beta_frames))
        weights = (len(self.memory) * probabilities[indices])**(-beta)
        weights /= weights.max()
        self.frame += 1

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        self.priorities[indices] = np.abs(td_errors) + 1e-6

    def __len__(self):
        return len(self.memory)

# --- N-step Buffer ---
class NStepBuffer:
    def __init__(self, n_step):
        self.n_step = n_step
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def get(self):
        if len(self.buffer) < self.n_step:
            return None

        n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done = self.buffer[0]
        for i in range(1, self.n_step):
            state, action, reward, next_state, done = self.buffer[i]
            n_step_reward += reward * GAMMA**i
            n_step_next_state = next_state
            n_step_done = done or n_step_done

        self.buffer = self.buffer[1:]
        return n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done

    def __len__(self):
        return len(self.buffer)

# --- DQNAgent ---
class DQNAgent:
    def __init__(self, state_shape, num_actions, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay, memory_capacity, batch_size, tau, n_step, grad_clip_norm, epsilon_decay_rate_step, use_noisy=True):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(state_shape, num_actions, use_noisy=use_noisy).to(self.device)
        self.target_net = DuelingDQN(state_shape, num_actions, use_noisy=use_noisy).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = PrioritizedReplayMemory(memory_capacity)
        self.n_step_buffer = NStepBuffer(n_step)
        self.epsilon_decay_rate_step = epsilon_decay_rate_step
        self.training_step = 0
        self.best_avg_reward = float('-inf')
        self.best_model_state = None

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.argmax(dim=1).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_start * (self.epsilon_decay ** (self.training_step / self.epsilon_decay_rate_step)))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None, None, None
        self.training_step += 1
        samples, indices, weights = self.memory.sample(self.batch_size)
        batch = tuple(zip(*samples))

        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values_online = self.policy_net(next_state_batch)
            next_actions = next_q_values_online.argmax(dim=1).unsqueeze(1)
            next_q_values_target = self.target_net(next_state_batch).gather(1, next_actions)
            expected_q_values = reward_batch + self.gamma * next_q_values_target * (1 - done_batch)

        td_errors = expected_q_values - q_values
        loss = (weights * F.smooth_l1_loss(q_values, expected_q_values, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy().squeeze())
        return loss.item(), q_values.mean().item(), torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm).item()

    def update_target_network(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, model_path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())

def scan_for_models(agent, model_dir, logger):
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))

    if not model_files:
        logger.info("No existing models found. Creating new model...")
        return agent  # Return the existing agent instead of creating new one

    print("Available models:")
    for i, file in enumerate(model_files):
        print(f"{i+1}. {os.path.basename(file)}")
    print(f"{len(model_files)+1}. Create new model")

    choice = input("Enter your choice: ")
    try:
        choice = int(choice)
        if choice in range(1, len(model_files) + 2):
            if choice <= len(model_files):
                model_path = model_files[choice - 1]
                try:
                    agent.load_model(model_path)
                    logger.info(f"Model loaded from {model_path}")
                    return agent
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    return agent  # Return existing agent on error
            else:
                logger.info("Creating new model...")
                return agent  # Return existing agent for new model
        else:
            logger.warning("Invalid choice. Using existing model.")
            return agent
    except ValueError:
        logger.warning("Invalid input. Using existing model.")
        return agent

def setup_vizdoom(scenario_path, wad_path=None, logger=None):
    try:
        from vizdoom import DoomGame, ScreenFormat, ScreenResolution, Mode
        game = DoomGame()

        if scenario_path:
            game.load_config(scenario_path)
        else:
            raise ValueError("Scenario path is not defined.")

        if wad_path:
            game.set_doom_game_path(wad_path)
        game.set_window_visible(False)
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_screen_resolution(ScreenResolution.RES_320X240)

        if RECORD_LMP:
            game.set_mode(Mode.PLAYER)
            os.makedirs(LMP_DIR, exist_ok=True)

        game.init()
        num_actions = game.get_available_buttons_size()
        actions = np.identity(num_actions, dtype=int).tolist()

        screen_height, screen_width = game.get_screen_height(), game.get_screen_width()
        channels = game.get_screen_channels()
        state_shape = (STACK_SIZE * channels, 84, 84)
        logger.info("ViZDoom initialized successfully.")
        return game, actions, state_shape
    except Exception as e:
        if logger:
            logger.error(f"Error during ViZDoom setup: {e}")
        print(f"Error during ViZDoom setup: {e}")
        return None, None, None

# --- Video Writer ---
def setup_video_writer(video_path, video_fps, logger=None):
    """Setup video writer with explicit path"""
    if video_path:
        try:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            video_writer = imageio.get_writer(video_path, fps=video_fps)
            if logger:
                logger.info(f"Video Writer set up at: {video_path}")
            return video_writer
        except Exception as e:
            if logger:
                logger.error(f"Error during video writer setup: {e}")
            return None
    return None

# --- TensorBoard Setup ---
def setup_tensorboard(log_dir, logger=None):
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M")
    log_dir = os.path.join(log_dir, f"{dt_string}_experiment")
    if logger:
      logger.info(f"TensorBoard logging to: {log_dir}")
    return SummaryWriter(log_dir=log_dir)

# --- Frame Stacking ---
def preprocess_frame(frames):
    if len(frames) == 0:
        return []
    processed_frames = []
    for frame in frames:
        if len(frame.shape) == 3 and frame.shape[0] == 3:
            frame = np.mean(frame, axis=0)
        frame = frame.astype(np.float32) / 255.0
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        if len(frame.shape) == 3:
            frame = frame.transpose(2, 0, 1)
        else:
            frame = np.expand_dims(frame, axis=0)
        processed_frames.append(frame)
    return processed_frames

def create_stacked_state(state_buffer):
    processed_frames = preprocess_frame(state_buffer)
    if any(f is None for f in processed_frames) or not all(f.shape == processed_frames[0].shape for f in processed_frames):
        print("Error: Inconsistent or None frames in processed_frames")
        return None
    stacked_state = np.concatenate(processed_frames, axis=0)
    return stacked_state

def get_game_state_info(game):
    game_state = game.get_state()
    if game_state is not None:
        damage_taken = game_state.game_variables[0]
        damage_inflicted = game_state.game_variables[1]
        return damage_taken, damage_inflicted
    else:
        return 0, 0

def process_game_step(game, action, frame_skip):
    """Processes a single step in the game environment."""
    reward = game.make_action(action, frame_skip)
    done = game.is_episode_finished()
    return reward, done

def update_agent(agent, state, action_index, reward, next_state, done, logger):
    """Updates the agent's knowledge based on the experience."""
    agent.n_step_buffer.push(state, action_index, reward, next_state, done)
    n_step_experience = agent.n_step_buffer.get()

    if n_step_experience:
        n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done = n_step_experience
        agent.memory.push(n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done)

    loss, avg_q_value, grad_norm = agent.learn()
    return loss, avg_q_value, grad_norm

def run_episode(agent, game, actions, episode, frame_skip, lmp_dir, record_video, video_writer, stack_size, state_shape, logger, episode_metrics_df):
    """Runs a single episode of the game."""
    if RECORD_LMP:
        lmp_file_path = os.path.join(lmp_dir, f"episode_{episode + 1}.lmp")
        game.new_episode(lmp_file_path)
    else:
        game.new_episode()

    game_state = game.get_state()
    if game_state is None or game_state.screen_buffer is None:
        logger.error(f"Skipping episode {episode+1} due to invalid game state.")
        return None, 0, 0, 0, 0, 0, None

    state_buffer = [game_state.screen_buffer] * stack_size
    state = create_stacked_state(state_buffer)
    if state is None:
        logger.error(f"Skipping episode {episode + 1} due to state initialization error.")
        return None, 0, 0, 0, 0, 0, None

    total_reward = 0
    step_count = 0
    episode_start_time = time.time()
    damage_taken = 0
    damage_inflicted = 0
    action_counts = Counter()
    inference_times = []
    loss, avg_q_value, grad_norm = None, None, None

    while not game.is_episode_finished():
        inference_start_time = time.time()
        action_index = agent.select_action(state)
        inference_end_time = time.time()
        inference_times.append(inference_end_time - inference_start_time)

        action = actions[action_index]
        action_counts[action_index] += 1

        reward, done = process_game_step(game, action, frame_skip)

        if not done:
            game_state = game.get_state()
            if game_state and game_state.screen_buffer is not None:
                next_frame = game_state.screen_buffer
                state_buffer.pop(0)
                state_buffer.append(next_frame)
                next_state = create_stacked_state(state_buffer)

                if next_state is None:
                    logger.error(f"Error: next_state is None in episode {episode + 1} at step {step_count + 1}.")
                    break

                damage_taken_step, damage_inflicted_step = get_game_state_info(game)
                damage_taken += damage_taken_step
                damage_inflicted += damage_inflicted_step
            else:
                logger.error(f"Error: Invalid state in episode {episode+1}, step {step_count+1}.")
                next_state = np.zeros(state_shape)
                break
        else:
            next_state = np.zeros(state_shape)

        loss, avg_q_value, grad_norm = update_agent(agent, state, action_index, reward, next_state, done, logger)

        state = next_state
        total_reward += reward
        step_count += 1

        if record_video and not done:
            most_recent_frame = state_buffer[-1]
            try:
                video_writer.append_data(most_recent_frame)
            except Exception as e:
                logger.error(f"Error appending frame to video: {e}")
                break

    episode_survival_time = time.time() - episode_start_time
    total_actions = sum(action_counts.values())
    action_diversity = sum(count / total_actions for count in action_counts.values()) / len(actions) if total_actions > 0 else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0

    if total_reward is not None:
        new_row = {
            'episode': episode + 1,
            'reward': total_reward,
            'steps': step_count,
            'survival_time': episode_survival_time,
            'damage_taken': damage_taken,
            'damage_inflicted': damage_inflicted,
            'action_diversity': action_diversity,
            'avg_inference_time': avg_inference_time,
            'loss': loss if loss is not None else np.nan,
            'avg_q_value': avg_q_value if avg_q_value is not None else np.nan,
            'grad_norm': grad_norm if grad_norm is not None else np.nan
        }
        return total_reward, step_count, episode_survival_time, damage_taken, damage_inflicted, action_diversity, avg_inference_time, new_row
    else:
        return None, 0, 0, 0, 0, 0, 0, None

# --- Validation ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # 'max' for reward maximization
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.improvement = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            self.improvement = True
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
            self.improvement = True
        else:
            self.counter += 1
            self.improvement = False
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

def validate_model(agent, game, actions, state_shape, stack_size, num_episodes, logger):
    """
    Validate the agent's performance over multiple episodes with minimal exploration.
    Returns mean reward and validation metrics dictionary.
    """
    # Store training state
    training_state = {
        'epsilon': agent.epsilon,
        'train': agent.policy_net.training
    }
    
    # Set validation state
    agent.epsilon = 0.05  # Minimal exploration during validation
    agent.policy_net.eval()  # Set network to evaluation mode
    
    validation_metrics = {
        'rewards': [],
        'steps': [],
        'survival_times': [],
        'damages_taken': [],
        'damages_inflicted': []
    }
    
    try:
        for episode in range(num_episodes):
            # Run validation episode
            result = run_episode(
                agent=agent,
                game=game,
                actions=actions,
                episode=episode,
                frame_skip=FRAME_SKIP_RECORDING,
                lmp_dir=None,
                record_video=False,
                video_writer=None,
                stack_size=stack_size,
                state_shape=state_shape,
                logger=logger,
                episode_metrics_df=pd.DataFrame()
            )
            
            if result[0] is not None:  # Check if episode was valid
                reward, steps, survival_time, damage_taken, damage_inflicted, _, _, _ = result
                validation_metrics['rewards'].append(reward)
                validation_metrics['steps'].append(steps)
                validation_metrics['survival_times'].append(survival_time)
                validation_metrics['damages_taken'].append(damage_taken)
                validation_metrics['damages_inflicted'].append(damage_inflicted)
        
        # Calculate summary statistics
        mean_metrics = {
            'mean_reward': np.mean(validation_metrics['rewards']),
            'mean_steps': np.mean(validation_metrics['steps']),
            'mean_survival': np.mean(validation_metrics['survival_times']),
            'mean_damage_taken': np.mean(validation_metrics['damages_taken']),
            'mean_damage_inflicted': np.mean(validation_metrics['damages_inflicted'])
        }
        
        logger.info("Validation Results:")
        for metric, value in mean_metrics.items():
            logger.info(f"{metric}: {value:.2f}")
            
        return mean_metrics['mean_reward'], validation_metrics, mean_metrics
        
    finally:
        # Restore training state
        agent.epsilon = training_state['epsilon']
        if training_state['train']:
            agent.policy_net.train()

def get_timestamp_prefix():
    """Get timestamp prefix in MMDDHHmm format"""
    return datetime.now().strftime("%m%d%H%M")

def save_model_with_timestamp(agent, model_dir, episode=None, is_best=False, logger=None):
    """Save model with timestamp prefix"""
    try:
        timestamp = get_timestamp_prefix()
        if is_best:
            filename = f"{timestamp}_best_model.pth"
        else:
            filename = f"{timestamp}_model_episode_{episode}.pth"

        model_path = os.path.join(model_dir, filename)
        agent.save_model(model_path)
        if logger:
            logger.info(f"Model saved: {filename}")
        return model_path
    except Exception as e:
        if logger:
            logger.error(f"Error saving model: {e}")
        return None

def best_model_callback(agent, episode_rewards, logger, model_dir, best_model_smoothing_window):
    """Callback to save the best model based on a smoothed average reward."""
    if len(episode_rewards) >= best_model_smoothing_window:
        avg_reward = np.mean(episode_rewards[-best_model_smoothing_window:])
    else:
        avg_reward = np.mean(episode_rewards) if episode_rewards else float('-inf')

    if avg_reward > agent.best_avg_reward:
        agent.best_avg_reward = avg_reward
        agent.best_model_state = deepcopy(agent.policy_net.state_dict())
        model_path = save_model_with_timestamp(agent, model_dir, is_best=True, logger=logger)
        if model_path:
            logger.info(f"Best model saved with reward: {agent.best_avg_reward}")

# --- Save metrics to CSV ---
def save_metrics(episode_rewards, episode_lengths, episode_survival_times,
                 episode_damage_taken, episode_damage_inflicted, logger, episode_metrics_df, training_time,
                 ram_usage_list, gpu_memory_usage_list, action_diversity_list):
    base_metrics = pd.DataFrame({
        'episode': range(1, len(episode_rewards) + 1),
        'reward': episode_rewards,
        'steps': episode_lengths,
        'epsilon': [EPSILON_START * (EPSILON_DECAY ** (i / EPSILON_DECAY_RATE_STEP)) for i in range(len(episode_rewards))],
        'training_time': training_time,
        'survival_time': episode_survival_times,
        'damage_taken': episode_damage_taken,
        'damage_inflicted': episode_damage_inflicted,
        'ram_usage': ram_usage_list,
        'gpu_memory_usage': gpu_memory_usage_list,
        'action_diversity': action_diversity_list
    })

    # Merge with episode metrics if they exist
    if not episode_metrics_df.empty:
        metrics_df = pd.merge(base_metrics, episode_metrics_df, on='episode', how='left')
    else:
        metrics_df = base_metrics

    # Add hyperparameters
    metrics_df['model_save_freq'] = MODEL_SAVE_FREQ
    metrics_df['batch_size'] = BATCH_SIZE
    metrics_df['memory_capacity'] = MEMORY_CAPACITY
    metrics_df['learning_rate'] = LEARNING_RATE
    metrics_df['tau'] = TAU
    metrics_df['n_step'] = N_STEP

    # Save to file
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M")
    metrics_filename = f"{dt_string}_training_metrics.csv"
    metrics_path = os.path.join(DRIVE_MODEL_DIR, metrics_filename)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Training metrics saved to {metrics_path}")

# --- Main Training Loop ---
def cleanup_resources():
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def close_writers(video_writer=None, tensorboard_writer=None):
    try:
        if video_writer:
            video_writer.close()  # Use close() instead of release() for imageio writer
        if tensorboard_writer:
            tensorboard_writer.close()
    except Exception as e:
        print(f"Error closing writers: {e}")

def validate_input(value, param_type, min_val, max_val, default):
    try:
        if param_type == "float":
            val = float(value)
        else:
            val = int(value)

        if min_val <= val <= max_val:
            return val
        print(f"Value out of range ({min_val}-{max_val}). Using default: {default}")
        return default
    except ValueError:
        print(f"Invalid input. Using default: {default}")
        return default

def get_hyperparameters():
    params = {
        "learning_rate": {"default": 0.00025, "type": "float", "min": 0, "max": 1,
                         "desc": "Step size for optimizer (0-1)"},
        "batch_size": {"default": 64, "type": "int", "min": 32, "max": 512,
                      "desc": "Batch size for training (32-512)"},
        "memory_capacity": {"default": 50000, "type": "int", "min": 10000, "max": 1000000,
                          "desc": "Max experiences in memory (10k-1M)"},
        "gamma": {"default": 0.99, "type": "float", "min": 0, "max": 1,
                 "desc": "Discount factor (0-1)"},
        "tau": {"default": 0.005, "type": "float", "min": 0, "max": 1,
                "desc": "Target network update rate (0-1)"},
        "epsilon_start": {"default": 1.0, "type": "float", "min": 0, "max": 1,
                         "desc": "Initial exploration rate (0-1)"},
        "epsilon_end": {"default": 0.05, "type": "float", "min": 0, "max": 1,
                       "desc": "Final exploration rate (0-1)"},
        "epsilon_decay": {"default": 0.995, "type": "float", "min": 0, "max": 1,
                         "desc": "Exploration decay rate (0-1)"},
        "epsilon_decay_rate_step": {"default": 5000, "type": "int", "min": 1000, "max": 100000,
                                   "desc": "Steps for epsilon decay (1k-100k)"},
        "n_step": {"default": 3, "type": "int", "min": 1, "max": 10,
                   "desc": "Steps for n-step learning (1-10)"},
        "grad_clip_norm": {"default": 1.0, "type": "float", "min": 0, "max": 10,
                          "desc": "Gradient clipping norm (0-10)"},
        "frame_stack_size": {"default": 4, "type": "int", "min": 2, "max": 4,
                            "desc": "Number of stacked frames (2-4)"}
    }

    print("\nHyperparameter Configuration Menu")
    print("=================================")

    for param_name, param_info in params.items():
        print(f"\n{param_name}: {param_info['desc']}")
        print(f"Default: {param_info['default']}")
        user_input = input(f"Enter value (or press Enter for default): ").strip()

        if user_input:
            params[param_name]["value"] = validate_input(
                user_input,
                param_info["type"],
                param_info["min"],
                param_info["max"],
                param_info["default"]
            )
        else:
            params[param_name]["value"] = param_info["default"]

    return params

def get_run_mode():
    print("\nSelect Run Mode:")
    print("1. Quick Run (Default settings)")
    print("2. Custom Run (Configure all settings)")
    while True:
        try:
            choice = int(input("Enter your choice (1 or 2): ").strip())
            if choice in [1, 2]:
                return choice
            print("Please enter 1 or 2")
        except ValueError:
            print("Please enter a valid number")

# Add new utility function for video path
def get_recording_path():
    """Get recording path with timestamp prefix"""
    timestamp = datetime.now().strftime("%m%d%H%M")
    filename = f"{timestamp}_game_recording.mp4"
    os.makedirs(VIDEO_DIR, exist_ok=True)  # Ensure directory exists
    return os.path.join(VIDEO_DIR, filename)

# Update quick_run_setup function to handle recording
def quick_run_setup(logger):
    """Quick run setup with model selection and recording option"""
    try:
        # First handle model choice
        print("\nModel Selection:")
        print("1. Create new model")
        print("2. Load existing model")
        while True:
            model_choice = input("Enter choice (1 or 2): ").strip()
            if model_choice in ['1', '2']:
                break
            print("Please enter 1 or 2")

        # Add recording choice
        print("\nRecording Option:")
        print("1. Record gameplay")
        print("2. No recording")
        while True:
            record_choice = input("Enter choice (1 or 2): ").strip()
            if record_choice in ['1', '2']:
                break
            print("Please enter 1 or 2")

        # Get number of episodes with input validation
        while True:
            try:
                episodes_input = input("Enter number of episodes (default=1): ").strip()
                episodes = int(episodes_input) if episodes_input else 1
                if episodes > 0:
                    break
                print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")

        # Get save frequency with input validation
        while True:
            try:
                save_freq_input = input("Enter model save frequency (default=1): ").strip()
                save_freq = int(save_freq_input) if save_freq_input else 1
                if save_freq > 0:
                    break
                print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")

        should_record = record_choice == '1'
        video_path = get_recording_path() if should_record else None

        config = default_config.copy()
        config.update({
            "num_episodes": episodes,
            "model_save_freq": save_freq,
            "record_video": should_record,
            "video_path": video_path,
            "frame_skip": FRAME_SKIP_RECORDING if should_record else FRAME_SKIP_TRAINING,
            "create_new_model": model_choice == '1'
        })

        logger.info(f"Quick run setup complete. Episodes: {episodes}, Save frequency: {save_freq}, Recording: {should_record}")
        if should_record:
            logger.info(f"Video will be saved to: {video_path}")
        return config

    except Exception as e:
        logger.error(f"Error in quick run setup: {e}")
        return default_config

def get_run_config(logger):
    """Get run configuration with proper logger"""
    config = {
        "scenario_path": None,
        "wad_path": None,
        "drive_model_dir": DRIVE_MODEL_DIR,
    }

    # Get WAD setup
    wad_choice = get_wad_choice(logger)
    config["scenario_path"], config["wad_path"] = setup_scenario(wad_choice, logger)

    # Get run mode
    run_mode = get_run_mode()

    if run_mode == 1:  # Quick Run
        quick_config = quick_run_setup(logger)
        config.update(quick_config)
    else:  # Custom Run
        params = get_hyperparameters()
        config.update({
            **params,
            "create_new_model": True  # Custom runs always create new model
        })

    return config

def main():
    # Initialize logger first
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir, timestamp)

    game = None
    video_writer = None
    tensorboard_writer = None

    try:
        # Get configuration with logger
        config = get_run_config(logger)

        if not config["scenario_path"]:
            logger.error("No scenario path specified")
            return

        # Initialize game
        game, actions, state_shape = setup_vizdoom(
            config["scenario_path"],
            config["wad_path"],
            logger
        )

        if not all([game, actions, state_shape]):
            logger.error("Failed to initialize ViZDoom environment")
            return

        # Initialize agent
        agent = DQNAgent(
            state_shape=state_shape,
            num_actions=len(actions),
            learning_rate=config.get("learning_rate", LEARNING_RATE),
            gamma=config.get("gamma", GAMMA),
            epsilon_start=config.get("epsilon_start", EPSILON_START),
            epsilon_end=config.get("epsilon_end", EPSILON_END),
            epsilon_decay=config.get("epsilon_decay", EPSILON_DECAY),
            memory_capacity=config.get("memory_capacity", MEMORY_CAPACITY),
            batch_size=config.get("batch_size", BATCH_SIZE),
            tau=config.get("tau", TAU),
            n_step=config.get("n_step", N_STEP),
            grad_clip_norm=config.get("grad_clip_norm", GRAD_CLIP_NORM),
            epsilon_decay_rate_step=config.get("epsilon_decay_rate_step", EPSILON_DECAY_RATE_STEP)
        )

        # Load existing model if requested
        if not config.get("create_new_model", True):
            agent = scan_for_models(agent, config["drive_model_dir"], logger)

        # Setup writers
        video_writer = None
        if config.get("record_video", False):
            video_path = config.get("video_path") or get_recording_path()
            video_writer = setup_video_writer(video_path, VIDEO_FPS, logger)
        tensorboard_writer = setup_tensorboard(log_dir, logger)

        # Initialize metrics tracking
        metrics_data = []  # Store episode metrics as dictionaries
        episode_rewards = []
        episode_lengths = []
        episode_survival_times = []
        episode_damage_taken = []
        episode_damage_inflicted = []
        ram_usage_list = []
        gpu_memory_usage_list = []
        action_diversity_list = []
        training_start_time = time.time()

        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.1,
            mode='max'
        )
        
        best_validation_reward = float('-inf')
        validation_history = []

        # Training loop
        for episode in range(config["num_episodes"]):
            try:
                # Run episode
                metrics = run_episode(
                    agent=agent,
                    game=game,
                    actions=actions,
                    episode=episode,
                    frame_skip=config["frame_skip"],
                    lmp_dir=LMP_DIR if RECORD_LMP else None,
                    record_video=config["record_video"],
                    video_writer=video_writer,
                    stack_size=STACK_SIZE,
                    state_shape=state_shape,
                    logger=logger,
                    episode_metrics_df=None  # Remove DataFrame dependency
                )

                if metrics:
                    reward, steps, survival_time, damage_taken, damage_inflicted, action_diversity, inference_time, new_row = metrics

                    # Update metrics lists
                    episode_rewards.append(reward)
                    episode_lengths.append(steps)
                    episode_survival_times.append(survival_time)
                    episode_damage_taken.append(damage_taken)
                    episode_damage_inflicted.append(damage_inflicted)
                    action_diversity_list.append(action_diversity)

                    # Resource monitoring
                    ram_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    ram_usage_list.append(ram_usage)

                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                        gpu_memory_usage_list.append(gpu_memory)
                    else:
                        gpu_memory_usage_list.append(0)

                    # Store metrics dictionary
                    if new_row:
                        metrics_data.append(new_row)

                # Save model if needed
                if (episode + 1) % config["model_save_freq"] == 0:
                    save_model_with_timestamp(agent, config["drive_model_dir"], episode=episode+1, logger=logger)

                # Save best model if enabled
                if USE_BEST_MODEL_CALLBACK:
                    best_model_callback(agent, episode_rewards, logger, config["drive_model_dir"], BEST_MODEL_SMOOTHING_WINDOW)

                # Periodic validation
                if (episode + 1) % MODEL_SAVE_FREQ == 0:
                    mean_reward, val_metrics, mean_metrics = validate_model(
                        agent=agent,
                        game=game,
                        actions=actions,
                        state_shape=state_shape,
                        stack_size=STACK_SIZE,
                        num_episodes=VALIDATION_EPISODES,
                        logger=logger
                    )
                    
                    validation_history.append(mean_metrics)
                    
                    # Update best model if improved
                    if mean_reward > best_validation_reward:
                        best_validation_reward = mean_reward
                        save_model_with_timestamp(agent, config["drive_model_dir"], episode=episode+1, is_best=True, logger=logger)
                    
                    # Check early stopping
                    if early_stopping(mean_reward):
                        logger.info("Early stopping triggered!")
                        break
                    elif early_stopping.improvement:
                        logger.info(f"Validation improved! New best reward: {mean_reward:.2f}")

                # Log progress
                logger.info(f"Episode {episode+1}/{config['num_episodes']}: "
                          f"Reward={reward:.2f}, Steps={steps}, "
                          f"Epsilon={agent.epsilon:.3f}")

                # Cleanup after episode
                cleanup_resources()

            except Exception as e:
                logger.error(f"Error in episode {episode}: {e}")
                continue

        # Create final metrics DataFrame
        episode_metrics_df = pd.DataFrame(metrics_data)

        # Save final metrics
        training_time = time.time() - training_start_time
        save_metrics(
            episode_rewards, episode_lengths, episode_survival_times,
            episode_damage_taken, episode_damage_inflicted, logger,
            episode_metrics_df, training_time, ram_usage_list,
            gpu_memory_usage_list, action_diversity_list
        )

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise e

    finally:
        # Cleanup
        try:
            close_writers(video_writer, tensorboard_writer)
            if game:
                game.close()
            cleanup_resources()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()

{
  "type": "setting",
  "settings": {
    "python.logging.level": "INFO"
  }
}