import unittest
import numpy as np
import torch
import os
import yaml
from unittest.mock import Mock, patch

from VIZDOOMMAIN import (
    DQN, 
    PrioritizedReplayMemory,
    NStepBuffer,
    DQNAgent,
    load_config,
    preprocess_frame,
    create_stacked_state
)

class TestViZDoomMain(unittest.TestCase):
    def setUp(self):
        self.test_config = {
            "learning_rate": 0.0001,
            "batch_size": 32,
            "memory_capacity": 1000,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.995,
            "tau": 0.005,
            "n_step": 3,
            "grad_clip_norm": 1.0,
            "epsilon_decay_rate_step": 1000,
        }
        
        # Create temporary config file
        with open("test_config.yaml", "w") as f:
            yaml.dump(self.test_config, f)
            
        self.state_shape = (4, 84, 84)  # (stack_size * channels, height, width)
        self.num_actions = 3
        
    def tearDown(self):
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")

    def test_config_loading(self):
        """Test configuration loading functionality"""
        config = load_config("test_config.yaml")
        self.assertEqual(config["learning_rate"], self.test_config["learning_rate"])
        self.assertEqual(config["batch_size"], self.test_config["batch_size"])

    def test_dqn_architecture(self):
        """Test DQN model architecture"""
        model = DQN(self.state_shape, self.num_actions)
        
        # Test model structure
        self.assertIsInstance(model.conv1, torch.nn.Conv2d)
        self.assertIsInstance(model.conv2, torch.nn.Conv2d)
        self.assertIsInstance(model.conv3, torch.nn.Conv2d)
        self.assertIsInstance(model.fc1, torch.nn.Linear)
        self.assertIsInstance(model.fc2, torch.nn.Linear)
        
        # Test forward pass
        batch_size = 1
        x = torch.randn(batch_size, *self.state_shape)
        output = model(x)
        self.assertEqual(output.shape, (batch_size, self.num_actions))

    def test_prioritized_replay_memory(self):
        """Test PrioritizedReplayMemory functionality"""
        memory = PrioritizedReplayMemory(capacity=100)
        
        # Test pushing experience
        state = np.random.rand(4, 84, 84)
        next_state = np.random.rand(4, 84, 84)
        memory.push(state, 1, 1.0, next_state, False)
        
        self.assertEqual(len(memory), 1)
        
        # Test sampling
        for _ in range(10):  # Add more experiences
            memory.push(state, 1, 1.0, next_state, False)
        
        batch_size = 4
        samples, indices, weights = memory.sample(batch_size)
        
        self.assertEqual(len(samples), batch_size)
        self.assertEqual(len(indices), batch_size)
        self.assertEqual(len(weights), batch_size)

    def test_nstep_buffer(self):
        """Test NStepBuffer functionality"""
        n_step = 3
        buffer = NStepBuffer(n_step)
        
        # Test pushing experience
        state = np.random.rand(4, 84, 84)
        next_state = np.random.rand(4, 84, 84)
        
        # Push n-1 experiences (should return None)
        for _ in range(n_step-1):
            buffer.push(state, 1, 1.0, next_state, False)
            self.assertIsNone(buffer.get())
            
        # Push nth experience (should return first experience)
        buffer.push(state, 1, 1.0, next_state, False)
        experience = buffer.get()
        self.assertIsNotNone(experience)

    def test_dqn_agent(self):
        """Test DQNAgent core functionality"""
        agent = DQNAgent(
            state_shape=self.state_shape,
            num_actions=self.num_actions,
            learning_rate=0.0001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            memory_capacity=1000,
            batch_size=32,
            tau=0.005,
            n_step=3,
            grad_clip_norm=1.0,
            epsilon_decay_rate_step=1000
        )
        
        # Test action selection
        state = np.random.rand(*self.state_shape)
        action = agent.select_action(state)
        self.assertTrue(0 <= action < self.num_actions)
        
        # Test epsilon update
        initial_epsilon = agent.epsilon
        agent.update_epsilon()
        self.assertLess(agent.epsilon, initial_epsilon)
        
        # Test model saving and loading
        agent.save_model("test_model.pth")
        agent.load_model("test_model.pth")
        os.remove("test_model.pth")

    def test_frame_preprocessing(self):
        """Test frame preprocessing functions"""
        # Test single frame
        frame = np.random.rand(3, 240, 320)  # RGB frame
        processed_frames = preprocess_frame([frame])
        self.assertEqual(len(processed_frames), 1)
        self.assertEqual(processed_frames[0].shape, (1, 84, 84))
        
        # Test frame stacking
        frames = [np.random.rand(3, 240, 320) for _ in range(4)]
        stacked_state = create_stacked_state(frames)
        self.assertEqual(stacked_state.shape, (4, 84, 84))

    @patch('vizdoom.DoomGame')
    def test_game_integration(self, mock_doom_game):
        """Test game integration with mocked ViZDoom"""
        mock_game = Mock()
        mock_doom_game.return_value = mock_game
        
        # Mock game state
        mock_state = Mock()
        mock_state.screen_buffer = np.random.rand(3, 240, 320)
        mock_state.game_variables = [0, 0]
        mock_game.get_state.return_value = mock_state
        
        # Set up game responses
        mock_game.is_episode_finished.return_value = False
        mock_game.make_action.return_value = 1.0
        
        # Test game initialization
        mock_game.init.return_value = True
        self.assertTrue(mock_game.init())
        
        # Test game step
        reward = mock_game.make_action([1, 0, 0], 4)
        self.assertEqual(reward, 1.0)
        
        # Test game state
        state = mock_game.get_state()
        self.assertIsNotNone(state.screen_buffer)

if __name__ == '__main__':
    unittest.main()