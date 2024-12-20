"""
store all the agents here
"""
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import time
import pickle
from collections import deque
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from tensorflow.keras.losses import Huber

def huber_loss(y_true, y_pred, delta=1):
    """Keras implementation for huber loss
    loss = {
        0.5 * (y_true - y_pred)**2 if abs(y_true - y_pred) < delta
        delta * (abs(y_true - y_pred) - 0.5 * delta) otherwise
    }
    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        loss values for all points
    """
    error = y_true - y_pred
    quad_error = 0.5*torch.square(error)
    lin_error = delta*(torch.abs(error) - 0.5*delta)
    # quadratic error, linear error
    return torch.where(torch.abs(error) < delta, quad_error, lin_error)

def mean_huber_loss(y_true, y_pred, delta=1):
    """Calculates the mean value of huber loss

    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        average loss across points
    """
    return torch.mean(huber_loss(y_true, y_pred, delta))

class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    HamiltonianCycleAgent
    BreadthFirstSearchAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2).reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col

class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values
    This class extends to the following classes
    PolicyGradientAgent
    AdvantageActorCriticAgent

    Attributes
    ----------
    _model : TensorFlow Graph
        Stores the graph of the DQN model
    _target_net : TensorFlow Graph
        Stores the target network graph of the DQN model
    """
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(self.device)
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        self.reset_models()

    def reset_models(self):
        self._model = self._agent_model(self._board_size, self._n_frames, self._n_actions, self._version).to(self.device)
        print(f'{self._model=}')
        self._actor_optimizer = optim.RMSprop(self._model.parameters(), lr=0.0005)

        if self._use_target_net:
            self._target_net = self._agent_model(self._board_size, self._n_frames, self._n_actions, self._version).to(self.device)
            self.update_target_net()


    def _prepare_input(self, board):
        """Reshape input and normalize
        
        Parameters
        ----------
        board : Numpy array
            The board state to process

        Returns
        -------
        board : Numpy array
            Processed and normalized board
        """
        if(board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy())
        #board = np.transpose(board, (0, 3, 1, 2)) 
        #return torch.tensor(board, dtype=torch.float32).to(self.device)
        return board.copy()

    def _get_model_outputs(self, board, model=None):
        """Get action values from the DQN model

        Parameters
        ----------
        board : Numpy array
            The board state for which to predict action values
        model : TensorFlow Graph, optional
            The graph to use for prediction, model or target network

        Returns
        -------
        model_outputs : Numpy array
            Predicted model outputs on board, 
            of shape board.shape[0] * num actions
        """
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        board_tensor = torch.tensor(board, dtype=torch.float32).to(self.device)
        # the default model to use
        if model is None:
            model = self._model
        model.eval()
        with torch.no_grad():
            model_outputs = model(board_tensor).cpu().numpy()
        return model_outputs

    def _normalize_board(self, board):
        """Normalize the board before input to the network
        
        Parameters
        ----------
        board : Numpy array
            The board state to normalize

        Returns
        -------
        board : Numpy array
            The copy of board state after normalization
        """
        # return board.copy()
        # return((board/128.0 - 1).copy())
        return board.astype(np.float32) / 4.0

    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value
        
        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        """
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._model)
        return np.argmax(np.where(legal_moves==1, model_outputs, -np.inf), axis=1)

    def _agent_model_non_working(self, board_size, n_frames, n_actions, version):
        """Returns the model which evaluates Q values for a given state input

        Returns
        -------
        model : TensorFlow Graph
            DQN model graph
        """
        class DQNModel(nn.Module):
            def __init__(self, board_size, n_frames, n_actions, version):
            #def __init__(self, board_size, n_frames, n_actions):
                super(DQNModel, self).__init__()

                with open('model_config/{:s}.json'.format(version), 'r') as f:
                    model_config = json.load(f)['model']

                layers = []
                in_channels = n_frames
                #self.flattened_size = None  # Placeholder for the dynamically computed flattened size
                for layer_name, layer_config in model_config.items():
                    if 'Conv2D' in layer_name:
                        layers.append(nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=layer_config['filters'],
                            kernel_size=tuple(layer_config['kernel_size']),
                            stride=layer_config.get('stride', 1),
                            padding=layer_config.get('padding') if layer_config.get('padding') else 0
                        ))
                        if layer_config['activation'] == 'relu':
                            layers.append(nn.ReLU())
                        elif layer_config['activation'] == 'sigmoid':
                            layers.append(nn.Sigmoid())
                            
                        in_channels = layer_config['filters']
                    elif 'Flatten' in layer_name:
                        layers.append(nn.Flatten())
                        pass
                    elif 'Dense' in layer_name:
                        if len(layers) == 0:
                            in_features = n_frames*board_size*board_size
                        elif isinstance(layers[-1], nn.Flatten):
                            in_features = in_channels*(board_size-6)*(board_size-6)
                        else:
                            in_features = out_features
                        
                        layers.append(nn.Linear(
                                in_features=in_features,
                                out_features=layer_config['units']
                                ))
                        out_features = layer_config['units']
                        
                        if layer_config['activation'] == 'relu':
                            layers.append(nn.ReLU())
                        elif layer_config['activation'] == 'sigmoid':
                            layers.append(nn.Sigmoid())
                        else:
                            raise NotImplementedError(
                            f'Activation function not implemented :{layer_config["activation"]}'
                            )
                            
               
                layers.append(nn.Linear(out_features, n_actions))  # Final layer
               
                    
                print(layers)
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                x = x.permute(0, 3, 1, 2)
                return self.model(x)
            
                
        """
        input_board = Input((self._board_size, self._board_size, self._n_frames,), name='input')
        x = Conv2D(16, (3,3), activation='relu', data_format='channels_last')(input_board)
        x = Conv2D(32, (3,3), activation='relu', data_format='channels_last')(x)
        x = Conv2D(64, (6,6), activation='relu', data_format='channels_last')(x)
        x = Flatten()(x)
        x = Dense(64, activation = 'relu', name='action_prev_dense')(x)
        # this layer contains the final output values, activation is linear since
        # the loss used is huber or mse
        out = Dense(self._n_actions, activation='linear', name='action_values')(x)
        # compile the model
        model = Model(inputs=input_board, outputs=out)
        model.compile(optimizer=RMSprop(0.0005), loss=mean_huber_loss)
        # model.compile(optimizer=RMSprop(0.0005), loss='mean_squared_error')
        """

        return DQNModel(board_size, n_frames, n_actions, version)
        #return DQNModel(self.board_size, self.n_frames, self._n_actions)

    def _agent_model(self, board_size, n_frames, n_actions, version):
        """Returns the model which evaluates Q values for a given state input

        Returns
        -------
        model : TensorFlow Graph
            DQN model graph
        """
        class DQNModel(nn.Module):
            """PyTorch implementation of the DQN model
            Same architecture as the TensorFlow version
            """
            def __init__(self, board_size, frames, n_actions, version):
                super(DQNModel, self).__init__()
                self.board_size = board_size
                self.frames = frames
                self.n_actions = n_actions
                
                # model config
                with open(f'model_config/{version}.json', 'r') as f:
                    m = json.loads(f.read())
                    
                # 1st conv layer
                self.conv1 = nn.Conv2d(
                    in_channels=frames,
                    out_channels=16,
                    kernel_size=3,
                    padding='same'
                )
                
                # 2nd conv layer
                self.conv2 = nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3
                )
                
                # 3rd conv layer (64 filters, 5x5, valid padding)
                self.conv3 = nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=5
                )

                # calc size after convolutions
                # After conv2 (valid padding): size = board_size - 2
                # After conv3 (valid padding): size = board_size - 2 - 4        
                conv_out_size = board_size - 6  # total reduction in size
                conv_out_size = 64 * conv_out_size * conv_out_size
                
                # dense layers
                self.fc1 = nn.Linear(conv_out_size, 64)
                self.fc2 = nn.Linear(64, n_actions)
                
            def forward(self, x):
                """Forward pass of the network
                
                Parameters
                ----------
                x : torch.Tensor
                    Input tensor in NHWC format (TensorFlow style)
                    
                Returns
                -------
                torch.Tensor
                    Output tensor with action values
                """
                # convert NHWC (TensorFlow) to NCHW (PyTorch) format
                x = x.permute(0, 3, 1, 2)
                
                # conv layers with ReLU
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                
                # flatten
                x = x.reshape(x.size(0), -1)
                
                # dense layers
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                
                return x
        
        return DQNModel(board_size, n_frames, n_actions, version)

    def set_weights_trainable(self):
        """Set selected layers to non trainable and compile the model"""
        for layer in self._model.layers:
            layer.trainable = False
        # the last dense layers should be trainable
        for s in ['action_prev_dense', 'action_values']:
            self._model.get_layer(s).trainable = True
        self._model.compile(optimizer = self._model.optimizer, 
                            loss = self._model.loss)


    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes
        
        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1).reshape((-1,1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        """Save the current models to disk using tensorflow's
        inbuilt save model function (saves in h5 format)
        saving weights instead of model as cannot load compiled
        model with any kind of custom object (loss or metric)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        torch.save(self._model.state_dict(), f"{file_path}/model_{iteration:04d}.pth")
        if self._use_target_net:
            torch.save(self._target_net.state_dict(), f"{file_path}/model_{iteration:04d}_target.pth")

    def load_model(self, file_path='', iteration=None):
        """ load any existing models, if available """
        """Load models from disk using tensorflow's
        inbuilt load model function (model saved in h5 format)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """

        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}.pth"))
        if self._use_target_net:
            self._target_net.load_state_dict(torch.load(f"{file_path}/model_{iteration:04d}_target.pth"))
        # print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        """Print the current models using summary method"""
        print('Training Model')
        print(self._model)
        if(self._use_target_net):
            print('Target Network')
            print(self._target_net)

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error.
        We are predicting the expected future discounted reward for all
        actions with our model. The target for training the model is calculated
        in two parts:
        1) dicounted reward = current reward + 
                        (max possible reward in next state) * gamma
           the next reward component is calculated using the predictions
           of the target network (for stability)
        2) rewards for only the action take are compared, hence while
           calculating the target, set target value for all other actions
           the same as the model predictions
        
        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Whether to clip the rewards using the numpy sign command
            rewards > 0 -> 1, rewards <0 -> -1, rewards == 0 remain same
            this setting can alter the learned behaviour of the agent

        Returns
        -------
            loss : float
            The current error (error metric is defined in reset_models)
        """
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, done, legal_moves = self._buffer.sample(batch_size)

        # Reward clipping
        if reward_clip:
            rewards = np.sign(rewards)

        # Convert data to PyTorch tensors
        states = torch.FloatTensor(self._normalize_board(states)).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)  # Make rewards shape (batch_size, 1)
        next_states = torch.FloatTensor(self._normalize_board(next_states)).to(self.device)
        done = torch.FloatTensor(done).to(self.device)#.unsqueeze(1)  # Shape (batch_size, 1)
        legal_moves = torch.FloatTensor(legal_moves).to(self.device)

        """ # Compute Q-values for the current states
        q_values = self._model(states)

        # Select Q-values corresponding to the taken actions
        q_values = torch.sum(q_values * actions, dim=1, keepdim=True)  # Shape (batch_size, 1)

        # Compute Q-values for the next states
        with torch.no_grad():
            target_model = self._target_net if self._use_target_net else self._model
            next_q_values = target_model(next_states)

            # Mask illegal moves by setting them to -inf, then compute max
            next_q_values[legal_moves == 0] = -float('inf')
            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]

            # Compute the discounted reward
            discounted_reward = rewards + self._gamma * max_next_q_values * (1 - done)

        # Prepare target Q-values
        self._model.train()
        with torch.no_grad():
            targets = self._model(states)
            targets = targets * (1 - actions) + discounted_reward * actions """
        
        # target values
        with torch.no_grad():
            if self._use_target_net:
                next_model_outputs = self._target_net(next_states)
            else:
                next_model_outputs = self._model(next_states)
                
            # Get max Q-value for next state considering only legal moves
            max_next_q = torch.max(
                torch.where(legal_moves == 1, next_model_outputs, torch.tensor(-float('inf')).to(self.device)),
                dim=1
            )[0].reshape(-1, 1)
            
            # target Q-values
            discounted_reward = rewards + (self._gamma * max_next_q * (1 - done))
        
        # current Q-values and create target
        self._model.train()
        current_q = self._model(states)
        target = current_q.clone()
        target = (1 - actions) * target + actions * discounted_reward

        # Compute the loss (Huber loss)
        loss = mean_huber_loss(target, current_q)

        # Backpropagation
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        return loss.item()

    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if(self._use_target_net):
            self._target_net.load_state_dict(self._model.state_dict())

    def compare_weights(self):
        """Simple utility function to heck if the model and target 
        network have the same weights or not
        """
        for i in range(len(self._model.layers)):
            for j in range(len(self._model.layers[i].weights)):
                c = (self._model.layers[i].weights[j].numpy() == \
                     self._target_net.layers[i].weights[j].numpy()).all()
                print('Layer {:d} Weights {:d} Match : {:d}'.format(i, j, int(c)))

    def copy_weights_from_agent(self, agent_for_copy):
        """Update weights between competing agents which can be used
        in parallel training
        """
        assert isinstance(agent_for_copy, self), "Agent type is required for copy"

        self._model.set_weights(agent_for_copy._model.get_weights())
        self._target_net.set_weights(agent_for_copy._model_pred.get_weights())
