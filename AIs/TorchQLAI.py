from random import choice, random

import torch
import torch.nn as nn

from godot import exposed, export
from godot.bindings import *
from godot.globals import *

from AIs.QLAI import QLAI
from AIs.structs import Experience
import util


class DQN(nn.Module):
	def __init__(self, arch, learning_rate, weight_decay, model_params=None):
		super(DQN, self).__init__()
		self.layers = len(arch) - 1
		modules = []
		for i in range(self.layers):
			modules.append(nn.Linear(arch[i], arch[i+1], bias=False))
		self.model = nn.ModuleList(modules)
		self.relu = nn.Tanh()
		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,
										  weight_decay=weight_decay)
		if not (model_params is None):
			self.model.load_state_dict(model_params)
    
	def forward(self, features):
		output = features
		for i in range(self.layers-1):
			output = self.relu(self.model[i](output))
		output = self.model[self.layers-1](output)
		return output
 
	def back(self, predicted, label):
		self.optimizer.zero_grad()
		loss = self.criterion(predicted, label)
		# normalized_loss = torch.tanh(loss)
		loss.backward()
		self.optimizer.step()
		return loss

	def predict(self, features):
		return torch.argmax(self.forward(features)).item()

@exposed
class TorchQLAI(QLAI):
	"""
	Q-Learning AI implemented using PyTorch.

	This implementation uses a neural network to aproximate the policy
	function, and uses gradient descent to update its weights.

	This implementation has 4 hyperparams:
		- Discount: Discounts past states' values
		- Exploring rate: Chance that the agent executes a random action
		- Learning rate: The rate that the agent learns
		- Exploring rate decay: Decays the exploring rate each cycle

	Some good presets:
	* Ideal: [[[0.273222,0.218746,-0.238173,0.097687,-0.002475,0.158756,-0.206196,0.012181,0.136202]],[-0.216642]]
	"""
	def _ready(self):
		super(TorchQLAI, self)._ready()
	
	def init(self, params):
		super(TorchQLAI, self).init(params)
		self.network_key = None
		model_params = None
		if not (params["network_id"] is None):
			self.character_id = params["character_id"]
			self.network_id = params["network_id"]
			self.network_key = f"{self.character_id}_TorchQLAI_{self.network_id}"
			model_params = NNSaveManager.get_params(self.network_key)
		self.learning_model = DQN([self.features_size, 16, 1], self.alpha,
								  0.01, model_params=model_params)
	
	def end(self):
		if not (self.network_key is None):
			NNSaveManager.set_params(self.network_key, self.learning_model.state_dict())

	def get_info(self):
		# TODO: Use state_dict method
		return util.py2gdArray([param.tolist() for param in self.learning_model.parameters()])

	def _torch_get_q_value(self, state, action):
		features = self.get_features_after_action(state, action)
		return self.learning_model.forward(features)[0]

	def get_q_value(self, state, action):
		return self._torch_get_q_value(state, action).item()

	def reset(self, timeout):
		super(TorchQLAI, self).reset(timeout)
		if self.use_experience_replay:
			exp_sample = self.ep.sample()
			if not (exp_sample is None):
				loss = self._update_weights_experience(exp_sample)
				self.logger.push("loss", loss.item())
	
	def _update_weights_experience(self, exp_sample):
		actual_val_vec = []
		next_val_vec = []
		reward_vec = []

		for exp in exp_sample:
			actual_val_vec.append(self.learning_model.forward(exp.features)[0])

			next_val = self.compute_value_from_q_values(exp.next_state)
			next_val_vec.append(next_val)

			reward_vec.append(exp.reward)

		actual_val_vec = torch.stack(actual_val_vec)
		next_val_vec = torch.stack(next_val_vec)
		reward_vec = torch.tensor(reward_vec)
		lable_vec = reward_vec + self.discount * next_val_vec

		return self.learning_model.back(actual_val_vec, lable_vec)

	def update_weights(self, state, action, next_state, reward, last):
		features = self.get_features_after_action(state, action)
		experience = Experience(features, reward, None if last else next_state)
		self.ep.add(experience)
		exp_sample = self.ep.simple_sample()
		self._update_weights_experience(exp_sample)
		self.logger.push("reward", reward)

	def compute_value_from_q_values(self, state):
		if state is None:
			return torch.tensor(0.0)
		legal_actions = self.parent.get_legal_actions(state)
		features_list = torch.stack([self.get_features_after_action(state, a) for a in legal_actions])
		return torch.max(self.learning_model.forward(features_list))

	def compute_action_from_q_values(self, state):
		legal_actions = self.parent.get_legal_actions(state)
		if random() < self.epsilon:
			return choice(legal_actions)
		features_list = torch.stack([self.get_features_after_action(state, a) for a in legal_actions])
		q_vals = self.learning_model.forward(features_list)
		prediction = torch.argmax(q_vals).item()
		return legal_actions[prediction]

	# Print some variables for debug here
	def _on_DebugTimer_timeout(self):
		super(TorchQLAI, self)._on_DebugTimer_timeout()
		print("------ TorchQLAI ------")
		stats = ["max", "min", "avg"]
		self.logger.print_stats("update_state", stats)
		# self.logger.print_stats("max_q_val", stats)
		# self.logger.print_stats("reward", stats)
		self.logger.flush("update_state")
		# self.logger.flush("max_q_val")
		# self.logger.flush("reward")
		# print("Max weight: ", util.apply_list_func(self.get_info(), max))
		# print("Min weight: ", util.apply_list_func(self.get_info(), min))
		# print("epsilon: {}".format(self.epsilon))
		# print(self.get_info())
