from random import choice, random

import torch
import torch.nn as nn

from godot import exposed, export
from godot.bindings import *
from godot.globals import *

from Characters.AIs.QLAI import QLAI
from Characters.AIs.structs import Experience
import util


class CDN(nn.Module):
	def __init__(self, arch, learning_rate, weight_decay):
		super(CDN, self).__init__()
		self.layers = len(arch) - 1
		modules = []
		for i in range(self.layers):
			modules.append(nn.Linear(arch[i], arch[i+1], bias=False))
		self.model = nn.ModuleList(modules)
		self.tanh = nn.Tanh()
		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,
										  weight_decay=weight_decay)
    
	def forward(self, features):
		output = features
		for i in range(self.layers-1):
			output = self.tanh(self.model[i](output))
		output = self.model[self.layers-1](output)
		return output

	def back(self, predicted, label):
		self.optimizer.zero_grad()
		loss = self.criterion(predicted, label)
		# normalized_loss = torch.tanh(loss)
		loss.backward()
		self.optimizer.step()
		return loss

@exposed
class ClassQLAI(QLAI):
	"""
	Q-Learning AI implemented using PyTorch.

	This implementation uses a classifier neural network to aproximate the policy
	function of all states at once, and uses an adaptative gradient descent to
	update its weights.
	"""
	def _ready(self):
		super(ClassQLAI, self)._ready()
		self.learning_model = CDN([self.features_size, 12, 12, Action._get_size()], self.alpha, 0.01)
	
	def get_info(self):
		# TODO: Use state_dict method
		return util.py2gdArray([param.tolist() for param in self.learning_model.parameters()])

	def _torch_get_q_value(self, state, action_id):
		features = self.get_features_after_action(state, Action.IDLE)
		q_values = self.learning_model.forward(features)
		return q_values[action_id]

	def get_q_value(self, state, action):
		action_id = Action._get_action_id(action)
		return self._torch_get_q_value(state, action_id).item()
	
	def reset(self, timeout):
		super(ClassQLAI, self).reset(timeout)
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
			q_values = self.learning_model.forward(exp.features)
			actual_val_vec.append(q_values[exp.action])

			next_val = self.compute_value_from_q_values(exp.next_state)
			next_val_vec.append(next_val)

			reward_vec.append(exp.reward)
		actual_val_vec = torch.stack(actual_val_vec)
		next_val_vec = torch.stack(next_val_vec)
		reward_vec = torch.tensor(reward_vec)
		lable_vec = reward_vec + self.discount * next_val_vec
		return self.learning_model.back(actual_val_vec, lable_vec)

	def update_weights(self, state, action, next_state, reward, last):
		features = self.get_features_after_action(state, Action.IDLE)
		action_id = Action._get_action_id(action)
		experience = Experience(features, reward, None if last else next_state, action_id)
		self.ep.add(experience)
		exp_sample = self.ep.simple_sample()
		self._update_weights_experience(exp_sample)
		self.logger.push("reward", reward)

	def compute_value_from_q_values(self, state):
		if state is None:
			return torch.tensor(0.0)
		legal_actions_ids = [Action._get_action_id(a) for a in self.parent.get_legal_actions(state)]
		return torch.max(self._torch_get_q_value(state, legal_actions_ids))

	def compute_action_from_q_values(self, state):
		legal_actions = self.parent.get_legal_actions(state)
		if random() < self.epsilon:
			return choice(legal_actions)
		legal_actions_ids = [Action._get_action_id(a) for a in legal_actions]
		prediction = self._torch_get_q_value(state, legal_actions_ids)
		best_action_id = legal_actions_ids[torch.argmax(prediction)]
		return Action._get_action_from_id(best_action_id)

	# Print some variables for debug here
	def _on_DebugTimer_timeout(self):
		super(ClassQLAI, self)._on_DebugTimer_timeout()
		print("------ ClassQLAI ------")
		stats = ["max", "min", "avg"]
		# self.logger.print_stats("update_state", stats)
		# self.logger.print_stats("max_q_val", stats)
		self.logger.print_stats("reward", stats)
		# self.logger.flush("update_state")
		# self.logger.flush("max_q_val")
		self.logger.flush("reward")
		# print("Max weight: ", util.apply_list_func(self.get_info(), max))
		# print("Min weight: ", util.apply_list_func(self.get_info(), min))
		# print("epsilon: {}".format(self.epsilon))
		# print(self.get_info())
