from random import choice, random
import math

import torch

from godot import exposed, export
from godot.bindings import *
from godot.globals import *

from AIs.QLAI import QLAI
import util

@exposed
class BerkeleyQLAI(QLAI):
	"""
	Q-Learning AI implemented using UC Berkeley CS188 course's material
	(http://ai.berkeley.edu/home.html).

	The material uses an one-dimentional matrix of weights to approximate
	the policy function, and uses Belman's equation to update them.

	This implementation has 5 hyperparams:
		- Discount: Discounts past states' values
		- Exploring rate: Chance that the agent executes a random action
		- Learning rate: The rate that the agent learns
		- Exploring rate decay: Decays the exploring rate each cycle
		- Reuse last action chance: If the agent executes a random action
			this is the chance that the result action is the same as the
			last action performed (for smoothness)
	"""
	def _ready(self):
		super(BerkeleyQLAI, self)._ready()
		self.learning_weights = 2.0 * torch.rand(self.features_size) + 1.0
	
	def get_info(self):
		# TODO: Use state_dict method
		return util.py2gdArray(self.learning_weights.tolist())

	def get_q_value(self, state, action):
		features = self.get_features_after_action(state, action)
		return features @ self.learning_weights

	def update_weights(self, state, action, next_state, reward, last):
		next_state_action_value = self.compute_value_from_q_values(next_state)
		target = reward
		if not last:
			target += self.discount * next_state_action_value
		prediction = self.get_q_value(state, action)
		correction = target - prediction
		features = self.get_features_after_action(state, action)

		self.learning_weights += self.alpha * correction * features
		self.learning_weights /= self.learning_weights.norm()

		self.logger.push("loss", math.fabs(correction))
		self.logger.push("max_q_val", next_state_action_value)
		self.logger.push("reward", reward)

	# Print some variables for debug here
	def _on_DebugTimer_timeout(self):
		super(BerkeleyQLAI, self)._on_DebugTimer_timeout()
		print("------ BerkeleyQLAI ------")
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