from random import choice, random
import math

import torch

from godot import exposed, export
from godot.bindings import *
from godot.globals import *

from Characters.AIs.QLAI import QLAI
import util

@exposed
class PerceptronQLAI(QLAI):
	"""
	Q-Learning AI implemented using UC Berkeley CS188 course's material
	(http://ai.berkeley.edu/home.html).

	The material uses an one-dimensional matrix of weights to approximate
	the policy function, and uses Belman's equation to update them.
	"""
	def _ready(self):
		super(PerceptronQLAI, self)._ready()
	
	def init(self, params):
		super(PerceptronQLAI, self).init(params)
		if not (params["network_id"] is None):
			character_type = params["character_type"]
			network_id = params["network_id"]
			self.network_key = f"{character_type}_PerceptronQLAI_{network_id}"
		persisted_params = self.load_params()
		self.learning_weights = 2.0 * torch.rand(self.features_size) + 1.0
		if not (persisted_params is None):
			self.learning_weights = persisted_params.get("model_params")
			self.time = persisted_params.get("time")

	def end(self):
		persistence_dict = {
			"time": self.time,
			"model_params": self.learning_weights
		}
		self.save_params(persistence_dict)

	def get_info(self):
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
		features = self.get_features(next_state)

		self.learning_weights += self.alpha * correction * features
		self.learning_weights /= self.learning_weights.norm()

		self.logger.push("loss", math.fabs(correction))
		self.logger.push("max_q_val", next_state_action_value)
		self.logger.push("reward", reward)

	# Print some variables for debug here
	def _on_DebugTimer_timeout(self):
		super(PerceptronQLAI, self)._on_DebugTimer_timeout()
		print("------ PerceptronQLAI ------")
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