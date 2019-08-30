import os

import torch
import torch.nn as nn

from godot import exposed, export
from godot.bindings import *
from godot.globals import *


@exposed
class NNParamsSaveManager(Node):
	FILE_PATH = os.path.join(OS.get_user_data_dir(), "nn_params.save")
	def _ready(self):
		self._load()
	
	def _load(self):
		if os.path.exists(self.FILE_PATH):
			self.params_cache = torch.load(self.FILE_PATH)
		else:
			self.params_cache = {}
	
	def _save(self):
		torch.save(self.params_cache, self.FILE_PATH)
	
	def get_params(self, key):
		return self.params_cache.get(key)

	def set_params(self, key, new_params):
		self.params_cache[key] = new_params
		self._save()