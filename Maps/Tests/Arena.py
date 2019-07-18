from random import random

import matplotlib.pyplot as plt

from godot import exposed, export
from godot.bindings import *
from godot.globals import *

@exposed
class Arena(Node2D):
	def _ready(self):
		self.debug_mode = False
		self.initial_positions = {}
		self.tile_size = 32
		self.arena_width = 27 * self.tile_size
		self.arena_height = 13 * self.tile_size
		for character in self.get_tree().get_nodes_in_group("character"):
			self.initial_positions[character.name] = character.position
			character.connect("character_death", self, "_on_character_death")

	def print_info(self):
		glob = self.get_node("/root/global")
		loss_info = {}
		print("------------")
		for character in self.get_tree().get_nodes_in_group("robot"):
			team = glob.get_team(character)
			print(character.get_pretty_name() + ": " + str(character.life) + " (" + team + ")")
			if loss_info.get(team) is None:
				loss_info[team] = {}
			loss_info[team][character.name] = character.controller.get_loss()
		
		fig, ax = plt.subplots(len(loss_info), 1)
		for i, (team, val) in enumerate(loss_info.items()):
			for j, (name, loss) in enumerate(val.items()):
				if len(loss_info) == 1:
					ax.set_title(name)
					ax.plot(loss)
				elif len(val) == 1:
					ax[i].set_title(name)
					ax[i].plot(loss)
				else:
					ax[i][j].set_title(name)
					ax[i][j].plot(loss)
		plt.show()

	def reset(self, timeout):
		if self.debug_mode:
			self.print_info()
		# self.get_parent().reset_game()
		for character in self.get_tree().get_nodes_in_group("character"):
			character.before_reset(timeout)
		for character in self.get_tree().get_nodes_in_group("character"):
			off = 2 * self.tile_size
			character.position = Vector2(off + self.arena_width*random(), off + self.arena_height*random())
			character.reset(timeout)
		for character in self.get_tree().get_nodes_in_group("character"):
			character.after_reset(timeout)

	def _on_character_death(self):
		self.get_node("TimeoutTimer").start()
		self.reset(False)

	def _on_TimeoutTimer_timeout(self):
		self.reset(True)
