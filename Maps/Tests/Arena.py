from random import random

import matplotlib.pyplot as plt

from godot import exposed, export
from godot.bindings import *
from godot.globals import *

GRAPH_FREQUENCY = 10

@exposed
class Arena(Node2D):
	def _ready(self):
		self.debug_mode = False
		self.initial_positions = {}
		self.tile_size = 32
		self.arena_width = 27 * self.tile_size
		self.arena_height = 13 * self.tile_size
		self.rounds = 0
		for character in self.get_tree().get_nodes_in_group("character"):
			self.initial_positions[character.name] = character.position
			character.connect("character_death", self, "_on_character_death", Array([character]))
			character.init(Dictionary())
			# character.init(Dictionary({"network_id": character.name}))

	def init(self, params):
		pass

	def print_info(self):
		glob = self.get_node("/root/global")
		loss_info = {}
		print("------------")
		for character in self.get_tree().get_nodes_in_group("robot"):
			pretty_name = character.get_pretty_name()
			print(pretty_name + ": " + str(character.life))
			loss_info[pretty_name] = character.controller.get_loss()
		
		if self.rounds % GRAPH_FREQUENCY == 0:
			fig, ax = plt.subplots(1, 1, figsize=(7, 3))
			for i, (name, loss) in enumerate(loss_info.items()):
				# ax.set_title(name)
				ax.plot(loss, label=name)
			ax.legend()
			plt.tight_layout()
			plt.show()

	def reset(self, timeout):
		self.rounds += 1
		if self.debug_mode:
			self.print_info()
		# self.get_parent().reset_game()
		characters = self.get_tree().get_nodes_in_group("character")
		for character in characters:
			character.before_reset(timeout)
		for character in characters:
			off = 2 * self.tile_size
			xPos = off + self.arena_width * random()
			yPos = off + self.arena_height * random()
			character.position = Vector2(xPos, yPos)
			character.reset(timeout)
		for character in characters:
			character.after_reset(timeout)
		for character in characters:
			character.end()

	def _on_character_death(self, character):
		self.get_node("TimeoutTimer").start()
		print(f"{character.name} lost!")
		self.reset(False)

	def _on_TimeoutTimer_timeout(self):
		self.reset(True)
		print("Timeout!")
