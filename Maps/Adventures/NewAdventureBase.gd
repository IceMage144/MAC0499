extends Node2D

signal room_cleared(room_info)
signal room_dropped(room_info)

enum GeneratorAgorithm { RANDOM, RATING }

const GENERATOR_PATH = {
	GeneratorAgorithm.RANDOM: "res://Maps/Adventures/GenerationAlgorithms/RandomGenerator.tscn",
	GeneratorAgorithm.RATING: "res://Maps/Adventures/GenerationAlgorithms/RatingGenerator.tscn"
}

# Abstract
const room_config = []
const monster_config = []
const resource_config = []

export(GeneratorAgorithm) var generator_class = GeneratorAgorithm.RANDOM
export(int, 1, 50) var max_rooms = 1
export(int, 1, 50) var min_rooms = 1

var debug_mode = false
var generator = null
var current_room_id = 0
var current_room_node = null
var player_attributes = {}
var rooms_info = {}

#rooms_info = [
#	{
#		"type": preload("res://.../Room1.tscn"),
#		"time": 13.0, # seconds
#		"alive_monsters": 2,
#		"monsters": {
#			"Position2D": {
#				"type": preload("res://.../Goblin.tscn"),
#				"attributes": {
#					"character_type": "Goblin",
#					"network_id": 2,
# 					"max_life": 3,
#					"life": 3,
# 					"damage": 1,
# 					"defense": 0,
# 					"ai_type": 1 # TorchQLAI
#				}
#			},
#			"Position2D2": {
#				"type": preload("res://.../Spider.tscn"),
#				"attributes": {
#					"character_type": "Spider",
#					"network_id": 3,
# 					"max_life": 3,
#					"life": 2,
# 					"damage": 1,
# 					"defense": 0,
# 					"ai_type": 2 # MemoQLAI
#				}
#			}
#		},
#		"resources": {
#			"Position2D": {
#				"type": preload("res://.../MoneyChest.tscn"),
#				"attributes": {
#					"amount": 100
#				}
#			}
#		},
#		"exits": {
#			"right": {
#				"room": -1 # city
#			},
#			"left": {
#				"room": 3,
#				"entrance": "right"
#			},
#			"teleport": {
#				"room": 5,
#				"entrance": "teleport"
#			}
#		}
#	},
#	...
#]

#room_config = [
#	{
#		"type": preload("res://.../Room1.tscn")
#		"exits": ["left", "right", "top", "down"],
#		"monsters": 2,
#		"resources": 1
#	},
#	{
#		"type": preload("res://.../Room2.tscn")
#		"exits": ["left", "right", "top", "down", "teleport"],
#		"monsters": 0,
#		"resources": 2
#	},
#	...
#]

#monster_config = [
# 	"Goblin",
# 	"Spider",
# 	...
# ]

#resource_config = [
#	{
#		"type": preload("res://.../MoneyChest.tscn"),
#		"attributes": {
#			"amount": {
#				"type": "number",
#				"minimum": 100,
#				"maximum": 300
#			}
#		}
#	},
#	{
#		"type": preload("res://.../ItemChest.tscn"),
#		"attributes": {
#			"item": {
#				"type": "item",
#				"enum": ["Small Potion", "Medium Potion"]
#			}
#		}
#	},
#	...
#]

func init(params):
	var generator_class = load(GENERATOR_PATH[self.generator_class])
	var generator = generator_class.instance()
	self.generator = generator
	self.add_child(generator)
	generator.init({
		"debug_mode": self.debug_mode,
		"max_rooms": self.max_rooms,
		"min_rooms": self.min_rooms
	})
	var room_config = self.get_room_config()
	var enemy_config = self.get_enemy_config()
	var resource_config = self.get_resource_config()
	self.rooms_info = generator.generate_dungeon(room_config, enemy_config, resource_config)
	self.rooms_info.change_room()

func _on_monster_death(monster, spawner):
	var current_room = self.rooms_info[self.current_room_id]
	var monster_info = current_room.monsters[spawner.name]
	self._save_attributes(monster_info.instance, monster_info.attributes)
	monster.queue_free()
	monster_info.erase("instance")
	current_room.alive_monsters -= 1
	if current_room.alive_monsters == 0:
		self.emit_signal("room_cleared", current_room)

func _on_resource_collected(resource, spawner):
	var current_room = self.rooms_info[self.current_room_id]
	current_room.resources.erase(spawner.name)

func _on_player_death():
	self._save_room()
	self.emit_signal("room_dropped", self.rooms_info[self.current_room_id])
	self.back_to_city()

func _on_Timer_timeout():
	self.rooms_info[self.current_room_id].time += 1.0
