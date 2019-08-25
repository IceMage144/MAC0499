extends Node2D

enum GeneratorAgorithm { RANDOM, ELO_RATING }

const GENERATOR_PATH = {
	GeneratorAgorithm.RANDOM: "res://Maps/Adventures/GenerationAlgorithms/RandomGenerator.tscn",
	GeneratorAgorithm.ELO_RATING: "res://Maps/Adventures/GenerationAlgorithms/EloRatingGenerator.tscn"
}

const room_config = []
const monster_config = []
const resource_config = []

export(GeneratorAgorithm) var generator = GeneratorAgorithm.RANDOM

var debug_mode = false
var current_room_id = 0
var current_room_node = null
var rooms_info = {}

#rooms_info = [
#	{
#		"type": preload("res://.../Room1.tscn"),
#		"monsters": {
#			"Position2D": {
#				"type": preload("res://.../Goblin.tscn"),
#				"life": 3
#			},
#			"Position2D2": {
#				"type": preload("res://.../Spider.tscn"),
#				"life": 2
#			}
#		},
#		"resources": {
#			"Position2D": {
#				"type": preload("res://.../MoneyChest.tscn"),
#				"amount": 100
#			}
#		},
#		"exits": {
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
#	preload("res://.../Goblin.tscn"),
#	preload("res://.../Spider.tscn"),
#	...
#]

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

func _ready():
	var generator_class = load(GENERATOR_PATH[generator])
	var generator = generator_class.instance()
	self.add_child(generator)
	self.rooms_info = generator.generate_dungeon(room_config, monster_config, resource_config)
	self.create_room(0, "left")

func init(params):
	pass

func back_to_city():
	var CityScene = load("res://Maps/City/City.tscn")
	var main = global.find_entity("main")
	main.change_map(CityScene, {"player_pos": "dungeon"})

func create_room(room_id, player_pos):
	var room_info = self.rooms_info[room_id]
	self.current_room_node = room_info.type.instance()
	self.add_child(self.current_room_node)
	var node_params = {
		"player_pos": player_pos,
		"available_entrances": room_info.exits.keys()
	}
	self.current_room_node.init(node_params)
	self.current_room_node.debug_mode = self.debug_mode
	self.current_room_id = room_id

func change_room(exit_id):
	var current_room_exits = self.rooms_info[self.current_room_id].exits
	var next_room_id = current_room_exits[exit_id].room
	if next_room_id == -1:
		self.back_to_city()
		return
	var entrance = current_room_exits[exit_id].entrance
	self.current_room_node.queue_free()
	yield(self.current_room_node, "tree_exited")
	self.create_room(next_room_id, entrance)