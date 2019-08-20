extends Node2D

const CityScene = preload("res://Maps/City/City.tscn")

enum BuilderAgorithm { RANDOM, ELO_RATING }

const ALGORITHM_PATH = {
	BuilderAgorithm.RANDOM: "res://Maps/Adventures/GenerationAlgorithms/RandomGenerator.tscn",
	BuilderAgorithm.ELO_RATING: "res://Maps/Adventures/GenerationAlgorithms/EloRatingGenerator.tscn"
}

export(BuilderAgorithm) var algorithm = BuilderAgorithm.RANDOM

var debug_mode = false
var current_room_id = 0
var current_room_node = null
var rooms_info = {}

#rooms_info = {
#	0: {
#		"type": preload("res://.../room1.tscn"),
#		"monsters": {
#			"Position2D": "Goblin",
#			"Position2D2": "Spider"
#		},
#		"resources": {
#			"Position2D": "Chest"
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
#}

func _ready():
	self.generate_dungeon()

func generate_dungeon():
	pass

func back_to_city():
	var main = global.find_entity("main")
	main.change_map(CityScene, {"player_pos": "dungeon"})

func change_room(exit_id):
	var current_room_exits = self.rooms_info[self.current_room_id].exits
	var next_room_id = current_room_exits[exit_id].room
	var entrance = current_room_exits[exit_id].entrance
	var next_room_info = self.room_types[next_room_id]
	
	self.current_room_node.queue_free()
	yield(self.current_room_node, "tree_exited")
	self.current_room_node = next_room_info.type.instance()
	self.add_child(self.current_room_node)
	var node_params = {
		"player_pos": entrance,
		"available_entrances": next_room_info.exits.keys()
	}
	self.current_room_node.init(node_params)
	self.current_room_node.debug_mode = self.environment_debug
	self.current_room_id = next_room_id