extends Node2D

const CityScene = preload("res://Maps/City/City.tscn")

var debug_mode = false
var current_room_id = 0
var current_room_node = null
var rooms_info = {}

#rooms_info = {
#	0: {
#		"type": preload("res://.../room1.tscn"),
#		"left": {
#			"room": 3,
#			"entrance": "right"
#		},
#		"teleport": {
#			"room": 5,
#			"entrance": "teleport"
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
	var current_room_info = self.rooms_info[self.current_room_id]
	var next_room_id = current_room_info[exit_id]["room"]
	var entrance = current_room_info[exit_id]["entrance"]
	var next_room_type = self.room_types[next_room_id]["type"]
	
	self.current_room_node.queue_free()
	yield(self.current_room_node, "tree_exited")
	self.current_room_node = next_room_type.instance()
	self.add_child(self.current_room_node)
	self.current_room_node.init({"player_pos": entrance})
	self.current_room_node.debug_mode = self.environment_debug
	self.current_room_id = next_room_id