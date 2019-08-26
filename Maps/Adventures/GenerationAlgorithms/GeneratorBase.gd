extends Node

const exit_entrance_map = {
	"left": "right",
	"right": "left",
	"top": "bottom",
	"bottom": "top"
}
const str_to_matrix_index = {
	"left": [0, -1],
	"right": [0, 1],
	"top": [-1, 0],
	"bottom": [1, 0]
}

var debug_mode = false
var max_rooms = 1

func init(params):
	self.max_rooms = params.max_rooms
	self.debug_mode = params.debug_mode

func create_room(room_config):
	var num_exits = global.randi_range(1, len(room_config.exits) - 1)
	var exit_ids = global.choose(room_config.exits, num_exits)
	var exits = {}
	for exit in exit_ids:
		exits[exit] = null
	return {
		"type": room_config.type,
		"monsters": room_config.monsters,
		"resources": room_config.resources,
		"exits": exits
	}

func create_exit(room, exit):
	return {
		"room": room,
		"entrance": global.dict_get(self.exit_entrance_map, exit, exit)
	}

func print_graph(room_info):
	if not self.debug_mode:
		return
	for i in range(len(room_info)):
		var room = room_info[i]
		print(str(i) + ":")
		for exit_id in room.exits.keys():
			var exit = room.exits[exit_id]
			print(" " + exit_id + ": " + str(exit.room))
		print()

func print_monsters(room_info):
	if not self.debug_mode:
		return
	for i in range(len(room_info)):
		var room = room_info[i]
		print(str(i) + ":")
		for monster in room.monsters:
			print(" " + str(monster))

func print_resources(room_info):
	if not self.debug_mode:
		return
	for i in range(len(room_info)):
		var room = room_info[i]
		print(str(i) + ":")
		for resource in room.resources:
			print(" " + str(resource))

# Abstract
func generate_dungeon(room_config, monster_config, resource_config):
	pass