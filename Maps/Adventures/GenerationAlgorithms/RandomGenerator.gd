extends Node

const exit_entrance_map = {
	"left": "right",
	"right": "left",
	"top": "bottom",
	"bottom": "top"
}

func generate_dungeon(room_config, monster_config, resource_config):
	return [
		{
			"type": room_config[0].type,
			"monsters": {},
			"resources": {},
			"exits": {
				"right": {
					"room": 1,
					"entrance": "left"
				},
				"left": {
					"room": -1
				}
			}
		},
		{
			"type": room_config[1].type,
			"monsters": {},
			"resources": {},
			"exits": {
				"left": {
					"room": 0,
					"entrance": "right"
				}
			}
		}
	]
	pass
	# var entrance_map = {}
	# for i in range(len(room_config)):
	# 	var room = room_config[i]
	# 	for exit in room.exits:
	# 		var entrance = exit
	# 		if exit_entrance_map.has(exit):
	# 			entrance = exit_entrance_map[exit]
	# 		if not entrance_map.has(entrance):
	# 			entrance_map[entrance] = []
	# 		entrance_map[entrance].append(i)
	
	# var rooms_info = []
