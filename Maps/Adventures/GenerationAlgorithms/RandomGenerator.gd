extends "res://Maps/Adventures/GenerationAlgorithms/GeneratorBase.gd"

func _process_attributes(config):
	var attributes = {}
	for attribute_name in config.keys():
		var value
		var attribute = config[attribute_name]
		if typeof(attribute) != TYPE_DICTIONARY or not attribute.has("type"):
			value = attribute
		elif attribute.type == TYPE_INT:
			value = global.randi_range(attribute.minimum, attribute.maximum)
		attributes[attribute_name] = value
	return attributes

func _create_monsters(num, config):
	var monsters = []
	var sampled_monsters = global.sample(config, num)
	for monster_config in sampled_monsters:
		monsters.append({
			"type": monster_config.type,
			"attributes": {
				"character_type": monster_config.name,
				"damage": monster_config.damage,
				"defense": monster_config.defense,
				"max_life": monster_config.max_life,
				"ai_type": monster_config.ai_type
			}
		})
	return monsters

func _create_resources(num, config):
	var sampled_configs = global.sample(config, num, true)
	var resources = []
	for resource_config in sampled_configs:
		if resource_config == null:
			resources.append(null)
			continue
		var resource = {
			"type": resource_config.type,
			"attributes": self._process_attributes(resource_config.attributes)
		}
		resources.append(resource)
	return resources

func _generate_dungeon(room_config, monster_config, resource_config):
	var entrance_map = {}
	for i in range(len(room_config)):
		var room = room_config[i]
		for exit in room.exits:
			var entrance = global.dict_get(self.exit_entrance_map, exit, exit)
			if not entrance_map.has(entrance):
				entrance_map[entrance] = []
			entrance_map[entrance].append(i)
	
	var initial_room_config = room_config[global.choose_one(entrance_map.right)]
	var initial_room = self.create_room(initial_room_config)
	initial_room.exits.left = self.create_exit(-1, null)
	var map = global.create_matrix(2 * self.max_rooms, self.max_rooms)
	map[self.max_rooms][0] = 0
	var tp_map = {}
	var rooms_info = [initial_room]
	var queue = [[self.max_rooms, 0]]
	while len(queue) != 0:
		var pos = queue.pop_front()
		var room_id = map[pos[0]][pos[1]]
		var room = rooms_info[room_id]
		room.monsters = self._create_monsters(room.monsters, monster_config)
		room.resources = self._create_resources(room.resources, resource_config)
		for exit in room.exits.keys():
			if room.exits[exit] != null:
				# Already has a room associated with this exit
				continue
			if self.str_to_matrix_index.has(exit):
				var diff = self.str_to_matrix_index[exit]
				if pos[1] + diff[1] < 0:
					# Out of map bounds
					room.exits.erase(exit)
					continue
				var neighbor_id = map[pos[0] + diff[0]][pos[1] + diff[1]]
				if neighbor_id == null:
					if len(rooms_info) >= self.max_rooms:
						# Needs a neighbor but max rooms reached
						room.exits.erase(exit)
					else:
						# Needs a neighbor and max rooms not reached yet
						var neighbor_config = room_config[global.choose_one(entrance_map[exit])]
						var neighbor = self.create_room(neighbor_config)
						neighbor_id = len(rooms_info)
						map[pos[0] + diff[0]][pos[1] + diff[1]] = neighbor_id
						rooms_info.append(neighbor)
						queue.append([pos[0] + diff[0], pos[1] + diff[1]])
						room.exits[exit] = self.create_exit(neighbor_id, exit)
						var neighbor_exit = self.exit_entrance_map[exit]
						neighbor.exits[neighbor_exit] = self.create_exit(room_id, neighbor_exit)
				else:
					var neighbor = rooms_info[neighbor_id]
					if neighbor.exits.has(self.exit_entrance_map[exit]):
						# Already has a neighbor and it has an exit leading to this
						room.exits[exit] = self.create_exit(neighbor_id, exit)
					else:
						# Already has a neighbor and it does not have an exit leading to this
						room.exits.erase(exit)
			else:
				if tp_map.has(exit):
					# Has a room waiting for connection
					room.exits[exit] = self.create_exit(tp_map[exit], exit)
					var next_room = rooms_info[tp_map[exit]]
					next_room.exits[exit] = self.create_exit(room_id, exit)
					tp_map.erase(exit)
				else:
					# Does not have a room waiting for connection
					tp_map[exit] = room_id
	for exit in tp_map.keys():
		var room = rooms_info[tp_map[exit]]
		room.exits.erase(exit)
	self.print_graph(rooms_info)
	self.print_monsters(rooms_info)
	self.print_resources(rooms_info)
	return rooms_info