extends "res://Maps/Adventures/GenerationAlgorithms/GeneratorBase.gd"

# Limit time needed for score calculation
const LIMIT_TIME = 60.0
# Player rating changing rate
const PLAYER_K = 0.2
# Room rating changing rate
const ROOM_K = 0.2
# Win chance average
const WC_MEAN = 0.725
# Win chance standard deviation
const WC_STD = 0.07
# Minimum win chance
const MIN_WC = 0.5
# Maximum win chance
const MAX_WC = 0.95
# Initial player rating (decided according to self._calculate_char_initial_rating)
const INITIAL_PLAYER_RATING = 11.0 / 9.0

var ratings = {}

func _ready():
	self.get_parent().connect("room_cleared", self, "_update_ratings", [1])
	self.get_parent().connect("room_dropped", self, "_update_ratings", [0])
	self.ratings = $Model.get_data($Model.RATINGS)
	if not self.ratings.has("Player"):
		self.ratings.Player = INITIAL_PLAYER_RATING
		$Model.set_data($Model.RATINGS, self.ratings)

func _get_group_id(monsters_info):
	var monster_types = []
	for monster in monsters_info:
		if typeof(monster) == TYPE_DICTIONARY:
			monster_types.append(monster.attributes.character_type)
		else:
			monster_types.append(monster.name)
	monster_types.sort()
	var group_id = ":"
	for type in monster_types:
		group_id += type + ":"
	return group_id

func _calculate_char_initial_rating(max_life, attack, defense):
	return (max_life / 9.0 + defense + attack) / 10.0 - 1.0 / 9.0

func _calculate_encounter_initial_rating(encounter):
	var factor = 0.5 * (len(encounter) + 1)
	var sum = 0
	for character in encounter:
		var char_rating = self._calculate_char_initial_rating(character.max_life, character.damage,
															  character.defense)
		sum += char_rating
	return factor * sum

func _get_available_encountersR(monster_config, idx, depth, available_encounters, stack=[]):
	if len(stack) > 0:
		var encounter_id = self._get_group_id(stack)
		available_encounters[encounter_id] = stack.duplicate()

	if depth > 0:
		for i in range(idx):
			stack.append(monster_config[i])
			self._get_available_encountersR(monster_config, i + 1, depth - 1,
											available_encounters, stack)
			stack.pop_back()

func _get_available_encounters(room_config, monster_config):
	var max_monsters = -1
	for room in room_config:
		if room.monsters > max_monsters:
			max_monsters = room.monsters
	var available_encounters = {}
	self._get_available_encountersR(monster_config, len(monster_config),
									max_monsters, available_encounters)
	return available_encounters

func _calculate_dungeon_ratings(available_encounters):
	for encounter_id in available_encounters.keys():
		if not self.ratings.has(encounter_id):
			var encounter = available_encounters[encounter_id]
			self.ratings[encounter_id] = self._calculate_encounter_initial_rating(encounter)
	$Model.set_data($Model.RATINGS, self.ratings)

func _update_ratings(room_info, was_cleared):
	var player_rating = self.ratings.Player
	var room_id = self._get_group_id(room_info.monsters.values())
	var room_rating = self.ratings[room_id]
	var time = room_info.time
	var score = (2 * was_cleared - 1) * (1 - min(time, LIMIT_TIME) / LIMIT_TIME)
	var rating_diff = player_rating - room_rating
	var expected_score = 0
	if rating_diff != 0:
		expected_score = (exp(2 * rating_diff) + 1) / (exp(2 * rating_diff) - 1) - 1 / rating_diff
	self.ratings.Player = player_rating + PLAYER_K * (score - expected_score)
	self.ratings[room_id] = room_rating + ROOM_K * (expected_score - score)
	$Model.set_data($Model.RATINGS, self.ratings)

func _process_attributes(config, win_chance):
	var attributes = {}
	for attribute_name in config.keys():
		var value
		var attribute = config[attribute_name]
		if typeof(attribute) != TYPE_DICTIONARY or not attribute.has("type"):
			value = attribute
		elif attribute.type == TYPE_INT:
			var mn = attribute.minimum
			var mx = attribute.maximum
			value = mx - (mx - mn) * (win_chance - MIN_WC) / (MAX_WC - MIN_WC)
		attributes[attribute_name] = value
	return attributes

func _create_monsters(num, config, win_chance, available_encounters):
	if num == 0:
		return []
	var player_rating = self.ratings.Player
	var expected_rating = player_rating + log((1.0 - win_chance) / win_chance)
	var closest_encounters = []
	var min_encounter_rating_diff = INF
	for encounter_id in available_encounters.keys():
		if len(available_encounters[encounter_id]) > num:
			continue
		var encounter_rating = self.ratings[encounter_id]
		var encounter_rating_diff = abs(encounter_rating - expected_rating)
		if encounter_rating_diff < min_encounter_rating_diff:
			min_encounter_rating_diff = encounter_rating_diff
			closest_encounters = [available_encounters[encounter_id]]
		elif encounter_rating_diff == min_encounter_rating_diff:
			closest_encounters.append(available_encounters[encounter_id])
	var choosen_encounter = global.choose_one(closest_encounters)

	if self.debug_mode:
		var character_types = []
		for monster_info in choosen_encounter:
			character_types.append(monster_info.name)
		print(expected_rating, " ", character_types)
	
	var monsters = []
	for monster_config in choosen_encounter:
		var new_monster = {
			"type": monster_config.type,
			"attributes": {
				"character_type": monster_config.name,
				"damage": monster_config.damage,
				"defense": monster_config.defense,
				"max_life": monster_config.max_life
			}
		}
		for attribute in monster_config.ATTRIBUTES:
			new_monster.attributes[attribute] = monster_config[attribute]
		monsters.append(new_monster)

	for i in range(num - len(monsters)):
		monsters.append(null)

	return global.shuffle_array(monsters)

func _create_resources(num, config, win_chance):
	var sampled_configs = global.sample(config, num, true)
	var resources = []
	for resource_config in sampled_configs:
		if resource_config == null:
			resources.append(null)
			continue
		var resource = {
			"type": resource_config.type,
			"attributes": self._process_attributes(resource_config.attributes, win_chance)
		}
		resources.append(resource)
	return resources

func _generate_dungeon(room_config, monster_config, resource_config):
	var available_encounters = self._get_available_encounters(room_config, monster_config)
	self._calculate_dungeon_ratings(available_encounters)
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
		var win_chance = global.sample_from_normal_limited(WC_MEAN, WC_STD, [MIN_WC, MAX_WC])
		room.monsters = self._create_monsters(room.monsters, monster_config,
											  win_chance, available_encounters)
		room.resources = self._create_resources(room.resources, resource_config, win_chance)
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
						var min_exits = 1
						if len(queue) == 0 and len(rooms_info) + 1 < self.min_rooms:
							min_exits = 2
						var neighbor = self.create_room(neighbor_config, min_exits)
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