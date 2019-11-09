extends "res://Maps/Adventures/GenerationAlgorithms/GeneratorBase.gd"

const AdventureGraph = preload("res://Maps/Adventures/DataClasses/AdventureGraph.gd")
const Room = preload("res://Maps/Adventures/DataClasses/Room.gd")
const Enemy = preload("res://Maps/Adventures/DataClasses/Enemy.gd")
const Resource = preload("res://Maps/Adventures/DataClasses/Resource.gd")

# Limit time needed for score calculation
const LIMIT_TIME = 60.0
# Player rating changing rate
const PLAYER_K = 0.4
# Room rating changing rate
const ROOM_K = 0.4
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
var room_graph = null

func _ready():
	self.ratings = $Model.get_data($Model.RATINGS)
	if not self.ratings.has("Player"):
		self.ratings.Player = INITIAL_PLAYER_RATING
		$Model.set_data($Model.RATINGS, self.ratings)

# enemies: Array[EnemyEntry] -> string
func _get_group_id(enemies):
	var enemy_types = []
	for enemy in enemies:
		enemy_types.append(enemy.name)
	enemy_types.sort()
	var group_id = ":"
	for type in enemy_types:
		group_id += type + ":"
	return group_id

func _calculate_char_initial_rating(max_life, attack, defense):
	return (max_life / 9.0 + defense + attack) / 10.0 - 1.0 / 9.0

# group: Array[EnemyEntry] -> float
func _calculate_group_initial_rating(group):
	var factor = 0.5 * (len(group) + 1.0)
	var sum = 0
	for character in group:
		var char_rating = self._calculate_char_initial_rating(character.max_life, character.damage,
															  character.defense)
		sum += char_rating
	return factor * sum

# enemy_config: Array[EnemyEntry]
func _get_available_encountersR(enemy_config, idx, depth, available_encounters, stack=[]):
	if len(stack) > 0:
		var encounter_id = self._get_group_id(stack)
		available_encounters[encounter_id] = stack.duplicate()

	if depth > 0:
		for i in range(idx):
			stack.append(enemy_config[i])
			self._get_available_encountersR(enemy_config, i + 1, depth - 1,
											available_encounters, stack)
			stack.pop_back()

# room_config: Array[Table], enemy_config: Array[EnemyEntry] -> Array[Array[EnemyEntry]]
func _get_available_encounters(room_config, enemy_config):
	var max_enemies = -1
	for room in room_config:
		if room.enemies > max_enemies:
			max_enemies = room.enemies
	var available_encounters = {}
	self._get_available_encountersR(enemy_config, len(enemy_config),
									max_enemies, available_encounters)
	return available_encounters

# available_encounters: Array[EnemyEntries]
func _calculate_dungeon_ratings(available_encounters):
	for encounter_id in available_encounters.keys():
		if not self.ratings.has(encounter_id):
			var encounter = available_encounters[encounter_id]
			self.ratings[encounter_id] = self._calculate_group_initial_rating(encounter)
	$Model.set_data($Model.RATINGS, self.ratings)

# room_info: Room
func _update_ratings(room, was_cleared):
	var player_rating = self.ratings.Player
	var room_id = self._get_group_id(room.get_enemies())
	var room_rating = self.ratings[room_id]
	var time = room.time
	var score = (2 * was_cleared - 1) * (1 - min(time, LIMIT_TIME) / LIMIT_TIME)
	var rating_diff = player_rating - room_rating
	var expected_score = 0
	if rating_diff != 0:
		expected_score = (exp(2 * rating_diff) + 1) / (exp(2 * rating_diff) - 1) - 1 / rating_diff
	self.ratings.Player = player_rating + PLAYER_K * (score - expected_score)
	self.ratings[room_id] = room_rating + ROOM_K * (expected_score - score)
	$Model.set_data($Model.RATINGS, self.ratings)

# config: JsonSchema
func _process_attributes(config, win_chance):
	var attributes = {}
	for attribute_name in config.keys():
		var value
		var attribute = config[attribute_name]
		if typeof(attribute) != TYPE_DICTIONARY or \
		   (typeof(attribute) == TYPE_DICTIONARY and not attribute.has("type")):
			value = attribute
		elif attribute.type == TYPE_INT:
			var mn = attribute.minimum
			var mx = attribute.maximum
			value = mx - (mx - mn) * (win_chance - MIN_WC) / (MAX_WC - MIN_WC)
		attributes[attribute_name] = value
	return attributes

func _create_monsters(available_encounters, num, win_chance):
	if num == 0:
		return []
	var player_rating = self.ratings.Player
	var expected_rating = player_rating + log((1.0 - win_chance) / win_chance)
	var closest_encounters = []
	var min_diff = INF
	for encounter_id in available_encounters.keys():
		if len(available_encounters[encounter_id]) > num:
			continue
		var encounter_rating = self.ratings[encounter_id]
		var diff = abs(encounter_rating - expected_rating)
		if diff < min_diff:
			min_diff = diff
			closest_encounters = [available_encounters[encounter_id]]
		elif diff == min_diff:
			closest_encounters.append(available_encounters[encounter_id])
	var choosen_encounter = global.choose_one(closest_encounters)

	if self.debug_mode:
		var character_types = []
		for monster_info in choosen_encounter:
			character_types.append(monster_info.name)
		print(expected_rating, " ", character_types)
	
	var enemies = []
	for enemy_config in choosen_encounter:
		var enemy = Enemy.new(enemy_config.name)
		enemies.append(enemy)

	return enemies

# config: Array[table]
func _create_resources(config, num, win_chance):
	var sampled_configs = global.sample(config, num, true)
	global.remove_null(sampled_configs)
	var resources = []
	for resource_config in sampled_configs:
		var attributes = self._process_attributes(resource_config.attributes, win_chance)
		var resource = Resource.new(resource_config.type, attributes)
		resources.append(resource)
	return resources

func _create_room(room_config, available_encounters, resource_config, min_exits=1):
	var win_chance = global.sample_from_normal_limited(WC_MEAN, WC_STD, [MIN_WC, MAX_WC])
	var enemies = self._create_monsters(available_encounters, room_config.enemies, win_chance)
	var resources = self._create_monsters(resource_config, room_config.resources, win_chance)

	var num_exits = global.randi_range(min_exits, len(room_config.exits) - 1)
	var exit_ids = global.choose(room_config.exits, num_exits)
	return self.room_graph.create_room(room_config.type, exit_ids,
									   room_config.enemies, enemies,
									   room_config.resources, resources)

func _generate_dungeon(room_config, monster_config, resource_config):
	for i in range(len(monster_config)):
		monster_config[i] = MonsterDB.get_monster(monster_config[i])
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
	
	self.room_graph = AdventureGraph.new()
	var initial_room_config = room_config[global.choose_one(entrance_map.right)]
	var initial_room_id = self._create_room(initial_room_config, available_encounters,
											resource_config)
	var initial_room = self.room_graph.get_room(initial_room)
	self.room_graph.connect_to_city(initial_room_id, "left")
	var map = global.create_matrix(2 * self.max_rooms, self.max_rooms)
	map[self.max_rooms][0] = initial_room_id
	var tp_map = {}
	var queue = [[self.max_rooms, 0]]
	while len(queue) != 0:
		var pos = queue.pop_front()
		var room_id = map[pos[0]][pos[1]]
		var room = self.room_graph.get_room(room_id)
		room.connect("room_cleared", self, "_update_ratings", [1])
		room.connect("room_dropped", self, "_update_ratings", [0])
		for exit in room.get_exits():
			if self.str_to_matrix_index.has(exit):
				var diff = self.str_to_matrix_index[exit]
				if pos[1] + diff[1] < 0:
					# Out of map bounds
					room.remove_exit(exit)
					continue
				var neighbor_id = map[pos[0] + diff[0]][pos[1] + diff[1]]
				if neighbor_id == null:
					if len(rooms_info) >= self.max_rooms:
						# Needs a neighbor but max rooms reached
						room.remove_exit(exit)
					else:
						# Needs a neighbor and max rooms not reached yet
						var neighbor_config = room_config[global.choose_one(entrance_map[exit])]
						var min_exits = 1
						if len(queue) == 0 and len(rooms_info) + 1 < self.min_rooms:
							min_exits = 2
						neighbor_id = self._create_room(neighbor_config,
														available_encounters,
														resource_config, min_exits)
						map[pos[0] + diff[0]][pos[1] + diff[1]] = neighbor_id
						queue.append([pos[0] + diff[0], pos[1] + diff[1]])
						self.room_graph.connect_rooms(room_id, exit, neighbor_id)
				else:
					var neighbor = self.room_graph.get_room(neighbor_id)
					if neighbor.has_exit(self.exit_entrance_map[exit]):
						# Already has a neighbor and it has an exit leading to this
						neighbor.remove_exit(self.exit_entrance_map[exit])
					else:
						# Already has a neighbor and it does not have an exit leading to this
						room.remove_exit(exit)
			else:
				if tp_map.has(exit):
					# Has a room waiting for connection
					self.room_graph.connect_rooms(room_id, exit, tp_map[exit])
					tp_map.erase(exit)
				else:
					# Does not have a room waiting for connection
					tp_map[exit] = room_id
	for exit in tp_map.keys():
		var room = self.room_graph(tp_map[exit])
		room.remove_exit(exit)
	self.print_graph(rooms_info)
	self.print_monsters(rooms_info)
	self.print_resources(rooms_info)
	return rooms_info