extends Node2D

signal room_cleared(room_info)
signal room_dropped(room_info)

enum GeneratorAgorithm { RANDOM, ELO_RATING }

const GENERATOR_PATH = {
	GeneratorAgorithm.RANDOM: "res://Maps/Adventures/GenerationAlgorithms/RandomGenerator.tscn",
	GeneratorAgorithm.ELO_RATING: "res://Maps/Adventures/GenerationAlgorithms/EloRatingGenerator.tscn"
}

# Abstract
const room_config = []
const monster_config = []
const resource_config = []

const NUM_PERSISTED_NN = 4

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
#				 "type": preload("res://.../Goblin.tscn"),
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
# 	{
# 		"type": preload("res://Characters/Goblin/Goblin.tscn"),
# 		"attributes": {
# 			"character_type": "Goblin",
# 			"max_life": 3,
# 			"damage": 1,
# 			"defense": 0,
# 			"ai_type": 1 # TorchQLAI
# 		}
# 	},
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
	self.rooms_info = generator.generate_dungeon(room_config, monster_config, resource_config)
	for room in self.rooms_info:
		room.time = 0.0
		for monster in room.monsters:
			if monster != null:
				monster.attributes.network_id = global.randi_range(0, NUM_PERSISTED_NN - 1)
	self._create_room(0, "left")
	var player = global.find_entity("player")
	self._save_attributes(player, self.player_attributes)

func back_to_city():
	var CityScene = load("res://Maps/City/City.tscn")
	var main = global.find_entity("main")
	main.change_map(CityScene, {"player_pos": "dungeon"})

func change_room(exit_id):
	self._save_room()
	var current_room_exits = self.rooms_info[self.current_room_id].exits
	var next_room_id = current_room_exits[exit_id].room
	if next_room_id == -1:
		self.back_to_city()
		return
	var entrance = current_room_exits[exit_id].entrance
	self.current_room_node.queue_free()
	yield(self.current_room_node, "tree_exited")
	self._create_room(next_room_id, entrance)

func _create_room(room_id, player_pos):
	var room_info = self.rooms_info[room_id]
	self.current_room_node = room_info.type.instance()
	self.add_child(self.current_room_node)
	var node_params = {
		"player_pos": player_pos,
		"available_entrances": room_info.exits.keys()
	}
	self.current_room_node.debug_mode = self.debug_mode
	self.current_room_node.init(node_params)
	self.current_room_id = room_id

	if typeof(room_info.monsters) == TYPE_ARRAY:
		self._init_spawners(room_info)

	self._load_room()

func _init_spawners(room_info):
	var mapped_monsters = {}
	var spawners = self.current_room_node.get_node("MonsterSpawners").get_children()
	for i in range(len(spawners)):
		if room_info.monsters[i] == null:
			continue
		var spawner = spawners[i]
		mapped_monsters[spawner.name] = room_info.monsters[i]
	room_info.monsters = mapped_monsters
	room_info.alive_monsters = room_info.monsters.size()

	var mapped_resources = {}
	spawners = self.current_room_node.get_node("ResourceSpawners").get_children()
	for i in range(len(spawners)):
		if room_info.resources[i] == null:
			continue
		var spawner = spawners[i]
		mapped_resources[spawner.name] = room_info.resources[i]
	room_info.resources = mapped_resources

func _save_room():
	var room_info = self.rooms_info[self.current_room_id]
	for monster_info in room_info.monsters.values():
		if monster_info.has("instance"):
			self._save_attributes(monster_info.instance, monster_info.attributes)
			monster_info.instance.end()
			monster_info.erase("instance")
	for resource_info in room_info.resources.values():
		self._save_attributes(resource_info.instance, resource_info.attributes)
		resource_info.erase("instance")
	var player = global.find_entity("player")
	self._save_attributes(player, self.player_attributes)

func _load_room():
	var room_info = self.rooms_info[self.current_room_id]

	var player = global.find_entity("player")
	player.connect("character_death", self, "_on_player_death")
	player.init(self.player_attributes)
	if not self.player_attributes.has("life"):
		self.player_attributes.life = player.max_life

	var monster_spawners = self.current_room_node.get_node("MonsterSpawners").get_children()
	var resource_spawners = self.current_room_node.get_node("ResourceSpawners").get_children()
	var wall = self.current_room_node.get_node("Wall")
	for spawner in monster_spawners:
		if not room_info.monsters.has(spawner.name):
			continue
		var monster_info = room_info.monsters[spawner.name]
		if monster_info.attributes.has("life") and \
		   monster_info.attributes.life == 0:
			continue
		var monster = monster_info.type.instance()
		monster_info.instance = monster
		monster.position = spawner.position
		monster.add_to_group("team2")
		monster.connect("character_death", self, "_on_monster_death", [monster, spawner])
		wall.add_child(monster)
		monster.init(monster_info.attributes)
		if not monster_info.attributes.has("life"):
			monster_info.attributes.life = monster.max_life
	
	for spawner in resource_spawners:
		if not room_info.resources.has(spawner.name):
			continue
		var resource_info = room_info.resources[spawner.name]
		var resource = resource_info.type.instance()
		resource_info.instance = resource
		resource.position = spawner.position
		resource.connect("interacted", self, "_on_resource_collected", [resource, spawner])
		wall.add_child(resource)
		resource.init(resource_info.attributes)

func _save_attributes(scene, attribute_table):
	for attribute in attribute_table.keys():
		attribute_table[attribute] = scene[attribute]

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
