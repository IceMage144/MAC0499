extends Node2D

enum GeneratorAgorithm { RANDOM, ELO_RATING }

const GENERATOR_PATH = {
	GeneratorAgorithm.RANDOM: "res://Maps/Adventures/GenerationAlgorithms/RandomGenerator.tscn",
	GeneratorAgorithm.ELO_RATING: "res://Maps/Adventures/GenerationAlgorithms/EloRatingGenerator.tscn"
}

const room_config = []
const monster_config = []
const resource_config = []

export(GeneratorAgorithm) var generator_class = GeneratorAgorithm.RANDOM
export(int, 1, 50) var max_rooms = 1

var debug_mode = false
var generator = null
var current_room_id = 0
var current_room_node = null
var player_attributes = {"life": -1}
var rooms_info = {}

#rooms_info = [
#	{
#		"type": preload("res://.../Room1.tscn"),
#		"monsters": {
#			"Position2D": {
#				"type": preload("res://.../Goblin.tscn"),
#				"attributes": {
#					"life": 3
#				}
#			},
#			"Position2D2": {
#				"type": preload("res://.../Spider.tscn"),
#				"attributes": {
#					"life": 2
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

func init(params):
	var generator_class = load(GENERATOR_PATH[self.generator_class])
	var generator = generator_class.instance()
	self.generator = generator
	self.add_child(generator)
	generator.init({
		"debug_mode": self.debug_mode,
		"max_rooms": self.max_rooms
	})
	self.rooms_info = generator.generate_dungeon(room_config, monster_config, resource_config)
	self.create_room(0, "left")
	var player = global.find_entity("player")
	self._save_attributes(player, self.player_attributes)

func back_to_city():
	var CityScene = load("res://Maps/City/City.tscn")
	var main = global.find_entity("main")
	main.change_map(CityScene, {"player_pos": "dungeon"})

func _apply_attributes(scene, attribute_table):
	for attribute in attribute_table.keys():
		var value = attribute_table[attribute]
		if attribute == "life":
			if value == -1:
				attribute_table[attribute] = scene.max_life
			else:
				scene.set_life(value)
		else:
			scene[attribute] = value

func _save_attributes(scene, attribute_table):
	for attribute in attribute_table.keys():
		attribute_table[attribute] = scene[attribute]

func create_room(room_id, player_pos):
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

	var player = global.find_entity("player")
	player.connect("character_death", self, "_on_player_death")
	self._apply_attributes(player, self.player_attributes)

	if typeof(room_info.monsters) == TYPE_ARRAY:
		var mapped_monsters = {}
		var spawners = self.current_room_node.get_node("MonsterSpawners").get_children()
		for i in range(len(spawners)):
			var spawner = spawners[i]
			mapped_monsters[spawner.name] = room_info.monsters[i]
		room_info.monsters = mapped_monsters

		var mapped_resources = {}
		spawners = self.current_room_node.get_node("ResourceSpawners").get_children()
		for i in range(len(spawners)):
			if room_info.resources[i] == null:
				continue
			var spawner = spawners[i]
			mapped_resources[spawner.name] = room_info.resources[i]
		room_info.resources = mapped_resources

	var monster_spawners = self.current_room_node.get_node("MonsterSpawners").get_children()
	var resource_spawners = self.current_room_node.get_node("ResourceSpawners").get_children()
	var wall = self.current_room_node.get_node("Wall")
	for spawner in monster_spawners:
		if not room_info.monsters.has(spawner.name):
			continue
		var monster_info = room_info.monsters[spawner.name]
		var monster = monster_info.type.instance()
		monster_info.instance = monster
		monster.position = spawner.position
		monster.add_to_group("team2")
		monster.connect("character_death", self, "_on_monster_death", [monster, spawner])
		wall.add_child(monster)
		self._apply_attributes(monster, monster_info.attributes)
	
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

func change_room(exit_id):
	var room_info = self.rooms_info[self.current_room_id]
	for monster_info in room_info.monsters.values():
		self._save_attributes(monster_info.instance, monster_info.attributes)
		monster_info.erase("instance")
	for resource_info in room_info.resources.values():
		self._save_attributes(resource_info.instance, resource_info.attributes)
		resource_info.erase("instance")
	var player = global.find_entity("player")
	self._save_attributes(player, self.player_attributes)

	var current_room_exits = self.rooms_info[self.current_room_id].exits
	var next_room_id = current_room_exits[exit_id].room
	if next_room_id == -1:
		self.back_to_city()
		return
	var entrance = current_room_exits[exit_id].entrance
	self.current_room_node.queue_free()
	yield(self.current_room_node, "tree_exited")
	self.create_room(next_room_id, entrance)

func _on_monster_death(monster, spawner):
	# self.generator.monster_died(...)
	monster.queue_free()
	self.rooms_info[self.current_room_id].monsters.erase(spawner.name)

func _on_resource_collected(resource, spawner):
	self.rooms_info[self.current_room_id].resources.erase(spawner.name)

func _on_player_death():
	self.back_to_city()