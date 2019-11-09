extends Object

signal room_cleared(room)
signal room_dropped(room)

var type
var num_enemies
var num_resources
var enemies
var resources
var player
var start_time
var time
var exits
var instance
var initialized

func _init(type, exits, num_enemies, enemies, num_resources, resources):
	# Assert enemy list isn't bigger than supported number of enemies
	assert(len(enemies) <= num_enemies)
	# Assert resource list isn't bigger than supported number of resources
	assert(len(resources) <= num_resources)
	self.type = type
	self.time = 0.0
	self.instance = null
	self.initialized = false

	self.exits = {}
	for exit in exits:
		self.exits[exit] = null

	self.num_enemies = num_enemies
	self.enemies = enemies
	while len(self.enemies) < self.num_enemies:
		self.enemies.append(null)
	global.shuffle_array(self.enemies)
	
	self.num_resources = num_resources
	self.resources = resources
	while len(self.resources) < self.num_enemies:
		self.resources.append(null)
	global.shuffle_array(self.resources)

func connect_to_room(other_room, exit):
	# Assert exit exists
	assert(self.has_exit(exit))
	# Assert exit is not connected
	assert(not self.has_connection(exit))
	self.exits[exit] = other_room

func has_connection(exit):
	return self.exits[exit] != null

func get_connected(exit):
	return self.exits[exit]

func get_enemies():
	return self.enemies.values()

func has_exit(exit):
	return self.exits.has(exit)

func get_exits():
	return self.exits.values()

func remove_exit(exit):
	return self.exits.erase(exit)

func num_alive_enemies():
	var num_alive = 0
	for enemy in self.enemies.values():
		if not enemy.is_dead():
			num_alive += 1
	return num_alive

func get_instance(player_pos):
	if self.instance:
		return self.instance
	self.instance = self.type.instance()
	if not self.initialized:
		self._continue_init()
	var wall = self.instance.get_node("Wall")
	var spawners = self.instance.get_node("MonsterSpawners")
	for spawner_name in self.enemies.keys:
		var enemy_instance = self.enemies[spawner_name].get_instance()
		var spawner = spawners.get_child(spawner_name)
		enemy_instance.position = spawner.position
		enemy_instance.connect("enemy_death", self, "_on_enemy_death")
		wall.add_child(enemy_instance)
	spawners = self.instance.get_node("ResourceSpawners").get_children()
	for spawner_name in self.resources:
		var resource_instance = self.resources[spawner_name].get_instance()
		var spawner = spawners.get_child(spawner_name)
		resource_instance.position = spawner.position
		wall.add_child(resource_instance)
	
	var player = self.player.get_instance()
	player.connect("character_death", self, "_on_player_death")

	var instance_params = {
		"player_pos": player_pos,
		"available_entrances": self.exits.keys()
	}
	self.instance.init(instance_params)
	self.start_time = OS.get_system_time_secs()
	return self.instance

func kill_instance():
	# Assert an instance exists
	assert(self.instance != null)
	for enemy in self.enemies.values():
		if enemy:
			enemy.kill_instance()
	for resource in self.resources.values():
		if resource:
			resource.kill_instance()
	self.player.kill_instance()
	self.instance.queue_free()
	yield(self.instance, "tree_exited")
	self.instance = null
	self.time = OS.get_system_time_secs() - self.start_time

func _continue_init():
	var enemies_map = {}
	var spawners = self.instance.get_node("MonsterSpawners").get_children()
	for i in range(len(spawners)):
		var spawner = spawners[i]
		if self.enemies[i] != null:
			enemies_map[spawner.name] = self.enemies[i]
	self.enemies = enemies_map

	var resources_map = {}
	spawners = self.instance.get_node("ResourceSpawners").get_children()
	for i in range(len(spawners)):
		var spawner = spawners[i]
		if self.resources[i] != null:
			resources_map[spawner.name] = self.resources[i]
	self.resources = resources_map

func _on_enemy_death():
	if self.num_alive_enemies() == 0:
		self.time += OS.get_system_time_secs() - self.start_time
		self.emit_signal("room_cleared", self)

func _on_player_death():
	self.time += OS.get_system_time_secs() - self.start_time
	self.emit_signal("room_dropped", self)

func _on_TickTimer_timeout(timer):
	self.time += 1