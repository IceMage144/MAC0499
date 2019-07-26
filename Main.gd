extends Node

const PauseMenu = preload("res://Menus/PauseMenu.tscn")
const InventoryPopup = preload("res://Popups/InventoryPopup.tscn")

enum Scene { MAIN_MENU, ROBOT_ROBOT, PLAYER_ROBOT }

const scene_path = {
	Scene.MAIN_MENU: "res://Menus/MainMenu.tscn",
	Scene.ROBOT_ROBOT: "res://Maps/Tests/RobotRobotArena.tscn",
	Scene.PLAYER_ROBOT: "res://Maps/Tests/PlayerRobotArena.tscn"
}

export(int, "Main menu", "Robot vs Robot test", "Player vs Robot test") var first_scene = Scene.MAIN_MENU
export(bool) var character_debug = false
export(bool) var environment_debug = false
export(bool) var popup_debug = false

var FirstSceneClass
var current_scene = null
var current_popup = null

func _ready():
	self.FirstSceneClass = load(scene_path[first_scene])
	self.reset_game()
	if first_scene != Scene.MAIN_MENU:
		assert(global.has_entity("team1"))
		assert(global.has_entity("team2"))
		for timer in get_tree().get_nodes_in_group("debug_timer"):
			assert(timer is Timer)
	if self.character_debug:
		for timer in get_tree().get_nodes_in_group("debug_timer"):
			timer.start()
	self._check_persistence_tags()

func _process(_delta):
	if Input.is_action_just_pressed("pause"):
		if self.current_popup != null:
			self.close_popup()
		else:
			var pause_menu = PauseMenu.instance()
			self.add_child(pause_menu)
			pause_menu.debug_mode = self.popup_debug
	if Input.is_action_just_pressed("rewind"):
		self.reset_game()
	if Input.is_action_just_pressed("bag"):
		if global.has_entity("player"):
			self.show_popup(InventoryPopup)
	if Input.is_action_just_pressed("test"):
		var shop = load("res://Popups/ShopPopup.tscn")
		self.show_popup(shop, "equip")

func _check_persistence_tags():
	var tag_memo = {}
	for node in get_tree().get_nodes_in_group("persistence"):
		# Assert that no two persistence tags are the same
		assert(not tag_memo.has(node.tag))
		tag_memo[node.tag] = true

func reset_game():
	self.change_map(self.FirstSceneClass)

func change_map(scene, params=null):
	if self.current_scene:
		self.current_scene.queue_free()
		yield(self.current_scene, "tree_exited")
	self.current_scene = scene.instance()
	self.add_child(self.current_scene)
	self.current_scene.init(params)
	self.current_scene.debug_mode = self.environment_debug

func show_popup(popup, params=null):
	if self.current_popup:
		self.current_popup.queue_free()
		yield(self.current_popup, "tree_exited")
	self.current_popup = popup.instance()
	self.add_child(self.current_popup)
	self.current_popup.init(params)
	self.current_popup.debug_mode = self.popup_debug
	if global.has_entity("player"):
		var player = global.find_entity("player")
		player.block_action()

func close_popup():
	self.current_popup.queue_free()
	self.current_popup = null
	if global.has_entity("player"):
		var player = global.find_entity("player")
		player.unblock_action()