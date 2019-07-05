extends Node

const PauseMenu = preload("res://Menus/PauseMenu.tscn")

enum Screen { DEFAULT, ROBOT_ROBOT, PLAYER_ROBOT }

const screen_path = {
	Screen.DEFAULT: "res://Arenas/Arena.tscn",
	Screen.ROBOT_ROBOT: "res://Arenas/GeneticArena/GeneticArena.tscn",
	Screen.PLAYER_ROBOT: "res://Arenas/GeneticArena/GeneticArenaPlayer.tscn"
}

export(int, "Default", "Robot vs Robot test", "Player vs Robot test") var first_screen = Screen.DEFAULT
export(bool) var character_debug = false
export(bool) var arena_debug = false

var ScreenClass
var arena

func _ready():
	var global = get_node("/root/global")
	ScreenClass = load(screen_path[first_screen])
	reset_arena()
	assert(global.has_entity("team1"))
	assert(global.has_entity("team2"))
	for timer in get_tree().get_nodes_in_group("debug_timer"):
		assert(timer is Timer)
	if self.character_debug:
		for timer in get_tree().get_nodes_in_group("debug_timer"):
			timer.start()

func _process(_delta):
	if Input.is_action_just_pressed("pause"):
		add_child(PauseMenu.instance())
	if Input.is_action_just_pressed("rewind"):
		reset_arena()

func reset_arena():
	if self.arena:
		self.arena.queue_free()
		yield(self.arena, "tree_exited")
	self.arena = ScreenClass.instance()
	add_child(self.arena)
	self.arena.debug_mode = self.arena_debug
	