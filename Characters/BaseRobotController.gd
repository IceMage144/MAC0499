extends Node

const AINode = preload("res://AIs/AI.tscn")
const ActionClass = preload("res://Characters/ActionBase.gd")

enum Feature { ENEMY_DIST, SELF_LIFE, ENEMY_LIFE, ENEMY_ATTACKING, ENEMY_DIR_X, ENEMY_DIR_Y, BIAS }
enum AiType { BERKELEY, TORCH, MEMO, CLASSIFIER }

const FEATURES_SIZE = Feature.BIAS + 1

const ai_path = {
	AiType.BERKELEY: "res://AIs/BerkeleyQLAI.py",
	AiType.TORCH: "res://AIs/TorchQLAI.py",
	AiType.MEMO: "res://AIs/MemoQLAI.py",
	AiType.CLASSIFIER: "res://AIs/ClassQLAI.py"
}

const ai_color = {
	AiType.BERKELEY: Color(0.2, 1.0, 0.2, 1.0),
	AiType.TORCH: Color(1.0, 0.2, 0.2, 1.0),
	AiType.MEMO: Color(0.2, 0.2, 1.0, 1.0),
	AiType.CLASSIFIER: Color(1.0, 0.2, 1.0, 1.0)
}

var ai
var enemy
var tm
var color = Color(1.0, 1.0, 1.0, 1.0)
var parent
var debug_mode = false
var velocity = Vector2()

onready var Action = ActionClass.new()

func _is_aligned(act, vec):
	# TODO: Move this function to another place
	var dir = Action.get_direction(act)
	if vec.x < vec.y:
		if -vec.x < vec.y:
			return dir == Action.DOWN
		return dir == Action.LEFT
	if -vec.x > vec.y:
		return dir == Action.UP
	return dir == Action.RIGHT

func _ready():
	self.parent = self.get_parent()
	self.enemy = global.get_enemy(self.parent)
	self.tm = global.find_entity("floor")
	$ThinkTimer.wait_time = self.parent.think_time

func init(params):
	if params.has("debug_mode"):
		self.debug_mode = params["debug_mode"]
	self.color = ai_color[params["ai_type"]]

	self.ai = AINode.instance()
	self.ai.set_script(load(ai_path[params["ai_type"]]))
	self.ai.init({
		"learning_rate": params["learning_rate"],
		"discount": params["discount"],
		"max_exploration_rate": params["max_exploration_rate"],
		"min_exploration_rate": params["min_exploration_rate"],
		"exploration_rate_decay_time": params["exploration_rate_decay_time"],
		"momentum": params["momentum"],
		"reuse_last_action_chance": params["reuse_last_action_chance"],
		"experience_replay": params["experience_replay"],
		"experience_pool_size": params["experience_pool_size"],
		"think_time": params["think_time"],
		"features_size": FEATURES_SIZE,
		"initial_state": self.get_state(),
		"initial_action": Action.IDLE,
		"character_type": params["character_type"],
		"network_id": params["network_id"],
		"debug_mode": self.debug_mode
	})
	$DebugTimer.connect("timeout", self.ai, "_on_DebugTimer_timeout")
	$ThinkTimer.start()
	self.add_child(self.ai)

func end():
	self.ai.end()

func get_loss():
	return self.ai.get_loss()

func get_state():
	return {
		"self_pos": self.parent.position,
		"self_life": self.parent.life,
		"self_maxlife": self.parent.max_life,
		"self_damage": self.parent.damage,
		"self_act": self.parent.action,
		"enemy_pos": self.enemy.position,
		"enemy_life": self.enemy.life,
		"enemy_maxlife": self.enemy.max_life,
		"enemy_damage": self.enemy.damage,
		"enemy_act": self.enemy.action
	}

# Abstract
func get_legal_actions(state):
	pass

# Abstract
func get_reward(last_state, new_state, timeout):
	pass

# Abstract
func get_features_after_action(state, action):
	pass

func can_think():
	# TODO: Is this the right way to do it?
	return self.parent.is_process_action(self.parent.action)

func before_reset(timeout):
	self.ai.update_state(true, timeout)

# Abstract
func reset(timeout):
	pass

func after_reset(timeout):
	self.ai.reset(timeout)

func _on_ThinkTimer_timeout():
	if self.can_think():
		self.ai.update_state(false, false)

# Print some variables for debug here
func _on_DebugTimer_timeout():
	print("======== " + self.name + " ========")
	self.ai._on_DebugTimer_timeout()
