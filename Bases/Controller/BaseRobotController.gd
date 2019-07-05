extends Node

const AINode = preload("res://AIs/AI.tscn")

enum Feature { ENEMY_DIST, SELF_LIFE, ENEMY_LIFE, ENEMY_ATTACKING, ENEMY_DIR_X, ENEMY_DIR_Y, BIAS }
enum AiType { BERKELEY, TORCH, MEMO, CLASSIFIER }

const Direction = {
	"LEFT": "left",
	"RIGHT": "right",
	"UP": "up",
	"DOWN": "down"
}

# Movements
const DEATH = "death"
const IDLE = "idle"

const FEATURES_SIZE = Feature.BIAS + 1

const DIR_TO_VEC = {
	Direction.UP: Vector2(0, -1),
	Direction.RIGHT: Vector2(1, 0),
	Direction.DOWN: Vector2(0, 1),
	Direction.LEFT: Vector2(-1, 0)
}

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
var color
var parent
var velocity = Vector2()

func _is_aligned(dir, vec):
	# TODO: Receive vector of dir (change "==" to "in" ?)
	if vec.x < vec.y:
		if -vec.x < vec.y:
			return dir == Direction.DOWN
		return dir == Direction.LEFT
	if -vec.x > vec.y:
		return dir == Direction.UP
	return dir == Direction.RIGHT

func _ready():
	var glob = self.get_node("/root/global")

	self.parent = self.get_parent()
	self.enemy = glob.get_enemy(self.parent)
	self.tm = glob.find_entity("tile_map")
	$ThinkTimer.wait_time = self.parent.think_time

func init(params):
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
		"think_time": params["think_time"],
		"features_size": FEATURES_SIZE,
		"initial_state": self.get_state(),
		"initial_action": [IDLE, null]
	})
	$DebugTimer.connect("timeout", self.ai, "_on_DebugTimer_timeout")
	$ThinkTimer.start()
	self.add_child(self.ai)

func get_loss():
	return self.ai.get_loss()

func get_state():
	return {
		"self_pos": self.parent.position,
		"self_life": self.parent.life,
		"self_maxlife": self.parent.max_life,
		"self_damage": self.parent.damage,
		"self_mov": self.parent.movement,
		"self_dir": self.parent.direction,
		"enemy_pos": self.enemy.position,
		"enemy_life": self.enemy.life,
		"enemy_maxlife": self.enemy.max_life,
		"enemy_damage": self.enemy.damage,
		"enemy_mov": self.enemy.movement,
		"enemy_dir": self.enemy.direction
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
	return self.parent.is_process_movement(self.parent.movement)

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
