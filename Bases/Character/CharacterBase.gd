extends KinematicBody2D

const PlayerController = preload("res://Characters/Player/PlayerController.gd")
const ControllerNode = preload("res://Bases/Controller/Controller.tscn")
const ActionClass = preload("res://Bases/ActionBase.gd")

signal character_death

enum Controller { PLAYER, AI }

const ai_name = ["Berkeley", "Torch", "Memo", "Class"]

export(int) var speed = 120
export(int) var max_life = 3
export(int) var damage = 1
export(int, "Player", "AI") var controller_type = Controller.PLAYER
export(int, "Berkeley", "Torch", "Memo", "Class") var ai_type = 0
export(float, 0.0, 1.0, 0.0001) var learning_rate = 0.0
export(float, 0.0, 1.0, 0.001) var discount = 0.0
export(float, 0.0, 1.0, 0.001) var max_exploration_rate = 1.0
export(float, 0.0, 1.0, 0.001) var min_exploration_rate = 0.0
export(float) var exploration_rate_decay_time = 0.0
export(float, 0.0, 1.0, 0.001) var momentum = 0.0
export(float, 0.0, 1.0, 0.01) var reuse_last_action_chance = 0.0
export(bool) var experience_replay = false
export(float) var think_time = 0.1

var velocity = Vector2()
var action = ActionClass.compose(ActionClass.IDLE, ActionClass.DOWN)
var life = max_life
var can_act = true
var controller
var controller_name

onready var anim_node = $Sprite/AnimationPlayer
onready var Action = ActionClass.new()

func _init_ai_controller():
	var AIControllerScript = self._get_ai_controller_script()
	self.controller.set_script(AIControllerScript)
	self.add_child(self.controller)
	self.controller.init({
		"ai_type": self.ai_type,
		"learning_rate": self.learning_rate,
		"discount": self.discount,
		"max_exploration_rate": self.max_exploration_rate,
		"min_exploration_rate": self.min_exploration_rate,
		"exploration_rate_decay_time": self.exploration_rate_decay_time,
		"momentum": self.momentum,
		"reuse_last_action_chance": self.reuse_last_action_chance,
		"experience_replay": self.experience_replay,
		"think_time": self.think_time
	})
	$Sprite.modulate = self.controller.color

func _get_ai_controller_script():
	var filepath = self.get_script().get_path().get_basename()
	return load(filepath + "RobotController.gd")

func _ready():
	self.set_life(max_life)
	self.controller = ControllerNode.instance()

	match self.controller_type:
		Controller.PLAYER:
			self.controller.set_script(PlayerController)
			self.add_child(self.controller)
			self.controller_name = "Player"
			self.add_to_group("player")
		Controller.AI:
			self._init_ai_controller()
			self.controller_name = ai_name[self.ai_type]
			self.add_to_group("robot")

func _physics_process(delta):
	self.move_and_slide(self.speed * self.velocity)

func get_pretty_name():
	return self.name + " (" + self.controller_name + ")"

func set_life(new_life):
	if new_life >= 0:
		self.life = new_life
	$LifeBar.value = self.life

func set_movement(new_movement, force=false):
	if (self.action != Action.DEATH and self.can_act) or force:
		self.action = Action.compose(new_movement, self.action)

func set_action(new_action, force=false):
	if (self.action != Action.DEATH and self.can_act) or force:
		self.action = new_action

func take_damage(damage):
	self.set_life(self.life - damage)
	if self.life <= 0:
		self.set_action(Action.DEATH)

func is_process_action(a):
	return Action.get_movement(a) == Action.IDLE

func die():
	self.emit_signal("character_death")

func block_action():
	self.can_act = false

func unblock_action():
	self.can_act = true

func before_reset(timeout):
	self.controller.before_reset(timeout)
	
func reset(timeout):
	self.set_life(self.max_life)
	self.set_action(Action.compose(Action.IDLE, Action.DOWN), true)
	self.controller.reset(timeout)
	
func after_reset(timeout):
	self.controller.after_reset(timeout)

func _on_AnimationPlayer_animation_finished(anim_name):
	var death = Action.to_string(Action.DEATH)
	if anim_name.begins_with(death):
		self.die()