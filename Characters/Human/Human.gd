extends "res://Bases/Character/CharacterBase.gd"

var already_hit = []

func _ready():
	self.anim_node.play(Action.to_string(self.action))

func _process(delta):
	var mov = Action.get_movement(self.action)
	if mov == Action.ATTACK or mov == Action.DEATH:
		self.velocity = Vector2()
	else:
		self.velocity = self.controller.velocity
		if not self.velocity:
			self.action = Action.compose(Action.IDLE, self.action)
		else:
			if self.velocity.x < self.velocity.y:
				if -self.velocity.x < self.velocity.y:
					self.action = Action.compose(Action.WALK, Action.DOWN)
				else:
					self.action = Action.compose(Action.WALK, Action.LEFT)
			else:
				if -self.velocity.x > self.velocity.y:
					self.action = Action.compose(Action.WALK, Action.UP)
				else:
					self.action = Action.compose(Action.WALK, Action.RIGHT)
	
	var current_anim = self.anim_node.current_animation
	if not Action.bits_are_equal_string(self.action, current_anim):
		self.anim_node.play(Action.to_string(self.action))

func attack():
	self.set_movement(Action.ATTACK)

func is_process_action(a):
	return Action.get_movement(a) == Action.WALK or .is_process_action(a)

func _on_AnimationPlayer_animation_finished(anim_name):
	var attack = Action.to_string(Action.ATTACK)
	if anim_name.begins_with(attack):
		self.set_movement(Action.IDLE)
		self.already_hit = []
	else:
		._on_AnimationPlayer_animation_finished(anim_name)

func _on_AttackArea_area_entered(area):
	var entity = area.get_parent()
	if entity.is_in_group("damageble") and entity != self and \
	   not (entity in self.already_hit) and Action.get_movement(self.action) == Action.ATTACK and \
	   (entity.position - self.position).dot(Action.to_vec(self.action)) >= 0:
		entity.take_damage(self.damage)
		self.already_hit.append(entity)