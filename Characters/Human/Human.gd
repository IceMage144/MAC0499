extends "res://Bases/Character/CharacterBase.gd"

# Movements
const WALK = "walk"
const ATTACK = "attack"

var already_hit = []

onready var has_direction = {
	DEATH: false,
	WALK: true,
	IDLE: true,
	ATTACK: true
}

func _ready():
	self.anim_node.play(self.movement + "_" + self.direction)

func _process(delta):
	if self.movement == ATTACK or self.movement == DEATH:
		self.velocity = Vector2()
	else:
		self.velocity = self.controller.velocity
		if not self.velocity:
			self.movement = IDLE
		else:
			self.movement = WALK
			if self.velocity.x < self.velocity.y:
				if -self.velocity.x < self.velocity.y:
					self.direction = Direction.DOWN
				else:
					self.direction = Direction.LEFT
			else:
				if -self.velocity.x > self.velocity.y:
					self.direction = Direction.UP
				else:
					self.direction = Direction.RIGHT
	
	if not has_direction[self.movement]:
		if not self.anim_node.current_animation.begins_with(self.movement):
			self.anim_node.play(self.movement)
	else:
		if not self.anim_node.current_animation.begins_with(self.movement) or \
		   not self.anim_node.current_animation.ends_with(self.direction):
			self.anim_node.play(self.movement + "_" + self.direction)

func attack():
	self.set_movement(ATTACK)

func is_process_movement(a):
	return a == WALK or .is_process_movement(a)

func _on_AnimationPlayer_animation_finished(anim_name):
	if anim_name.begins_with(ATTACK):
		self.set_movement(IDLE)
		self.already_hit = []
	else:
		._on_AnimationPlayer_animation_finished(anim_name)

func _on_AttackArea_area_entered(area):
	var entity = area.get_parent()
	if entity.is_in_group("damageble") and entity != self and \
	   not (entity in self.already_hit) and self.movement == ATTACK and \
	   (entity.position - self.position).dot(DIR_TO_VEC[self.direction]) >= 0:
		entity.take_damage(self.damage)
		self.already_hit.append(entity)