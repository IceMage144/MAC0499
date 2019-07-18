extends Node2D

enum Direction { RIGHT, UP, LEFT, DOWN }

const DIR_TO_FLIP = [
	# HFlip, VFlip, Rotation
	[false, false, 0],
	[false, false, PI/2],
	[true, false, 0],
	[false, true, PI/2]
]

export(int, "Right", "Up", "Left", "Down") var direction = Direction.RIGHT
export(PackedScene) var destination

func _ready():
	match self.direction:
		Direction.UP:
			self.rotation = 3 * PI / 2
		Direction.LEFT:
			self.rotation = PI
			$Sprite.flip_v = true
		Direction.DOWN:
			self.rotation = PI / 2
			$Sprite.flip_v = true

func _on_Area2D_body_entered(body):
	if body.is_in_group("player"):
		var main = global.find_entity("main")
		main.change_map(self.destination)
