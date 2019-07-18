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