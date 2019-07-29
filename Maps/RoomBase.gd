extends Node2D

const Hud = preload("res://UI/HUD/Hud.tscn")

var debug_mode = false

func _ready():
	if global.has_entity("camera"):
		var camera = global.find_entity("camera")
		camera.limit_left = $CameraLimits/TopLeftCorner.position.x
		camera.limit_top = $CameraLimits/TopLeftCorner.position.y
		camera.limit_right = $CameraLimits/BottomRightCorner.position.x
		camera.limit_bottom = $CameraLimits/BottomRightCorner.position.y
	if global.has_entity("player"):
		self.add_child_below_node($Ceil, Hud.instance())

func init(params):
	pass