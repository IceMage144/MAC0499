extends "res://Bases/Map/MapBase.gd"

func _ready():
	if global.has_entity("camera"):
		var camera = global.find_entity("camera")
		camera.limit_left = $CameraLimits/TopLeftCorner.position.x
		camera.limit_top = $CameraLimits/TopLeftCorner.position.y
		camera.limit_right = $CameraLimits/BottomRightCorner.position.x
		camera.limit_bottom = $CameraLimits/BottomRightCorner.position.y