extends CanvasLayer

func _ready():
	get_tree().paused = true

func _process(_delta):
	if Input.is_action_just_pressed("pause"):
		get_tree().paused = false
		queue_free()

func _on_SaveArch_pressed():
	global.save_info(get_tree().get_nodes_in_group("has_arch"))
