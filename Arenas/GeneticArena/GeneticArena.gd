extends YSort

var initial_positions = {}
var arena_width = 29 * 32
var arena_height = 15 * 32

func _ready():
	for character in get_tree().get_nodes_in_group("character"):
		initial_positions[character.name] = character.position
		character.connect("character_death", self, "_on_character_death")

func print_info():
	var glob = get_node("/root/global")
	print("------------")
	for character in get_tree().get_nodes_in_group("character"):
		var team = glob.get_team(character)
		print(character.name + ": " + str(character.life) + " (" + team + ")")

func reset():
	print_info()
	# get_parent().reset_arena()
	for character in get_tree().get_nodes_in_group("character"):
		character.position = Vector2(64 + arena_width*randf(), 64 + arena_height*randf())
		character.reset()


func _on_character_death():
	self.get_node("TimeoutTimer").start()
	self.reset()

func _on_TimeoutTimer_timeout():
	self.reset()
