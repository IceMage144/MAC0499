extends YSort

var initial_positions = {}

func _ready():
	for character in get_tree().get_nodes_in_group("character"):
		initial_positions[character.name] = character.position
		character.connect("character_death", self, "_on_character_death", [character])

func _on_character_death(character):
	character.position = initial_positions[character.name]
	character.reset()