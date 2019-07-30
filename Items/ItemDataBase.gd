extends Node

export(Texture) var icon
export(String, MULTILINE) var description = ""
export(int) var price = 1

func _ready():
	self.add_to_group("item")