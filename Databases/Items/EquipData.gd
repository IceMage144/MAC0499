extends "res://Databases/Items/ItemDataBase.gd"

export(int) var damage = 1
export(String, "sword") var type = "sword"

func _ready():
	self.add_to_group(self.type)