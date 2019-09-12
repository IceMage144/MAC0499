extends "res://Maps/RoomBase.gd"

const CaveAdventure = preload("res://Maps/Adventures/Cave/CaveAdventure.tscn")

func _on_dungeon_interacted():
	var main = global.find_entity("main")
	main.change_map(CaveAdventure)
