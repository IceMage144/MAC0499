extends "res://Maps/Adventures/AdventureBase.gd"

const room_config = [
	{
		"type": preload("res://Maps/Adventures/Cave/CaveRoom1.tscn"),
		"exits": ["right", "top", "left", "bottom"],
		"monsters": 2,
		"resources": 0
	},
	{
		"type": preload("res://Maps/Adventures/Cave/CaveRoom2.tscn"),
		"exits": ["right", "top", "left", "bottom"],
		"monsters": 2,
		"resources": 1
	},
	{
		"type": preload("res://Maps/Adventures/Cave/CaveRoom4.tscn"),
		"exits": ["right", "top", "left", "bottom"],
		"monsters": 2,
		"resources": 0
	}
#	{
#		"type": preload("res://Maps/Adventures/Cave/CaveRoom3.tscn"),
#		"exits": ["right", "top", "left", "bottom"],
#		"monsters": 0,
#		"resources": 3
#	}
]

const monster_config = [
	"Goblin",
	"Spider"
]

const resource_config = [
	{
		"type": preload("res://Interactives/Chest/Chest.tscn"),
		"attributes": {
			"amount": {
				"type": TYPE_INT,
				"minimum": 10,
				"maximum": 30
			}
		}
	}
]