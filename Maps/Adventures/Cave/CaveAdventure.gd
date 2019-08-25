extends "res://Maps/Adventures/AdventureBase.gd"

const room_config = [
	{
		"type": preload("res://Maps/Adventures/Cave/CaveRoom1.tscn"),
		"exits": ["right", "top", "left", "bottom"],
		"mosters": 2,
		"resources": 0
	},
	{
		"type": preload("res://Maps/Adventures/Cave/CaveRoom2.tscn"),
		"exits": ["right", "top", "left", "bottom"],
		"mosters": 2,
		"resources": 1
	},
	{
		"type": preload("res://Maps/Adventures/Cave/CaveRoom3.tscn"),
		"exits": ["right", "top", "left", "bottom"],
		"mosters": 0,
		"resources": 3
	}
]

const monster_config = [
	preload("res://Characters/Goblin/Goblin.tscn"),
	preload("res://Characters/Spider/Spider.tscn")
]

const resource_config = [
	{
		"type": preload("res://Interactives/Chest/Chest.tscn"),
		"attributes": {
			"amount": {
				"type": "number",
				"minimum": 100,
				"maximum": 300
			}
		}
	}
]
