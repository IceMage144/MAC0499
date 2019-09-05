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
		"type": preload("res://Maps/Adventures/Cave/CaveRoom3.tscn"),
		"exits": ["right", "top", "left", "bottom"],
		"monsters": 0,
		"resources": 3
	}
]

const monster_config = [
	{
		"name": "Goblin",
		"type": preload("res://Characters/Goblin/Goblin.tscn"),
		"attributes": {
			"max_life": 3,
			"damage": 1,
			"defense": 0
		}
	},
	{
		"name": "Spider",
		"type": preload("res://Characters/Spider/Spider.tscn"),
		"attributes": {
			"max_life": 3,
			"damage": 1,
			"defense": 0
		}
	}
]

const resource_config = [
	{
		"type": preload("res://Interactives/Chest/Chest.tscn"),
		"attributes": {
			"amount": {
				"type": "integer",
				"minimum": 100,
				"maximum": 300
			}
		}
	}
]
