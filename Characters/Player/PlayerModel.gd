extends "res://Persistence/ModelBase.gd"

const MONEY = "money"
const BAG = "bag"
const QUICK = "quick"
const SWORD = "sword"

func get_tag():
	return "player"

func get_model():
	return {
		MONEY: {
			"type": TYPE_INT,
			"default": 0
		},
		BAG: {
			"type": TYPE_ARRAY,
			"default": []
		},
		QUICK: {
			"type": TYPE_ARRAY,
			"default": []
		},
		SWORD: {
			"type": TYPE_STRING,
			"default": ""
		}
	}
