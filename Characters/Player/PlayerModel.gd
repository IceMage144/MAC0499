extends "res://Persistence/ModelBase.gd"

const MONEY = "money"
const BAG = "bag"
const QUICK = "quick"
const SWORD = "sword"

const tag = "player"
const model = {
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
