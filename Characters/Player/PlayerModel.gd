extends "res://Bases/ModelBase.gd"

enum Field { MONEY, BAG, QUICK, SWORD }

const tag = "player"
const model = {
	Field.MONEY: {
		"type": TYPE_INT,
		"default": 0
	},
	Field.BAG: {
		"type": TYPE_ARRAY,
		"default": []
	},
	Field.QUICK: {
		"type": TYPE_ARRAY,
		"default": []
	},
	Field.SWORD: {
		"type": TYPE_STRING,
		"default": ""
	}
}
