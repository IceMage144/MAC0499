extends "res://Bases/Model/ModelBase.gd"

enum Field { MONEY, BAG, SWORD }

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
	Field.SWORD: {
		"type": TYPE_STRING,
		"default": ""
	}
}
