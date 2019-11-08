extends "res://Persistence/ModelBase.gd"

const RATINGS = "ratings"

func get_tag():
	return "rating_generator"

func get_model():
	return {
		RATINGS: {
			"type": TYPE_DICTIONARY,
			"default": {}
		}
	}