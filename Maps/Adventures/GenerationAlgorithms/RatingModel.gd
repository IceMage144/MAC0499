extends "res://Persistence/ModelBase.gd"

const RATINGS = "ratings"

func _get_tag():
	return "rating_generator"

func _get_model():
	return {
		RATINGS: {
			"type": TYPE_DICTIONARY,
			"default": {}
		}
	}