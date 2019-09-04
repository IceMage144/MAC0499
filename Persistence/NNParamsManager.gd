extends "res://Persistence/ModelBase.gd"

const PARAMS = "params"

var tag = "ai"
var model = {
	PARAMS: {
		"type": TYPE_DICTIONARY,
		"default": {}
	}
}

func get_params(key):
	var params_dict = self.get_data(PARAMS)
	if not params_dict.has(key):
		return null
	return params_dict[key]

func set_params(key, value):
	var params_dict = self.get_data(PARAMS)
	params_dict[key] = value
	self.set_data(PARAMS, params_dict)