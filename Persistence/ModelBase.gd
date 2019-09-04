extends Node

const model = null
const tag = ""

var cached_data = {}

func _ready():
	# Assert that the inherited class sets an unique tag
	assert(self.tag != "")
	# Assert that the inherited class sets a model
	assert(self.model != null)
	self.cached_data = SaveManager.load_data(self.tag)
	print(self.tag + " loaded " + self._print_debug(self.cached_data))
	for key in self.model:
		var data_schema = self.model[key]
		if data_schema.has("default") and not self.cached_data.has(key):
			print("Overwrited data: " + str(key) + " with " + str(data_schema["default"]))
			self.cached_data[key] = data_schema["default"]

func _print_debug(value):
	if typeof(value) == TYPE_STRING:
		if value.length() > 30:
			return str(value.length()) + " characters long string"
		return value
	elif typeof(value) == TYPE_DICTIONARY:
		if value.size() > 17:
			return str(value.size()) + " keys dictionary"
		var res = {}
		for key in value.keys():
			res[key] = self._print_debug(value[key])
		return str(res)
	elif typeof(value) == TYPE_ARRAY:
		if value.size() > 20:
			return str(value.size()) + " elements array"
		var res = []
		for el in value:
			res.append(self._print_debug(el))
		return str(res)
	return str(value)

func has_data(key):
	return self.cached_data.has(key)

func get_data(key):
	return self.cached_data[key]

func set_data(key, value):
	# Assert that key exists
	assert(self.model.has(key))
	# Assert that the value is the right type
	assert(self.model[key].type == typeof(value))
	self.cached_data[key] = value
	SaveManager.save_data(self.tag, self.cached_data)
	print(self.tag + " saved " + self._print_debug(self.cached_data))
