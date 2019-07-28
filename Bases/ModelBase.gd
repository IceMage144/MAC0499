extends Node

const SAVE_PATH = "user://save_data.json"
const model = null
const tag = ""

var stored_data = {}

func _ready():
	# Assert that the inherited class sets an unique tag
	assert(self.tag != "")
	# Assert that the inherited class sets a model
	assert(self.model != null)
	self._load_data()
	for key in self.model:
		var data_schema = self.model[key]
		if data_schema.has("default") and not self.stored_data.has(key):
			print("Overwrited data: " + str(key) + " with " + str(data_schema["default"]))
			self.stored_data[key] = data_schema["default"]

func _save_data():
	var save_file = File.new()
	var save_data = {}
	
	save_file.open(SAVE_PATH, File.WRITE_READ)
	var save_text = save_file.get_as_text()
	if save_text != "": # Save exists
		var parse_res = JSON.parse(save_text)
		# Assert that the JSON is not corrupted
		assert(parse_res.error == OK)
		save_data = parse_res.result
	
	save_data[self.tag] = self.stored_data
	save_file.store_string(JSON.print(save_data))
	save_file.close()
	
	print("Saved: ", self.stored_data)


func _load_data():
	var save_file = File.new()
	var save_data = {}
	if save_file.file_exists(SAVE_PATH):
		save_file.open(SAVE_PATH, File.READ)
		var parse_res = JSON.parse(save_file.get_as_text())
		# Assert that the JSON is not corrupted
		assert(parse_res.error == OK)
		save_data = parse_res.result[self.tag]
		print("Loaded: " + str(save_data))
	save_file.close()
	
	self.stored_data = {}
	for key in save_data:
		self.stored_data[int(key)] = save_data[key]

func has_data(key):
	return self.stored_data.has(key)

func get_data(key):
	return self.stored_data[key]

func set_data(key, value):
	# Assert that key exists
	assert(self.model.has(key))
	# Assert that the value is the right type
	assert(self.model[key].type == typeof(value))
	self.stored_data[key] = value
	self._save_data()
