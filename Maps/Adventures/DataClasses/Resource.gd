extends Object

var type
var attributes
var instance

func _init(type, attributes):
	self.type = type
	self.attributes = attributes
	self.instance = null

func get_instance():
	if self.instance:
		return self.instance
	self.instance = self.type.instance()
	self.instance.init(self.attributes)
	self.instance.connect("interacted", self, "_on_resource_collected")
	return self.instance

func kill_instance():
	if self.instance:
		self._save_attributes()
		self.instance.queue_free()
		self.instance = null

func _save_attributes():
	for key in self.attributes.keys():
		self.attributes[key] = self.instance[key]

func _on_resource_collected(new_state):
	self.state = new_state