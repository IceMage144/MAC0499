extends Button

export(Texture) var default_background = null

var display_item = null

func _ready():
	if self.default_background != null:
		self.icon = self.default_background

func display_item(item):
	self.display_item = item
	self.icon = item.icon

func remove_item():
	self.display_item = null
	self.icon = self.default_background

func get_display_item():
	return self.display_item