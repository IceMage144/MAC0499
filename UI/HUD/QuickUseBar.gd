extends PanelContainer

signal triggered_slot(index)
signal selected_slot(index)

const SLOT_NUMBER = 4

var current_items = []
var edit_mode = false

onready var ItemList = $MarginContainer/ItemList

func _process(delta):
	if not self.edit_mode:
		for i in range(SLOT_NUMBER):
			if Input.is_action_just_pressed("quick_slot" + str(i + 1)) and i < len(self.current_items):
				self.emit_signal("triggered_slot", self.current_items[i])

func display_items(item_list):
	assert(len(item_list) <= SLOT_NUMBER)
	self.current_items = item_list
	self.ItemList.clear()
	for item in item_list:
		self.ItemList.add_icon_item(item.icon)

func _on_item_selected(index):
	self.emit_signal("selected_slot", self.current_items[index])
