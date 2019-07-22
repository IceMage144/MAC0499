extends PanelContainer

signal nothing_selected
signal item_selected(item)
signal item_activated(item)

const MAX_ITEMS = 36

var current_item_list = []

func display_items(item_list):
	assert(len(item_list) <= MAX_ITEMS)
	self.current_item_list = item_list
	$MarginContainer/ItemList.clear()
	for item in item_list:
		$MarginContainer/ItemList.add_icon_item(item.icon)

func is_full():
	return len(self.current_item_list) >= MAX_ITEMS

func _on_nothing_selected():
	self.emit_signal("nothing_selected")

func _on_item_selected(index):
	self.emit_signal("item_selected", self.current_item_list[index])

func _on_item_activated(index):
	self.emit_signal("item_activated", self.current_item_list[index])
