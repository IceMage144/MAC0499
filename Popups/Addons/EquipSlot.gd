extends PanelContainer

signal nothing_selected
signal item_selected(item)
signal item_activated(item)

onready var ItemList = $MarginContainer/ItemList

var current_item = null

func display_equip(item):
	self.current_item = item
	self.ItemList.add_icon_item(item.icon)

func remove_equip():
	self.current_item = null
	self.ItemList.clear()

func _on_nothing_selected():
	self.emit_signal("nothing_selected")

func _on_item_selected(index):
	self.emit_signal("item_selected", self.current_item)

func _on_item_activated(index):
	self.emit_signal("item_activated", self.current_item)
