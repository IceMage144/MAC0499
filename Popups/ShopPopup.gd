extends "res://Bases/Map/PopupBase.gd"

const ItemDisplay = preload("res://Popups/ItemDisplay.tscn")

const ITEM_PER_SHELF = 6

var item_list = []

onready var Shelves = $Content/MarginContainer/PanelContainer/HSplitContainer/MarginContainer/PanelContainer/Shelves
onready var Info = $Content/MarginContainer/PanelContainer/HSplitContainer/MarginContainer2/PanelContainer/Info

func init(item_group):
	self.item_list = ItemDB.get_items_in_group(item_group)
	var num_shelves = Shelves.get_child_count()
	assert(len(self.item_list) <= num_shelves * ITEM_PER_SHELF)
	for i in range(len(self.item_list)):
		var display = ItemDisplay.instance()
		display.icon = self.item_list[i].icon
		display.connect("pressed", self, "_on_item_pressed", [i])
		var shelf = Shelves.get_node("Shelf" + str(int(i / ITEM_PER_SHELF + 1)))
		shelf.add_child(display)
	$Content/MarginContainer/PanelContainer/HSplitContainer.split_offset = 290

func _on_item_pressed(index):
	Info.visible = true
	var item = self.item_list[index]
	Info.get_node("Name").text = item.name
	Info.get_node("Icon").texture = item.icon
	Info.get_node("Description").text = item.description
	Info.get_node("Price").text = "Price: " + str(item.price) + " gold"