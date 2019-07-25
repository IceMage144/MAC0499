extends "res://Bases/Map/PopupBase.gd"

onready var Shelves = $Content/CenterContainer/PanelContainer/HBoxContainer/Shelves
onready var ItemInfo = $Content/CenterContainer/PanelContainer/HBoxContainer/ItemInfo
onready var player = global.find_entity("player")

func init(item_group):
	var item_list = ItemDB.get_items_in_group(item_group)
	Shelves.display_items(item_list)

func _on_nothing_selected():
	ItemInfo.remove_item()

func _on_item_selected(item):
	ItemInfo.display_item(item, ItemInfo.BUY)

func _on_item_bought(item):
	player.buy_item(item)

