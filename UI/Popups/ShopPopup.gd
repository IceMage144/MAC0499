extends "res://UI/Popups/PopupBase.gd"

export(NodePath) var ShelvesPath
export(NodePath) var ItemInfoPath
export(NodePath) var MoneyDisplayPath

onready var Shelves = self.get_node(ShelvesPath)
onready var ItemInfo = self.get_node(ItemInfoPath)
onready var MoneyDisplay = self.get_node(MoneyDisplayPath)
onready var Player = global.find_entity("player")

func init(item_group):
	var item_list = ItemDB.get_items_in_group(item_group)
	Shelves.display_items(item_list)
	MoneyDisplay.display_money(Player.get_money())

func _on_nothing_selected():
	ItemInfo.remove_item()

func _on_item_selected(item):
	ItemInfo.display_item(item, ItemInfo.BUY)

func _on_item_bought(item):
	Player.buy_item(item)
	MoneyDisplay.display_money(Player.get_money())

