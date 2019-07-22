extends "res://Bases/Map/PopupBase.gd"

onready var Inventory = $Content/CenterContainer/PanelContainer/HBoxContainer/Inventory
onready var PlayerInfo = $Content/CenterContainer/PanelContainer/HBoxContainer/PlayerInfo
onready var ItemInfo = $Content/CenterContainer/PanelContainer/HBoxContainer/ItemInfo

func _ready():
#	Get player items
#	Inventory.display_items(item_list)
	pass

func _on_nothing_selected():
	ItemInfo.remove_item()

func _on_item_selected(item):
	if item.is_in_group("equip"):
		ItemInfo.display_item(item, ItemInfo.EQUIP)
	elif item.is_in_group("consumable"):
		ItemInfo.display_item(item, ItemInfo.USE)

func _on_item_activated(item):
	if item.is_in_group("equip"):
		self._on_item_equiped(item)
	elif item.is_in_group("consumable"):
		self._on_item_used(item)

func _on_equip_selected(item):
	ItemInfo.display_item(item, ItemInfo.UNEQUIP)

func _on_item_used(item):
	# Use item
	pass

func _on_item_equiped(item):
	var old_item = PlayerInfo.equip_item(item)
	print("Equiped: " + item.name)
	# Update persisted player bag (change items)
	# Update Inventory (change items)

func _on_item_unequiped(item):
	if Inventory.is_full():
		return
	PlayerInfo.unequip_item(item)
	print("Unequiped: " + item.name)
	# Update persisted player bag (add unequiped item)
	# Update Inventory (add unequiped item)
