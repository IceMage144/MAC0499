extends "res://Bases/Map/PopupBase.gd"

onready var Inventory = $Content/CenterContainer/PanelContainer/HBoxContainer/Inventory
onready var PlayerInfo = $Content/CenterContainer/PanelContainer/HBoxContainer/PlayerInfo
onready var ItemInfo = $Content/CenterContainer/PanelContainer/HBoxContainer/ItemInfo
onready var player = global.find_entity("player")

func _ready():
	self.update_view()

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
	print("Equip selected " + item.name)
	ItemInfo.display_item(item, ItemInfo.UNEQUIP)

func _on_item_used(item):
	# Use item
	pass

func _on_item_equiped(item):
	print("Equiped: " + item.name)
	self.player.equip_item(item)
	self.update_view()

func _on_item_unequiped(item):
	print("Unequiped: " + item.name)
	self.player.unequip_item(item)
	self.update_view()

func update_view():
	Inventory.display_items(self.player.get_bag())
	PlayerInfo.display_equips(self.player.get_equips())
	PlayerInfo.display_money(self.player.get_money())
	ItemInfo.remove_item()
