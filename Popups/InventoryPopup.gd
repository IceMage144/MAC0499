extends "res://Bases/Map/PopupBase.gd"

onready var Inventory = $Content/CenterContainer/PanelContainer/VBoxContainer/HBoxContainer/Inventory
onready var EquipDisplay = $Content/CenterContainer/PanelContainer/VBoxContainer/HBoxContainer/EquipDisplay
onready var ItemInfo = $Content/CenterContainer/PanelContainer/VBoxContainer/HBoxContainer/ItemInfo
onready var MoneyDisplay = $Content/CenterContainer/PanelContainer/VBoxContainer/MoneyDisplay
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
	var item_pos = Inventory.ItemList.get_selected_items()[0]
	self.player.equip_item(item)
	if item_pos == Inventory.ItemList.get_item_count():
		item_pos -= 1
	self.update_view(item_pos)

func _on_item_unequiped(item):
	print("Unequiped: " + item.name)
	self.player.unequip_item(item)
	self.update_view()

func update_view(focus_index=-1):
	var bag = self.player.get_bag()
	Inventory.display_items(bag)
	EquipDisplay.display_equips(self.player.get_equips())
	MoneyDisplay.display_money(self.player.get_money())
	ItemInfo.remove_item()
	if focus_index >= 0 and focus_index < len(bag):
		Inventory.ItemList.select(focus_index)
		self._on_item_selected(bag[focus_index])
