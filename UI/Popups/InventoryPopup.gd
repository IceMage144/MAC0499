extends "res://UI/Popups/PopupBase.gd"

export(NodePath) var InventoryPath
export(NodePath) var EquipDisplayPath
export(NodePath) var ItemInfoPath
export(NodePath) var MoneyDisplayPath
export(NodePath) var ContentContainerPath

onready var Inventory = self.get_node(InventoryPath)
onready var EquipDisplay = self.get_node(EquipDisplayPath)
onready var ItemInfo = self.get_node(ItemInfoPath)
onready var MoneyDisplay = self.get_node(MoneyDisplayPath)
onready var ContentContainer = self.get_node(ContentContainerPath)
onready var QuickUseBar = global.find_entity("quick_use_bar")
onready var Hud = global.find_entity("hud")
onready var Player = global.find_entity("player")
onready var quick_bar_old_pos = QuickUseBar.rect_position
onready var quick_bar_parent = QuickUseBar.get_parent()

func _ready():
	quick_bar_parent.remove_child(QuickUseBar)
	ContentContainer.add_child(QuickUseBar)
	QuickUseBar.connect("selected_slot", self, "_on_quick_item_selected")
	QuickUseBar.edit_mode = true
	self.update_view()
	
func _exit_tree():
	ContentContainer.remove_child(QuickUseBar)
	quick_bar_parent.add_child(QuickUseBar)
	QuickUseBar.disconnect("selected_slot", self, "_on_quick_item_selected")
	QuickUseBar.edit_mode = false
	QuickUseBar.rect_position = self.quick_bar_old_pos

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

func _on_item_rmb_selected(item):
	if item.is_in_group("consumable"):
		Player.add_quick_item(item)
		QuickUseBar.display_items(Player.get_quick_items())

func _on_equip_selected(item):
	ItemInfo.display_item(item, ItemInfo.UNEQUIP)

func _on_item_used(item):
	var item_pos = Inventory.ItemList.get_selected_items()[0]
	Player.use_item(item)
	if item_pos == Inventory.ItemList.get_item_count():
		item_pos -= 1
	self.update_view(item_pos)

func _on_item_equiped(item):
	var item_pos = Inventory.ItemList.get_selected_items()[0]
	Player.equip_item(item)
	if item_pos == Inventory.ItemList.get_item_count():
		item_pos -= 1
	self.update_view(item_pos)

func _on_item_unequiped(item):
	Player.unequip_item(item)
	self.update_view()

func _on_quick_item_selected(item):
	Player.remove_quick_item(item)
	QuickUseBar.display_items(Player.get_quick_items())

func update_view(focus_index=-1):
	var bag = Player.get_bag()
	Inventory.display_items(bag)
	EquipDisplay.display_equips(Player.get_equips())
	MoneyDisplay.display_money(Player.get_money())
	ItemInfo.remove_item()
	if focus_index >= 0 and focus_index < len(bag):
		Inventory.ItemList.select(focus_index)
		self._on_item_selected(bag[focus_index])
