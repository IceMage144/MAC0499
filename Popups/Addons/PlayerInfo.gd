extends PanelContainer

signal item_selected(item)
signal item_activated(item)

enum EquipPosition { SWORD }

var equips_list = []

onready var EquipSlots = $MarginContainer/VBoxContainer

func _ready():
	for key in EquipPosition.keys():
		self.equips_list.append(null)
		var slot = EquipSlots.get_node(key.capitalize() + "Slot")
		slot.connect("focus_entered", self, "_on_slot_selected", [EquipPosition[key]])
		slot.connect("pressed", self, "_on_slot_activated", [EquipPosition[key]])
	# Update money value

func equip_item(item):
	for key in EquipPosition.keys():
		if item.is_in_group(key.to_lower()):
			var slot = EquipSlots.get_node(key.capitalize() + "Slot")
			var slot_id = EquipPosition[key]
			slot.icon = item.icon
			slot.disabled = false
			slot.focus_mode = FOCUS_ALL
			var old_item = self.equips_list[slot_id]
			self.equips_list[slot_id] = item
			return old_item

func unequip_item(item):
	for key in EquipPosition.keys():
		if item.is_in_group(key.to_lower()):
			var slot = EquipSlots.get_node(key.capitalize() + "Slot")
			var slot_id = EquipPosition[key]
			slot.icon = null
			slot.disabled = true
			slot.focus_mode = FOCUS_NONE
			self.equips_list[slot_id] = null

func _on_slot_selected(index):
	if self.equips_list[index] != null:
		self.emit_signal("item_selected", self.equips_list[index])

func _on_slot_activated(index):
	if self.equips_list[index] != null:
		self.emit_signal("item_activated", self.equips_list[index])
