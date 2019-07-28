extends PanelContainer

signal item_selected(item)
signal item_activated(item)

enum EquipPosition { sword }

var current_equips_list = []
var equip_slots_list = []

onready var EquipSlots = $MarginContainer/EquipSlots

func _ready():
	for key in EquipPosition.keys():
		self.current_equips_list.append(null)
		var slot = EquipSlots.get_node(key.capitalize() + "Slot")
		self.equip_slots_list.append(slot)
		slot.connect("item_selected", self, "_on_slot_selected")
		slot.connect("item_activated", self, "_on_slot_activated")

func display_equips(equips_list):
	for key in EquipPosition.keys():
		var slot = self.equip_slots_list[EquipPosition[key]]
		var item = equips_list[key]
		self.current_equips_list[EquipPosition[key]] = item
		slot.remove_equip()
		if item != null:
			slot.display_equip(item)

func _on_slot_selected(item):
	if item != null:
		self.emit_signal("item_selected", item)

func _on_slot_activated(item):
	if item != null:
		self.emit_signal("item_activated", item)
