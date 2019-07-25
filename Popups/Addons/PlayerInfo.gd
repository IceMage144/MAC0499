extends PanelContainer

signal item_selected(item)
signal item_activated(item)

enum EquipPosition { sword }

var current_equips_list = []

onready var EquipSlots = $MarginContainer/VBoxContainer
onready var MoneyLabel = $MarginContainer/VBoxContainer/HBoxContainer/Money

func _ready():
	for key in EquipPosition.keys():
		self.current_equips_list.append(null)
		var slot = EquipSlots.get_node(key.capitalize() + "Slot")
		slot.connect("focus_entered", self, "_on_slot_selected", [EquipPosition[key]])
		slot.connect("pressed", self, "_on_slot_activated", [EquipPosition[key]])

func display_equips(equips_list):
	for key in EquipPosition.keys():
		var slot = EquipSlots.get_node(key.capitalize() + "Slot")
		var item = equips_list[key]
		self.current_equips_list[EquipPosition[key]] = item
		if item != null:
			slot.icon = item.icon
			slot.disabled = false
			slot.focus_mode = FOCUS_ALL
		else:
			slot.icon = null
			slot.disabled = true
			slot.focus_mode = FOCUS_NONE

func display_money(ammount):
	MoneyLabel.text = str(ammount)

func _on_slot_selected(index):
	if self.current_equips_list[index] != null:
		self.emit_signal("item_selected", self.current_equips_list[index])

func _on_slot_activated(index):
	if self.current_equips_list[index] != null:
		self.emit_signal("item_activated", self.current_equips_list[index])
