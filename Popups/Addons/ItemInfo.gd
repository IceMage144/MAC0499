extends PanelContainer

signal item_bought(item)
signal item_sold(item)
signal item_used(item)
signal item_equiped(item)
signal item_unequiped(item)

enum DisplayMode { NONE, BUY, SELL, USE, EQUIP, UNEQUIP }

const MODE_TO_STR = ["", "Buy", "Sell", "Use", "Equip", "Unequip"]
const MODE_TO_PAST = ["", "bought", "sold", "used", "equiped", "unequiped"]

var current_display_mode = DisplayMode.NONE
var current_item = null

func display_item(item, display_mode=DisplayMode.NONE):
	$MarginContainer.visible = true
	$MarginContainer/VBoxContainer/Name.text = item.name
	$MarginContainer/VBoxContainer/Icon.texture = item.icon
	$MarginContainer/VBoxContainer/Description.text = item.description
	$MarginContainer/VBoxContainer/ActionButton.visible = (display_mode != DisplayMode.NONE)
	$MarginContainer/VBoxContainer/ActionButton.text = MODE_TO_STR[display_mode]
	if display_mode == DisplayMode.BUY:
		$MarginContainer/VBoxContainer/Price/PriceLabel.text = str(item.price)
		$MarginContainer/VBoxContainer/Price.visible = true
	elif display_mode == DisplayMode.SELL:
		$MarginContainer/VBoxContainer/Price/PriceLabel.text = str(item.price)
		$MarginContainer/VBoxContainer/Price.visible = true
	else:
		$MarginContainer/VBoxContainer/Price.visible = false
	self.current_display_mode = display_mode
	self.current_item = item

func remove_item():
	$MarginContainer.visible = false
	self.current_item = null
	self.current_display_mode = DisplayMode.NONE

func _on_ActionButton_pressed():
	assert(self.current_display_mode != DisplayMode.NONE)
	self.emit_signal("item_" + MODE_TO_PAST[self.current_display_mode], self.current_item)
