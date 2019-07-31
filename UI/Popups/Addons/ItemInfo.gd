extends PanelContainer

signal item_bought(item)
signal item_sold(item)
signal item_used(item)
signal item_equiped(item)
signal item_unequiped(item)

enum DisplayMode { NONE, BUY, SELL, USE, EQUIP, UNEQUIP }

const MODE_TO_STR = ["", "Buy", "Sell", "Use", "Equip", "Unequip"]
const MODE_TO_PAST = ["", "bought", "sold", "used", "equiped", "unequiped"]

export(NodePath) var NamePath
export(NodePath) var IconPath
export(NodePath) var DescriptionPath
export(NodePath) var ActionButtonPath
export(NodePath) var PricePath

var current_display_mode = DisplayMode.NONE
var current_item = null

onready var Name = get_node(NamePath)
onready var Icon = get_node(IconPath)
onready var Description = get_node(DescriptionPath)
onready var ActionButton = get_node(ActionButtonPath)
onready var Price = get_node(PricePath)

func display_item(item, display_mode=DisplayMode.NONE):
	$MarginContainer.visible = true
	Name.text = item.name
	Icon.texture = item.icon
	Description.text = item.description
	ActionButton.visible = (display_mode != DisplayMode.NONE)
	ActionButton.text = MODE_TO_STR[display_mode]
	if display_mode == DisplayMode.BUY:
		Price.get_node("PriceLabel").text = str(item.price)
		Price.visible = true
	elif display_mode == DisplayMode.SELL:
		Price.get_node("PriceLabel").text = str(item.sell_price)
		Price.visible = true
	else:
		Price.visible = false
	self.current_display_mode = display_mode
	self.current_item = item

func remove_item():
	$MarginContainer.visible = false
	self.current_item = null
	self.current_display_mode = DisplayMode.NONE

func _on_ActionButton_pressed():
	assert(self.current_display_mode != DisplayMode.NONE)
	self.emit_signal("item_" + MODE_TO_PAST[self.current_display_mode], self.current_item)
