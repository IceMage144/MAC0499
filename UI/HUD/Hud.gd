extends CanvasLayer

const InventoryPopup = preload("res://UI/Popups/InventoryPopup.tscn")

export(NodePath) var QuickUseBarPath

onready var QuickUseBar = self.get_node(QuickUseBarPath)
onready var Player = global.find_entity("player")

func _ready():
	QuickUseBar.display_items(Player.get_quick_items())
	QuickUseBar.connect("triggered_slot", Player, "use_item")

func _on_BagButton_pressed():
	var main = global.find_entity("main")
	main.show_popup(InventoryPopup)
