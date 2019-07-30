extends "res://Interactives/InteractiveBase.gd"

const ShopPopup = preload("res://UI/Popups/ShopPopup.tscn")

enum ShopType { BUY, SELL }

export(Texture) var character
export(ShopType) var shop_type = ShopType.BUY
export(String) var item_group = "item"

func _ready():
	$Sprite.texture = self.character

func interact(body):
	var main = global.find_entity("main")
	if self.shop_type == ShopType.BUY:
		main.show_popup(ShopPopup, self.item_group)
