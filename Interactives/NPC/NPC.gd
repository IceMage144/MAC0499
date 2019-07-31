extends "res://Interactives/InteractiveBase.gd"

const ShopPopup = preload("res://UI/Popups/ShopPopup.tscn")

enum ShopType { BUY, SELL }
enum Charater { DUMMY, MERLARA, LIA, SATOSHI, SUZERIAN, BALDRIC }

export(Charater) var character_name = Charater.DUMMY
export(ShopType) var shop_type = ShopType.BUY
export(String) var item_group = "item"

func _ready():
	$Sprite.frame = self.character_name

func interact(body):
	var main = global.find_entity("main")
	var popup_params = {
		"shop_type": self.shop_type,
		"item_group": self.item_group
	}
	main.show_popup(ShopPopup, popup_params)
