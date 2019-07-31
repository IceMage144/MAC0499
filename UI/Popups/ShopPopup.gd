extends "res://UI/Popups/PopupBase.gd"

enum ShopType { BUY, SELL }

export(NodePath) var ShelvesPath
export(NodePath) var ItemInfoPath
export(NodePath) var MoneyDisplayPath

var shop_type

onready var Shelves = self.get_node(ShelvesPath)
onready var ItemInfo = self.get_node(ItemInfoPath)
onready var MoneyDisplay = self.get_node(MoneyDisplayPath)
onready var Player = global.find_entity("player")

func init(params):
	self.shop_type = params.shop_type
	var item_list
	if params.shop_type == ShopType.BUY:
		item_list = ItemDB.get_items_in_group(params.item_group)
	elif params.shop_type == ShopType.SELL:
		item_list = Player.get_bag()
	Shelves.display_items(item_list)
	MoneyDisplay.display_money(Player.get_money())

func _on_nothing_selected():
	ItemInfo.remove_item()

func _on_item_selected(item):
	if self.shop_type == ShopType.BUY:
		ItemInfo.display_item(item, ItemInfo.BUY)
	elif self.shop_type == ShopType.SELL:
		ItemInfo.display_item(item, ItemInfo.SELL)

func _on_item_activated(item):
	if self.shop_type == ShopType.BUY:
		self._on_item_bought(item)
	elif self.shop_type == ShopType.SELL:
		self._on_item_sold(item)

func _on_item_bought(item):
	Player.buy_item(item)
	MoneyDisplay.display_money(Player.get_money())

func _on_item_sold(item):
	Player.sell_item(item)
	MoneyDisplay.display_money(Player.get_money())
	var bag = Player.get_bag()
	Shelves.display_items(bag)
	if not (item in bag):
		ItemInfo.remove_item()
