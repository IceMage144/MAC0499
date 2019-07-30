extends "res://Characters/Human/Human.gd"

const SELL_MULTIPLIER = 0.7
const BAG_SIZE = 36
const QUICK_SIZE = 4
const EQUIP_SLOTS = ["sword"]

var bag = []
var equipments = {}
var quick_items = []

func _ready():
	for item_name in $Model.get_data($Model.BAG):
		var item_node = ItemDB.get_item(item_name)
		self.bag.append(item_node)
	for item_name in $Model.get_data($Model.QUICK):
		var item_node = ItemDB.get_item(item_name)
		self.quick_items.append(item_node)
	for slot in EQUIP_SLOTS:
		if $Model.get_data($Model[slot.to_upper()]) != "":
			self.equipments[slot] = ItemDB.get_item($Model.get_data($Model[slot.to_upper()]))
		else:
			self.equipments[slot] = null

func _save_bag():
	var item_names = []
	for item in self.bag:
		item_names.append(item.name)
	$Model.set_data($Model.BAG, item_names)

func _save_equip(slot):
	if self.equipments[slot] == null:
		$Model.set_data($Model[slot.to_upper()], "")
	else:
		$Model.set_data($Model[slot.to_upper()], self.equipments[slot].name)

func _save_quick_items():
	var item_names = []
	for item in self.quick_items:
		item_names.append(item.name)
	$Model.set_data($Model.QUICK, item_names)

func get_bag():
	return self.bag

func get_equips():
	return self.equipments

func get_money():
	return $Model.get_data($Model.MONEY)

func get_quick_items():
	return self.quick_items

func bag_is_full():
	return len(self.bag) >= BAG_SIZE

func equip_item(item):
	print("Equiped: " + item.name)
	var item_index = self.bag.find(item)
	# Assert that the item is in the bag
	assert(item_index != -1)
	# Assert that the item is an equipment
	assert(item.is_in_group("equip"))

	self.bag.erase(item)
	if self.equipments[item.type] != null:
		self.bag.append(self.equipments[item.type])
	self.equipments[item.type] = item

	self._save_bag()
	self._save_equip(item.type)

func unequip_item(item):
	if self.bag_is_full():
		print("Bag is full")
		return false
	
	print("Unequiped: " + item.name)
	
	self.bag.append(item)
	self.equipments[item.type] = null

	self._save_bag()
	self._save_equip(item.type)

	return true

func remove_quick_item(item):
	self.quick_items.erase(item)
	self._save_quick_items()

func add_quick_item(item):
	if not (item in self.quick_items) and len(self.quick_items) < QUICK_SIZE:
		self.quick_items.append(item)
		self._save_quick_items()

func buy_item(item):
	var money = self.get_money()
	if money < item.price or self.bag_is_full():
		print("You don't have enough money to buy this")
		return false
	
	$Model.set_data($Model.MONEY, int(money - item.price))
	self.bag.append(item)

	self._save_bag()
	print("Purchased: " + item.name)
	
	return true

func sell_item(item):
	$Model.set_data($Model.MONEY, self.get_money() + ceil(SELL_MULTIPLIER * item.price))
	self.bag.erase(item)

	self._save_bag()
	print("Sold: " + item.name)

func use_item(item):
	print("Used: " + item.name)
	self.bag.erase(item)
	self.add_life(item.life_heal)
	self._save_bag()