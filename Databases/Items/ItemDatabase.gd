extends "res://Databases/DatabaseBase.gd"

func get_item(item_name):
	return self.get_entry(item_name)

func get_items_in_group(group_name):
	return self.get_entries_in_group(group_name)

func get_equip(equip_name):
	self._get_from_tab("Equips", equip_name)

func get_consumable(consumable_name):
	self._get_from_tab("Consumables", consumable_name)

func get_key(key_name):
	self._get_from_tab("Keys", key_name)

func get_drop(drop_name):
	self._get_from_tab("Drops", drop_name)
