extends Node

func _ready():
	for child in self.get_children():
		var group_name = child.name.to_lower().substr(0, child.name.length() - 1)
		for grand_child in child.get_children():
			grand_child.add_to_group(group_name)

func _get_from_tab(tab, item_name):
	return self.get_node(tab).get_child(item_name)

func get_item(item_name):
	for child in self.get_children():
		if child.has_node(item_name):
			return child.get_node(item_name)
	assert(false) # Item does not exist
	return null

func get_items_in_group(item_group):
	return get_tree().get_nodes_in_group(item_group)

func get_equip(equip_name):
	self._get_from_tab("Equips")

func get_consumable(consumable_name):
	self._get_from_tab("Consumables")

func get_key(key_name):
	self._get_from_tab("Keys")

func get_drop(drop_name):
	self._get_from_tab("Drops")
