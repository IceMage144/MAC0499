extends Node

const SELL_MULTIPLIER = 0.7

export(Texture) var icon
export(String, MULTILINE) var description = ""
export(int) var price = 1

onready var sell_price = int(ceil(SELL_MULTIPLIER * self.price))

func _ready():
	self.add_to_group("item")