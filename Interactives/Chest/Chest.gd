extends "res://Interactives/InteractiveBase.gd"

const MONEY_INTERVAL = 100

var amount = 0

func _ready():
	self.init(300)

func init(money_amount):
	self.amount = money_amount
	var frame = int(ceil(money_amount / MONEY_INTERVAL))
	$Sprite.frame = frame if frame <= 3 else 3

func interact(body):
	if amount != 0:
		body.collect_money(self.amount)
		self.amount = 0
		$Sprite.frame = 0
