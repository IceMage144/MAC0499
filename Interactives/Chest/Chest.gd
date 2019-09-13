extends "res://Interactives/InteractiveBase.gd"

const MONEY_INTERVAL = 10

var amount = 0

func _ready():
	self.init({"amount": 30})

func init(params):
	self.amount = params.amount
	var frame = int(ceil(params.amount / MONEY_INTERVAL))
	$Sprite.frame = frame if frame <= 3 else 3

func interact(body):
	if self.amount != 0:
		body.collect_money(self.amount)
		self.amount = 0
		$Sprite.frame = 0
