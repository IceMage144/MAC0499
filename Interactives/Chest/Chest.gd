extends "res://Interactives/InteractiveBase.gd"

const MONEY_INTERVAL = 100

var ammount = 0

func _ready():
	self.init(300)

func init(money_ammount):
	self.ammount = money_ammount
	var frame = int(ceil(money_ammount / MONEY_INTERVAL))
	$Sprite.frame = frame if frame <= 3 else 3

func interact(body):
	if ammount != 0:
		body.collect_money(self.ammount)
		self.ammount = 0
		$Sprite.frame = 0
