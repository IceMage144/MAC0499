extends "res://Interactives/InteractiveBase.gd"

var item

func init(item):
	self.item = item
	$Sprite.texture = item.icon

func interact(body):
	if body.collect(self.item):
		self.queue_free()