extends Area2D

enum Type { KEY, AREA }

export(Type) var type = Type.KEY
export(String) var influence_group = "player"

func _process(delta):
	if self.type == Type.KEY and Input.is_action_just_pressed("interact"):
		for body in self.get_overlapping_bodies():
			if body.is_in_group(self.influence_group):
				self.interact(body)

func interact(body):
	pass

func _on_body_entered(body):
	if self.type == Type.AREA and body.is_in_group(self.influence_group):
		self.interact(body)
