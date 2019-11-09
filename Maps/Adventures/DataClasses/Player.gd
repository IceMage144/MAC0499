extends Object

signal player_death()

var life
var instance

func _init():
	self.life = null

func get_instance():
	self.instance = global.find_entity("player")
	if self.life == null:
		self.life = self.instance.get_max_life()
	self.instance.init({
		"life": self.life
	})
	return self.instance

func kill_instance():
	self.life = self.instance.life
	self.instence.queue_free()
	self.instance = null