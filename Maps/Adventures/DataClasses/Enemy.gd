extends Object

signal enemy_death()

var name
var life
var entry
var instance

func _init(name):
	self.name = name
	self.entry = MonsterDB.get_entry(name)
	self.life = self.entry.max_life
	self.instance = null

func get_instance():
	if self.instance:
		return self.instance
	if self.is_dead():
		return null
	self.instance = self.entry.type.instance()
	self.instance.init(self.entry)
	self.instance.connect("character_death", self, "_on_enemy_death")
	self.instance.add_to_group("team2")
	return self.instance

func kill_instance():
	if self.instance:
		self._save_attributes()
		self.instance.queue_free()
		self.instance = null

func is_dead():
	return self.life == 0

func _save_attributes():
	self.life = self.instance.life

func _on_enemy_death():
	self.kill_instance()
	self.emit_signal("enemy_death")

