extends "res://Maps/RoomBase.gd"

func init(params):
	.init(params)
	if not params.has("available_entrances"):
		params["available_entrances"] = []
	for spawner in $PlayerSpawners.get_children():
		if spawner.name in params.available_entrances:
			spawner.connect("player_collided", self.get_parent(), "change_room", [spawner.name])
		else:
			if spawner.has_method("remove"):
				spawner.remove()
			spawner.queue_free()
