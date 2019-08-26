extends "res://Maps/RoomExits/ExitBase.gd"

const neighbors = [Vector2(0, 1), Vector2(1, 0),
				   Vector2(0, -1), Vector2(-1, 0)]

export(int) var fill_tile_id = 0

var cells = []

func _ready():
	self._find_cells()
	var marker_map = global.find_entity("marker")
	var hf_cell_size = marker_map.cell_size / 2
	var top = -INF
	var bottom = INF
	var right = -INF
	var left = INF
	for cell in self.cells:
		var cell_pos = marker_map.map_to_world(cell)
		left = min(left, cell_pos.x + hf_cell_size.x)
		right = max(right, cell_pos.x + hf_cell_size.x)
		bottom = min(bottom, cell_pos.y + hf_cell_size.y)
		top = max(top, cell_pos.y + hf_cell_size.y)
	var pos = self.position
	var polygon = PoolVector2Array([
		Vector2(left, top) - pos, Vector2(right, top) - pos,
		Vector2(right, bottom) - pos, Vector2(left, bottom) - pos
	])
	$Area2D/CollisionPolygon2D.polygon = polygon

func _find_cells():
	var wall = global.find_entity("wall")
	var marker_map = global.find_entity("marker")
	var initial_cell = marker_map.world_to_map(self.position)
	# Assert that this entrance has some associated marked cells
	assert(marker_map.get_cellv(initial_cell) != TileMap.INVALID_CELL)
	var stack = [initial_cell]
	var marked = {}
	while len(stack) != 0:
		var current_cell = stack.pop_back()
		marked[current_cell] = true
		if marker_map.get_cellv(current_cell) == 1:
			self.cells.append(current_cell)
		for vec in neighbors:
			var child = current_cell + vec
			if marked.has(child):
				continue
			var cell = marker_map.get_cellv(child)
			if cell == 2 or cell == 1:
				stack.append(child)

func remove():
	var wall = global.find_entity("wall")
	for cell in self.cells:
		wall.set_cellv(cell, self.fill_tile_id)
		wall.update_bitmask_area(cell)

func _on_Area2D_body_entered(body):
	if body.is_in_group("player"):
		self.emit_signal("player_collided")
