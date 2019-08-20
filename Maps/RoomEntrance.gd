extends Position2D

const neighbors = [Vector2(0, 1), Vector2(1, 0),
				   Vector2(0, -1), Vector2(-1, 0)]

export(int) var fill_tile_id = 0

func remove(pos):
	var wall = global.find_entity("wall")
	var marker_map = global.find_entity("marker")
	var initial_cell = marker_map.world_to_map(pos)
	# Assert that this entrance has some associated marked cells
	assert(marker_map.get_cellv(initial_cell) != TileMap.INVALID_CELL)
	var stack = [initial_cell]
	var marked = {}
	while len(stack) != 0:
		var current_cell = stack.pop_back()
		marked[current_cell] = true
		if marker_map.get_cellv(current_cell) == 1:
			wall.set_cellv(current_cell, self.fill_tile_id)
			wall.update_bitmask_area(current_cell)
		for vec in neighbors:
			var child = current_cell + vec
			if marked.has(child):
				continue
			var cell = marker_map.get_cellv(child)
			if cell == 2 or cell == 1:
				stack.append(child)
