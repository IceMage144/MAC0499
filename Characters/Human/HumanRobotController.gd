extends "res://Bases/Controller/BaseRobotController.gd"

const ATTACK_RANGE = 35

func get_legal_actions(state):
	var legal_actions = [Action.IDLE, Action.ATTACK]
	# TODO: Support for "walk_up_right", "walk_up_left", "walk_down_left", "walk_down_right"
	for dir in Action.directions(true):
		var cell = self.tm.get_cellv(self.tm.world_to_map(state["self_pos"]) + Action.to_vec(dir))
		if cell != 1:
			legal_actions.append(Action.compose(Action.WALK, dir))

	return legal_actions

func get_reward(last_state, new_state, timeout):
	if Action.get_movement(last_state["self_act"]) == Action.DEATH or \
	   Action.get_movement(last_state["enemy_act"]) == Action.DEATH:
		return 0.0

	if new_state["enemy_life"] == 0:
		return min(0.5 / (1.0 * self.parent.min_exploration_rate), 5.0)

	if new_state["self_life"] == 0 or timeout:
		return - min(0.5 / (1.0 * self.parent.min_exploration_rate), 5.0)

	# var walked_vec = new_state["self_pos"] - last_state["self_pos"]
	# var enemy_dist = last_state["enemy_pos"] - new_state["self_pos"]
	# var dot_dist = 1.0 if walked_vec.dot(enemy_dist) > 0.0 else -1.0

	# CAUTION: Needs normalization if damage per think is too high
	var self_life_dif = last_state["self_life"] - new_state["self_life"]
	var enemy_life_dif = last_state["enemy_life"] - new_state["enemy_life"]

	# Range: [-0.75, 0.25]
	return 0.5 * (enemy_life_dif - self_life_dif) - 0.25

func get_features_after_action(state, action):
	var self_mov = Action.get_movement(action)
	var enemy_mov = Action.get_movement(state["enemy_act"])
	var enemy_dir_vec = Action.to_vec(state["enemy_act"])
	var out = []
	for i in range(FEATURES_SIZE):
		out.append(0.0)

	# TODO: Use A* instead of euclidian distance
	out[ENEMY_DIST] = state["self_pos"].distance_to(state["enemy_pos"])
	out[SELF_LIFE] = 0.0
	out[ENEMY_LIFE] = 0.0
	out[ENEMY_ATTACKING] = 2.0 * float(enemy_mov == Action.ATTACK) - 1.0
	out[ENEMY_DIR_X] = enemy_dir_vec.x
	out[ENEMY_DIR_Y] = enemy_dir_vec.y
	out[BIAS] = 1.0

	if self_mov == Action.WALK:
		var dir_vec = Action.to_vec(action)
		var transform = Transform2D(0.0, state["self_pos"])
		# COMMENT: Make this test at get_legal_actions?
		if not self.parent.test_move(transform, dir_vec):
			out[ENEMY_DIST] = state["enemy_pos"].distance_to(state["self_pos"] + dir_vec)
	elif self_mov == Action.ATTACK:
		if self._is_aligned(state["self_act"], state["enemy_pos"] - state["self_pos"]) \
		   and out[ENEMY_DIST] < ATTACK_RANGE:
			out[ENEMY_LIFE] -= (ATTACK_RANGE - out[ENEMY_DIST]) / ATTACK_RANGE
			# out[ENEMY_LIFE] = -1.0
	
	# CAUTION: DO NOT REMOVE THIS ATTACK_RANGE VERIFICATION, OTHERWISE IT WILL ALWAYS PREFFER
	# ATTACK THAN OTHER ACTIONS
	if enemy_mov == Action.ATTACK \
	   and self._is_aligned(state["enemy_act"], state["self_pos"] - state["enemy_pos"]) \
	   and out[ENEMY_DIST] < ATTACK_RANGE:
		out[SELF_LIFE] -= (ATTACK_RANGE - out[ENEMY_DIST]) / ATTACK_RANGE
		# out[SELF_LIFE] = -1.0

	out[ENEMY_DIST] /= self.get_viewport().size.length()

	return out

func _on_ThinkTimer_timeout():
	._on_ThinkTimer_timeout()

	if self.can_think():
		var ai_action = self.ai.get_action()
		match Action.get_movement(ai_action):
			Action.ATTACK:
				# COMMENT: Is it good to set parent's action?
				self.parent.attack()
				self.velocity = Vector2() # Leave this here for consistency
			Action.WALK:
				self.velocity = Action.to_vec(ai_action)
			Action.IDLE:
				self.velocity = Vector2()