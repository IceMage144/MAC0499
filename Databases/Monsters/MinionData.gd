extends "res://Databases/Monsters/MonsterDataBase.gd"

enum AIType {BERKELEY, TORCH, MEMO, CLASSIFIER}

export(AIType) var ai_type = AIType.BERKELEY
export(float, 1.0, 0.0, 0.001) var learning_rate = 0.0
export(float, 0.0, 1.0, 0.001) var discount = 0.0
export(float, 0.0, 1.0, 0.001) var max_exploration_rate = 1.0
export(float, 0.0, 1.0, 0.001) var min_exploration_rate = 0.0
export(float) var exploration_rate_decay_time = 0.0
export(bool) var experience_replay = false
export(int) var experience_pool_size = 40
export(float) var think_time = 0.1