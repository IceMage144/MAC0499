extends "res://Databases/Monsters/MonsterDataBase.gd"

enum AIType {BERKELEY, TORCH, MEMO, CLASSIFIER}

export(AIType) var ai_type = AIType.BERKELEY