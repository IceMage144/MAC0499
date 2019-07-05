from godot import exposed, export
from godot.bindings import *
from godot.globals import *

class Movement:
    IDLE = "idle"
    WALK = "walk"
    ATTACK = "attack"
    DEATH = "death"

class Direction:
    RIGHT = "right"
    UP = "up"
    LEFT = "left"
    DOWN = "down"

    _dir_to_vec = {
        UP: Vector2(0, -1),
        RIGHT: Vector2(1, 0),
        DOWN: Vector2(0, 1),
        LEFT: Vector2(-1, 0)
    }

    def _to_vec(d):
        return Direction._dir_to_vec[d]
    
    def _get_size():
        return len([k for k in Direction.__dict__.keys() if k[0] != "_"])
    
    def _dir_vec_pairs():
        return Direction._dir_to_vec.items()

    def _items():
        return [Direction.__dict__.get(k) for k in Direction.__dict__.keys() if k[0] != "_"]

class Action:
    IDLE = (Movement.IDLE, None)
    ATTACK = (Movement.ATTACK, None)
    WALK_LEFT = (Movement.WALK, Direction.LEFT)
    WALK_RIGHT = (Movement.WALK, Direction.RIGHT)
    WALK_UP = (Movement.WALK, Direction.UP)
    WALK_DOWN = (Movement.WALK, Direction.DOWN)

    _pair_to_act = {
        Movement.IDLE: IDLE,
        Movement.ATTACK: ATTACK,
        Movement.WALK: {
            Direction.LEFT: WALK_LEFT,
            Direction.RIGHT: WALK_RIGHT,
            Direction.UP: WALK_UP,
            Direction.DOWN: WALK_DOWN
        }
    }
    _id_to_act = {
        0: IDLE,
        1: ATTACK,
        2: WALK_LEFT,
        3: WALK_RIGHT,
        4: WALK_UP,
        5: WALK_DOWN
    }
    _pair_to_id = {
        Movement.IDLE: 0,
        Movement.ATTACK: 1,
        Movement.WALK: {
            Direction.LEFT: 2,
            Direction.RIGHT: 3,
            Direction.UP: 4,
            Direction.DOWN: 5
        }
    }
    _act_to_id = {v: k for k, v in _id_to_act.items()}

    def _get_size():
        return len([k for k in Action.__dict__.keys() if k[0] != "_"])

    def _items():
        return [Action.__dict__.get(k) for k in Action.__dict__.keys() if k[0] != "_"]
    
    def _get_action(m, d=None):
        if type(Action._pair_to_act[m]) == tuple:
            return Action._pair_to_act[m]
        return Action._pair_to_act[m][d]
    
    def _get_action_id(act):
        dir_dict = Action._pair_to_id[act[0]]
        if type(dir_dict) == dict:
            return dir_dict[act[1]]
        return dir_dict
    
    def _get_action_from_id(id):
        return Action._id_to_act[id]


class Feature:
    ENEMY_DIST = 0
    SELF_LIFE = 1
    ENEMY_LIFE = 2
    ENEMY_ATTACKING = 3
    ENEMY_DIR_X = 4
    ENEMY_DIR_Y = 5
    BIAS = 6
    
    def _get_size():
        return len([k for k in Feature.__dict__.keys() if k[0] != "_"])