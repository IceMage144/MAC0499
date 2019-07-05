import time
import math

from godot.bindings import *
from godot.globals import *

class Logger:
    """
    Class for storing and logging statistics
    """
    def __init__(self):
        self.table = {}
    
    def push(self, name, val):
        if not self.table.get(name):
            self.table[name] = []
        self.table[name].append(val)
    
    def avg(self, name):
        if not self.table.get(name):
            return 0.0
        return sum(self.table[name])/len(self.table[name])
    
    def max(self, name):
        if not self.table.get(name):
            return 0.0
        return max(self.table[name])

    def min(self, name):
        if not self.table.get(name):
            return 0.0
        return min(self.table[name])
    
    def sum(self, name):
        if not self.table.get(name):
            return 0.0
        return sum(self.table[name])

    def flush(self, name):
        self.table[name] = []

    def size(self, name):
        return len(self.table[name])
    
    def print_stats(self, name, stats_list):
        if self.table.get(name) is None:
            return
        print(f"{name} ({self.size(name)} values):")
        for stat in stats_list:
            attr = getattr(self, stat, None)
            if not (attr is None) :
                print(f"\t{stat}: {attr(name)}")
    
    def get_stored(self, name):
        if self.table.get(name) is None:
            return []
        return self.table[name]

# Decorator that prints the execution time every execution
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print(f"{method.__name__}: {1000 * (te - ts)}ms")
        return result

    return timed

# Sign of a number
def sign(num):
    return 1.0 if num > 0.0 else (0.0 if num == 0.0 else -1.0)

# Translates python array to gd array
def py2gdArray(array):
	if not hasattr(array, '__iter__'):
		return array
	ret = Array()
	for item in array:
		ret.append(py2gdArray(item))
	return ret

# Applies a function recusivelly at a multidimentional array
def apply_list_func(array, func):
	if not hasattr(array, '__iter__'):
		return array
	return func([apply_list_func(el, func) for el in array])

def argmax(table):
    mx_val = float("-inf")
    mx_key = ""
    for key, val in table.items():
        if val > mx_val:
            mx_key = key
            mx_val = val
    return mx_key
