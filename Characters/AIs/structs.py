from random import sample

class Experience:
    def __init__(self, features, reward, next_state, action=None):
        self.features = features
        self.reward = reward
        self.next_state = next_state
        self.action = action
    
    def append(self, other):
        self.features.append(other[0])
        self.reward.append(other[1])
        self.next_state.append(other[2])
        if len(other) > 3:
            assert not (self.action is None), "Trying to append action when action is None."
            self.action.append(other[3])

class ExperiencePool:
    def __init__(self, sample_size):
        self.sample_size = sample_size
        self.experiences = []

    def sample(self):
        if len(self.experiences) < 2 * self.sample_size:
            return None
        return sample(self.experiences, self.sample_size)
    
    def simple_sample(self):
        return sample(self.experiences, 1)
    
    def add(self, exp):
        self.experiences.append(exp)