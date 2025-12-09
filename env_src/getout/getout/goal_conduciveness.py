"""
Docstring for env_src.getout.getout.goal_conduciveness


Implementation of Goal Conduciveness metric 
    - generation of subgoal set 
    - current state of subgoal completion 
    - computation of total GC score 
    - gamma value 
    - 
    
# NOTE: the output value has to at least be stored elsewhere to compute potential difference GC(t+1) - GC(t)
    
NOTE: 
unclear whether should implement within env classes:
env_src/getout/getout/getout.py
in/envs/getout/env.py

or agent classes: 
... logic_agent etc. 
"""

class SubGoal():
    def __init__(self, init_dist = 1.0, active = True):
        self.active = active
        self.progress = 0.0
        self.init_dist = init_dist
    
    # progress can be measured wrt (with respect to) agent's initial (in that episode) distance to subgoal
    def compute_progress(self, curr_dist): 
        if not self.active: 
            raise ValueError("Cannot compute progress for inactive subgoal")
        self.progress = (self.init_dist - curr_dist) / (self.init_dist + 1e-7)
        # do we also need to constrain it to non-negative
        # self.progress = max(0, self.progress)
        return self.progress

    def complete_subgoal(self):  # if goal is completed (decided at env/agent level)
        self.progress = 1.0
        self.active = False
    
    def reset_subgoal(self, init_dist): 
        self.init_dist = init_dist
        self.progress = 0.0
        self.active = True
        

# can GoalConduciveness be instantiated in train.py or ... env  and then added to params of agent 
class GoalConduciveness(): 
    # - generation of subgoal set 
    # - current state of subgoal completion 
    # - computation of total GC score 
    # - gamma value 

    def __init__(self, gamma = 0.01, normalize = False):
        self.gamma = gamma
        self.subgoals = {} # using dict to preserve both order and uniqueness
        self.active_subgoal = None 
        self.num_goals = 0
        self.GC_score = 0.0
        self.normalize = normalize
    
    def add_subgoal(self, init_dist, active): 
        new_subgoal = SubGoal(init_dist=init_dist, active=active)
        self.subgoals[self.num_goals] = new_subgoal
        self.num_goals += 1
        
    def compute_GC_score(self): 
        # summation 
        self.GC_score = 0.0
        for subgoal in self.subgoals: 
            self.GC_score += subgoal.progress
            
        # normalization
        if self.normalize: 
            self.GC_score = self.GC_score / len(self.subgoals)
            
        # apply gamma  (maybe shouldn't do here)
        self.GC_score = self.gamma * self.GC_score
        
        return self.GC_score
        
            
    def compute_potential_difference(self): 
        # this has to be computed with access to prev GC scores... 
        # so better done in higher level? 
        # 
        
        # each subgoal has a current status wrt agent - complete / incomplete 
        # GC_subgoal <- dist(agent, obj)
        # reward term is based on current subgoal 
        # each subgoal has value which must be reset at start of episode 
        # perhaps these both can be reduced to value