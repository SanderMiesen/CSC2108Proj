"""
Docstring for env_src.getout.getout.goal_conduciveness

Implementation of Goal Conduciveness metric 
    - generation of subgoal set 
    - current state of subgoal completion 
    - computation of total GC score 
    - gamma value 
    - generate reward term (used elsewhere?)
    
# NOTE: the output value has to at least be stored elsewhere to compute potential difference GC(t+1) - GC(t)
    
NOTE: 
unclear whether should implement within env classes:
env_src/getout/getout/getout.py
in/envs/getout/env.py

...or agent classes: 
logic_agent etc. 
"""


class SubGoal():
    # # # subgoal has to be associated with OBJECT (type) # # # 
    def __init__(self, obj_type, init_dist = 1.0, active = False):
        self.goal_obj = obj_type
        # self.init_dist = init_dist
        self.active = active
        self.progress = 0.0
    
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
        
    def init_subgoal(self, init_dist): 
        self.init_dist = init_dist
        self.progress = 0.0
        self.active = True
    
    def reset_subgoal(self): 
        self.init_dist = 1.0
        self.progress = 0.0
        self.active = False
        

# can GoalConduciveness be instantiated in train.py or ... env  and then added to params of agent 
class GoalConduciveness(): 
    # - generation of subgoal set 
    # - current state of subgoal completion 
    # - computation of total GC score 
    # - gamma value 

    def __init__(self, gamma = 1.0, normalize = True, update='with_agent'):
        self.gamma = gamma
        self.subgoals = {} # using dict to preserve both order and uniqueness
        self.subgoal_queue = {} # subgoals added but not available for use until next agent update
        self.active_subgoal = None 
        self.num_goals = 1
        self.GC_score = 0.0
        self.normalize = normalize
        
        if update not in {'with_agent', 'episodic'}:
            raise ValueError("update parameter must be 'with_agent' or 'episodic'")
        self.update = update
        
    # load GC metric from previously trained agent 
    def load_GC(self, gc_info): 
        pass
    
    def reset_GC_progress(self, state_dict): 
        for subgoal in self.subgoals.values():
            subgoal.reset_subgoal()
        self.GC_score = 0.0
        
        # set first subgoal to 'active' and compute initial distance to subgoal object
        # MAYBE this whole thing is just its own function called activate_subgoal
        first_goal = self.subgoals[1]
        if first_goal and first_goal.goal_obj in state_dict:
            init_dist = dist(state_dict['player'][0], state_dict[first_goal][0]) 
            first_goal.init_subgoal(init_dist)
        
    
    def add_subgoal_to_queue(self, obj_type): 
        if any(subgoal.goal_obj == obj_type for subgoal in list(self.subgoals.values()) + list(self.subgoal_queue.values())):
            return "subgoal already exists"

        new_subgoal = SubGoal(obj_type=obj_type, active=False)
        self.subgoal_queue[self.num_goals] = new_subgoal
        self.num_goals += 1
    
    def append_queue(self): 
        self.subgoals = self.subgoals | self.subgoal_queue # add new subgoals from queue
        self.subgoal_queue = {} # reset queue
        
        
    def compute_GC_score(self): 
        # summation 
        self.GC_score = 0.0
        for subgoal in self.subgoals: 
            self.GC_score += subgoal.progress
            
        # normalization
        if self.normalize: 
            self.GC_score = self.GC_score / len(self.subgoals)
            
        # apply gamma (can do outside of here!)
        # self.GC_score_gamma = self.gamma * self.GC_score
        
        return self.GC_score
    
    
    def return_active_subgoal(self): 
        active_subgoals = [(num, subgoal) for num, subgoal in self.subgoals.items() if subgoal.active]
        if len(active_subgoals) > 1:
            raise ValueError("Multiple active subgoals found")
        elif len(active_subgoals) == 0:
            print("Warning: No active subgoals")
            return None
        else:
            return active_subgoals[0]
    
    def complete_current_subgoal(self): 
        # find current active subgoal --> TODO confirm it is same as that passed in?...
        current_subgoal = self.return_active_subgoal()
        if current_subgoal:
            goal_num, current_subgoal = current_subgoal
            current_subgoal.complete_subgoal()
            if goal_num < len(self.subgoals): 
                self.subgoals[goal_num+1].init_subgoal()
                # TODO problem here is that we're not getting external value for distance of the object! 
                # maybe we just set it to ACTIVE with init 
                # then from above we call get ACTIVE goal and check obj distance and use that to set its progress 

        
    def display_GC(self): 
        # display all subgoal progress (** nothing is recomputed here **)
        output = ""
        for goal_num, subgoal in self.subgoals.items(): 
            output += f"goal {goal_num}: {subgoal.goal_obj} -- progress {subgoal.progress}\n"
        output += f"Total Goal Progress:  {self.GC_score}"
        return output
        
        
            
    def compute_potential_difference(self): 
        pass
        # NOTE: hence gc_score should be kept in buffer..
        
        # this has to be computed with access to prev GC scores...
        # so better done in higher level? 
        # 
        # GC_pd = GC_score_t+1 - GC_score_t
        
        # each subgoal has a current status wrt agent - complete / incomplete 
        # GC_subgoal <- dist(agent, obj)
        # reward term is based on current subgoal 
        # each subgoal has value which must be reset at start of episode 
        # perhaps these both can be reduced to value
        
        
        
        # WE could also pass the whole state representation INTO GoalConduciveness and then we can just USE it 
        # for any update necessary
        

# helper function to calculate horizontal distance between objects
def dist(obj1_x, obj2_x):
    return abs(obj1_x - obj2_x)
            
"""
# decoding the converted_state_representation 

# kind of annoying it would be nice to just work with the object types directly 

# bypass by getting env to pass out unconverted state up to higher level 


            if key == 'player':
                logic_state[0][0] = 1
                logic_state[0][-2:] = value
            elif key == 'key':
                logic_state[1][1] = 1
                logic_state[1][-2:] = value
            elif key == 'door':
                logic_state[2][2] = 1
                logic_state[2][-2:] = value
            elif key == 'enemy':
                logic_state[3][3] = 1
                logic_state[3][-2:] = value
            elif key == 'enemy2':
                logic_state[4][3] = 1
                logic_state[4][-2:] = value
            elif key == 'enemy3':
                logic_state[5][3] = 1
                logic_state[5][-2:] = value
"""