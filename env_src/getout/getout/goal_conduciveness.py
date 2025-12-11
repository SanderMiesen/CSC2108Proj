"""
Docstring for env_src.getout.getout.goal_conduciveness

Implementation of Goal Conduciveness metric:
    - generation of subgoal set:
        - adding *new* subgoals first to queue, then adding queue to subgoal set when triggered
        - (see 'with_agent', 'episodic')
    - current state of subgoal completion 
    - computation of total GC score
    - gamma value
    - reward term is computed outside of class
"""


class SubGoal():
    def __init__(self, obj_type, active = False):
        self.goal_obj = obj_type
        self.init_dist = -999.0
        self.active = active
        self.progress = 0.0
    
    # progress can be measured wrt (with respect to) agent's initial (in that episode) distance to subgoal
    def compute_progress(self, curr_dist): 
        if not self.active: 
            raise ValueError("Cannot compute progress for inactive subgoal")
        self.progress = (self.init_dist - curr_dist) / (self.init_dist + 1e-7)
        
        # do we also need to constrain it to non-negative?
        # self.progress = max(0, self.progress)
        
        return self.progress

    def complete_subgoal(self):
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
    
    
    def get_active_subgoal(self): 
        # returns subgoal and its 'index' 
        active_subgoals = [(num, subgoal) for num, subgoal in self.subgoals.items() if subgoal.active]
        if len(active_subgoals) > 1:
            raise ValueError("Multiple active subgoals found")
        elif len(active_subgoals) == 0:
            # print("No active subgoals")
            return None, None
        else:
            return active_subgoals[0] # should only be one active subgoal
        
        
    def compute_active_progress(self, state_dict): 
        goal_num, current_goal = self.get_active_subgoal()
        if current_goal:
            assert current_goal.goal_obj in state_dict
            curr_dist = dist(state_dict['player'][0], state_dict[current_goal.goal_obj][0])
            current_goal.compute_progress(curr_dist)
        
    
    def complete_current_subgoal(self, obj_type, state_dict): 
        goal_num, current_subgoal = self.get_active_subgoal()
        if current_subgoal:
            if current_subgoal.goal_obj == obj_type and goal_num != None:
            
                current_subgoal.complete_subgoal()
                
                if goal_num < len(self.subgoals): 
                    self.init_next_subgoal(goal_num+1, state_dict)
                else: 
                    return "no further subgoals"
            else: 
                return "reward not associated with active goal"
        else: 
            return "no active goals"
            
        
    def init_next_subgoal(self, goal_num, state_dict): 
        if goal_num in self.subgoals:
            next_goal = self.subgoals[goal_num]   
            if next_goal.goal_obj in state_dict:
                init_dist = dist(state_dict['player'][0], state_dict[next_goal.goal_obj][0]) 
                next_goal.init_subgoal(init_dist)
            else: 
                return "no object match found"
        else: 
            return "no subgoals"
        
            
    def reset_GC_progress(self, state_dict): 
        for subgoal in self.subgoals.values():
            subgoal.reset_subgoal()
        self.GC_score = 0.0
        return self.init_next_subgoal(1, state_dict) # set first subgoal to 'active' and compute initial distance to subgoal object

    
    def add_subgoal_to_queue(self, obj_type): 
        # check if subgoal already exists
        if any(subgoal.goal_obj == obj_type for subgoal in list(self.subgoals.values()) + list(self.subgoal_queue.values())):
            return False
        new_subgoal = SubGoal(obj_type=obj_type, active=False)
        self.subgoal_queue[self.num_goals] = new_subgoal
        self.num_goals += 1
        return True
    
    def append_queue(self): 
        self.subgoals = self.subgoals | self.subgoal_queue # add new subgoals from queue
        self.subgoal_queue = {} # reset queue
        
    
    def compute_GC_score(self): 
        # summation 
        self.GC_score = 0.0 
        ### TODO actually compute progress of active goal here using state_dict...
        for subgoal in self.subgoals.values(): 
            self.GC_score += subgoal.progress
            
        # normalization
        if self.normalize and len(self.subgoals) > 0: 
            self.GC_score = self.GC_score / len(self.subgoals)
            
        # apply gamma for reward term (done outside of here!)
        # self.GC_score_gamma = self.gamma * self.GC_score
        
        return self.GC_score
    
        
    def display_GC(self): 
        # display all subgoal progress (** nothing is recomputed here **)
        output = ""
        for goal_num, subgoal in self.subgoals.items(): 
            output += f"goal {goal_num}: {subgoal.goal_obj} -- progress {subgoal.progress}\n"
        output += f"Total Goal Progress:  {self.GC_score}"
        return output
        
    # def compute_potential_difference(self): 
    #     pass

# helper function to calculate horizontal distance between objects
def dist(obj1_x, obj2_x):
    return abs(obj1_x - obj2_x)