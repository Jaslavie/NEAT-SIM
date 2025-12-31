import random 
from abc import ABC, abstractmethod

class WargameEnv:
    def __init__(self):
        # unit counts
        self.blue_units=27
        self.blue_pos = [500, 500]
        self.blue_hp = 1.0
        self.blue_firepower = 0.8

        self.red_units=33
        self.red_pos  = [800, 500]
        self.red_hp  = 1.0
        self.red_firepower = 0.5

        # global states
        self.time = 0
        self.max_time = 100

        # env states
        self.wind_speed=0.3

        self.red_fsm = RedStateMachine()

    def reset(self):
        """
        reset the simulation after each run
        """
        self.__init__()
    
    def step(self, action):
        """
        execute on an action at each timestamp
        intensity of action is determined separately if activated
        - action: resulting actions of forward pass (array of activation floats of each output node)
        """
        self.get_blue_action(action)

        # counter attacks from red (only if red has units and blue has units)
        if self.red_units > 0 and self.blue_units > 0:
            red_damage = self.get_red_actions(self.blue_pos, self.red_pos)
            self.blue_units -= red_damage
            self.blue_units = max(0, int(round(self.blue_units)))  # ensure units remain integer
        
        # increment time
        self.time += 1

        # terminate condition
        done = self.is_done()
        # reward = self.calculate_reward(done) # calc the reward at the end state
        return self.get_observation(), done

    def get_observation(self):
        """
        get the current state of env as input in the NN
        """
        # return nodes for neural net
        return {
            1: self.blue_units,
            2: self.red_units,

            3: self.wind_speed,

            # 4: self.blue_hp,
            # 5: self.red_hp,
            4: self.blue_firepower,
            5: self.red_firepower,
        }
    
    def check_victory(self):
        """
        victory condition is based on the lanchester function
        """
        blue_combat_power = self.blue_firepower * (self.blue_units ** 2)
        red_combat_power = self.red_firepower * (self.red_units ** 2)

        # A wins if:\(\alpha A_{0}>\beta B_{0}\)\
        if blue_combat_power > red_combat_power:
            return "blue_advantage"
        elif red_combat_power > blue_combat_power:
            return "red_advantage"
        return "even"
    
    def is_done(self):
        """
        check if simulation should terminate early
        """
        return (self.blue_units <= 0 or self.red_units <= 0 or self.time >= self.max_time)

    # ======== ACTIONS ========== 
    def calculate_damage(self, units, firepower, hit_probability=0.6):
        """
        damage computed from lanchester equation
        \\(dB/dt=-aA\\)
        """
        damage_rate = units * firepower

        # determine if units are hit at all, then calculate damage
        if random.random() < hit_probability:
            actual_damage = damage_rate * random.uniform(0, 1.0)  # number of units destroyed
        else:
            actual_damage = 0
        
        return actual_damage
    
    def get_blue_action(self, action):
        """
        potential actions selected by NEAT algorithm
        """
        # compute highest probability action
        max_indx = action.index(max(action))

        # engage
        if max_indx == 0:
            engagement_intensity = (action[0] - 0.5) * 2  
            move_distance = 10 * engagement_intensity  # scale movement by intensity
            self.blue_pos[0] += move_distance
        
        # retreat
        if max_indx == 1:
            retreat_intensity = (action[1] - 0.5) * 2  
            move_distance = 10 * retreat_intensity  # scale movement by intensity
            self.blue_pos[0] -= move_distance
        
        # attack
        if max_indx == 2:
            if self.red_units > 0:
                # map intensity to [0.0, 1.0] range
                # turns the output state (damaged units) into a continuous function
                attack_intensity = (action[2] - 0.5) * 2
                damage = self.calculate_damage(self.blue_units, self.blue_firepower) * attack_intensity
                self.red_units -= damage
                self.red_units = max(0, int(round(self.red_units)))  # prevent units from going negative and ensure integer
        
        # hold/defend
        if max_indx == 3:
            pass

    def get_red_actions(self):
        """
        selected from red state machine
        """
        pass
        
    
    # def interpret_actions(action):

class StateMachine(ABC):
    def __init__(self, name: str):
        self.name = name
    # ==== lifecycle hooks to define current state ====
    def on_enter(self, obs):
        print(f"entering {self.name}")
        pass
    def on_update(self, obs):
        pass
    def on_exit(self, obs):
        print(f"exiting {self.name}")
        pass

    # ==== hfsm management ====
    def load_substate(self):
        """
        substates are strategic positions: defend, attack, etc
        """
        pass
    def add_transition(self):
        pass

    def enter_state_machine():
        pass

    def update_state_machine():
        pass

    def exit_state_machine():
        pass

    # implemented by children
    @abstractmethod
    def select_action(self, obs) -> dict:
        pass


class RedStateMachine(StateMachine):
    def __init__(self):
        super().__init__(name="RED")
        # init states
        self.states = {
            'LINE': LineFormState,
            'DEFENSE': DefenseState,
            'RETREAT': RetreatState
        }

        # begin in line formation
        self.current_state = self.states['LINE']
        self.current_state.enter_state_machine({})

        # collect history of actions
        self.action_history = []

    def update(self, obs):
        """
        update red's actions for each step
        """
        # check if we need to transition to a new state
        new_state = self.check_transition(obs)
        if new_state and new_state != self.current_state:
            print(f"transitioned to {new_state}")
            self.transition_to(new_state)

        # execute action based on current state
        action = self.current_state.select_action(obs)

        return action
    
    def check_transition(self, obs):
        """
        check if observations trigger a strategy state transition
        """
        pass

    def transition_to(self, new_state: str):
        """
        execute transitions to diff states
        """
        print(f"red fleet: {self.current_state} transitioned to {new_state}")

        # switch to new state
        self.current_state = self.states[new_state]

    
class LineFormState:
    """
    maintain position with controlled fire
    """

    def select_action(self, obs):
        """
        distance based actions
        - long range = minimal fire
        - med range = controlled fire
        - close range: transition to different state
        """
        # compute on x axis
        distance = abs(obs['blue_pos'][0] - obs['red_pos'][0])

        red_strength_ratio = obs['red_units'] / 33.0
        
        # red ships will only attack upon close range
        if distance > 100:
            type = 'LONG_RANGE_FIRE'
            red_effective_units = obs['red_units'] * 0.3
        elif distance < 50:
            type = 'CLOSE_RANGE_FIRE'
            red_effective_units = obs['red_units'] * 0.8
        else:
            type = 'CONTROLLED_FIRE'
            red_effective_units = obs['red_units'] * 0.5 
        
        return {
            'type': type,
            'effective_units': red_effective_units
        }

class DefenseState:
    """
    franco-spanish ships fragment after the british breaks the line
    """
    def select_action(self, obs):
        pass

class RetreatState:
    """
    full retreat to cadiz port
    """
    def select_action(self, obs):
        pass

test_obs = {
    'blue_pos': [890, 500],
    'red_pos': [800, 500],
    'red_units': 33,
    'red_firepower': 0.5,
    'blue_units': 27,
    'blue_firepower': 0.8
}
state = LineFormState()
action = state.select_action(test_obs)
print(action)