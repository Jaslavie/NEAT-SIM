import random 

class WargameEnv:
    def __init__(self):
        # unit states
        self.blue_units=27
        self.blue_pos = [200, 200]
        self.blue_hp = 1.0
        self.blue_firepower = 0.8

        self.red_units=33
        self.red_pos  = [400, 200]
        self.red_hp  = 1.0
        self.red_firepower = 0.5

        # global states
        self.time = 0
        self.max_time = 100

        # env states
        self.wind_speed=0.3

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
        # only pick action with highest activation
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

        # counter attacks from red (only if red has units and blue has units)
        if self.red_units > 0 and self.blue_units > 0:
            red_damage = self.calculate_damage(self.red_units, self.red_firepower)
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

        # A wins if:\(\alpha A_{0}>\beta B_{0}\)
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
    def move_forward(self, blue_pos):
        """
        advance
        move right on the x axis
        """
        blue_pos[0] += 10
        return blue_pos

    def move_backward(self, blue_pos):
        """
        retreat
        move left on the x axis
        """
        blue_pos[0] -= 10
        return blue_pos
    
    def calculate_damage(self, units, firepower, hit_probability=0.3):
        """
        damage computed from lanchester equation
        \(dB/dt=-aA\)
        """
        damage_rate = units * firepower

        # determine if units are hit at all, then calculate damage
        if random.random() < hit_probability:
            actual_damage = damage_rate * random.uniform(0, 1.0)  # number of units destroyed
        else:
            actual_damage = 0
        
        return actual_damage
    
    # def interpret_actions(action):
