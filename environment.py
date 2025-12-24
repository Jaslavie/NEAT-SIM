
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
        execute on an action given an activation number
        - action: resulting actions of forward pass (array of activation floats of each output node)
        """
        if action[0] > 0.5:
            self.blue_pos = self.move_forward(self.blue_pos)
        if action[1] > 0.5:
            self.blue_pos = self.move_backward(self.blue_pos)
        if action[2] > 0.5:
            if self.red_units > 0:
                damage = self.calculate_damage(self.blue_units, self.blue_firepower)
                self.red_units -= damage
                self.red_units = max(0, self.red_units) # prevent units from reaching negative
        if action[3] > 0.5:
            pass # hold/defend
        
    
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
    
    def calculate_damage(self, blue_units, blue_firepower):
        """
        damage computed from lanchester equation
        \(dB/dt=-aA\)
        """
        return blue_units * blue_firepower
    
    # def interpret_actions(action):
