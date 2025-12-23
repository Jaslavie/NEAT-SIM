class WargameEnv:
    def __init__(self):
        # unit states
        self.blue_units=27
        self.blue_pos=[100,100]
        self.blue_health=1.0
        self.blue_firepower = 0.8

        self.red_units=33
        self.red_pos = [0,0]
        self.red_firepower = 0.5
        self.red_health = 1.0

        # global states
        self.time = 0
        self.max_time = 100

        # env states
        self.wind_speed=0.3

    def reset(self):
        """
        reset the simulation after each run
        """
    
    def step(self, action):
        """
        execute on a series of actions
        """
    def get_observation(self):
        """
        get the current state of env as input in the NN
        """
        # return nodes for neural net
        return {
            1: self.blue_pos[0], # blue x position
            2: self.blue_pos[1], # blue y position
            3: self.red_pos[0],
            4: self.red_pos[1],

            5: self.wind_speed,

            6: self.blue_health,
            7: self.red_health,
            8: self.blue_firepower,
            9: self.red_firepower,
            10: self.blue_units,
            11: self.red_units,
        }