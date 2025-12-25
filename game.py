from environment import WargameEnv
import pyglet
from pyglet import shapes, text

class Game:
    _window = None # set global variable for window

    def __init__(self, env: WargameEnv):
        """
        initialize state of the world once for the entire game
        """
        self.env = env
        if Game._window is None:
            Game._window = pyglet.window.Window()
        self.window = Game._window
        self.batch = pyglet.graphics.Batch()

        # initialize blue and red forces
        self.blue = pyglet.shapes.Circle(
            x = env.blue_pos[0],
            y = env.blue_pos[1],
            radius = 50,
            color = (0, 0, 255),
            batch=self.batch
        )
        self.red = pyglet.shapes.Circle(
            x = env.red_pos[0],
            y = env.red_pos[1],
            radius = 50,
            color = (255, 0, 0),
            batch=self.batch
        )
        # blue_objs = self.env.blue_pos.values()
        # self.blue_circles = {}

        # for blue_obj in blue_objs:
        #     position = self.env.blue_pos[blue_obj]
        #     x, y = position

        #     circle = pyglet.shapes.Circle(
        #         x = x,
        #         y = y,
        #         radius = 20, 
        #         color=(0, 0, 255),
        #         batch=self.batch
        #     )

            # self.blue_circles[blue_obj] = circle

        # text labels for unit counts
        self.blue_label = text.Label(
            f'Blue: {env.blue_units}',
            font_name='Courier New',
            font_size=24,
            x=env.blue_pos[0],
            y=env.blue_pos[1] + 30,
            anchor_x='center',
            anchor_y='center',
            color=(0, 0, 255, 255),
            batch=self.batch
        )
        self.red_label = text.Label(
            f'Red: {env.red_units}',
            font_name='Courier New',
            font_size=24,
            x=env.red_pos[0],
            y=env.red_pos[1] + 30,
            anchor_x='center',
            anchor_y='center',
            color=(255, 0, 0, 255),
            batch=self.batch
        )

        # tracking label
        self.generation = text.Label(
            font_name='Courier New',
            font_size=28,
            x=10,
            y=380,
            color=(255, 255, 255, 255),
            batch=self.batch
        )

        @self.window.event
        def on_draw():
            self.window.clear()
            self.batch.draw()

    def render(self, current_gen, fitness_score):
        # initialize all object positions in display
        # for uid, pos in self.env.blue_pos.items():
        #     circle = self.blue_circles[uid]
        #     x, y = pos
        #     circle.x = x
        #     circle.y = y
        self.blue.x = self.env.blue_pos[0]
        self.blue.y = self.env.blue_pos[1]

        self.red.x = self.env.red_pos[0]
        self.red.y = self.env.red_pos[1]

        # update circle sizes based on the unit count
        # min size: 5, max size: 40
        self.blue.radius = max(5, min(40, 10 + self.env.blue_units * 0.8))
        self.red.radius = max(5, min(40, 10 + self.env.red_units * 0.8))

        # update labels
        self.blue_label.text = f'Blue: {self.env.blue_units}'
        self.blue_label.x = self.env.blue_pos[0]
        self.blue_label.y = self.env.blue_pos[1] + self.blue.radius + 15
        
        self.red_label.text = f'Red: {self.env.red_units}'
        self.red_label.x = self.env.red_pos[0]
        self.red_label.y = self.env.red_pos[1] + self.red.radius + 15
        
        # Update time
        self.generation.text = f'Gen: {current_gen} | Time: {self.env.time}/{self.env.max_time} | Fitness: {fitness_score}'

        # main rendering
        pyglet.clock.tick()
        self.window.switch_to()
        self.window.dispatch_events()  # process pending events (keyboard, mouse, etc.)
        self.window.clear()  # clear the window
        self.batch.draw()  # draw all shapes in the batch
        self.window.flip()  # swap buffers to display the frame

