from environment import WargameEnv
import pyglet
from pyglet import shapes

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
            radius = 20,
            color = (0, 0, 255),
            batch=self.batch
        )
        self.red = pyglet.shapes.Circle(
            x = env.red_pos[0],
            y = env.red_pos[1],
            radius = 20,
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

        @self.window.event
        def on_draw():
            self.window.clear()
            self.batch.draw()

    def render(self):
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

        # main rendering
        pyglet.clock.tick()
        self.window.switch_to()
        self.window.dispatch_events()  # process pending events (keyboard, mouse, etc.)
        self.window.clear()  # clear the window
        self.batch.draw()  # draw all shapes in the batch
        self.window.flip()  # swap buffers to display the frame

