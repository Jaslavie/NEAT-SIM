from environment import WargameEnv
import pyglet
from pyglet import shapes

class Game:
    def __init__(self, env: WargameEnv):
        """
        initialize state of the world once for the entire game
        """
        self.env = env
        self.window = pyglet.window.Window()
        self.batch = pyglet.graphics.Batch()

        self.blue = pyglet.shapes.Circle(
            x = self.env.blue_pos[0],
            y = self.env.blue_pos[1],
            radius = 20, 
            color=(0, 0, 255),
            batch=self.batch
        )

        @self.window.event
        def on_draw():
            self.window.clear()
            self.batch.draw()

    def render(self):
        # initialize object positions in display
        self.blue.x = self.env.blue_pos[0]
        self.blue.y = self.env.blue_pos[1]

        # main rendering
        pyglet.clock.tick()
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.dispatch_event('on_draw')
        self.window.flip()

