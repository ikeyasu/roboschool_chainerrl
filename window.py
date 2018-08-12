import pyglet, pyglet.window as pw, pyglet.window.key as pwk
from pyglet import gl


# noinspection PyAbstractClass
class PygletInteractiveWindow(pw.Window):
    def __init__(self, _env, window_w=600, window_h=400):
        super().__init__()
        pw.Window.__init__(self, width=window_w, height=window_h, vsync=False, resizable=True)

        self.win_h = 0

        @self.event
        def on_close():
            pass

        @self.event
        def on_resize(width, height):
            self.win_w = width
            self.win_h = height

        self.keys = {}
        self.left_pressed = False
        self.right_pressed = False

    def imshow(self, arr):
        h, w, c = arr.shape
        assert c == 3
        image = pyglet.image.ImageData(w, h, 'RGB', arr.tobytes(), pitch=w * -3)
        self.clear()
        self.switch_to()
        self.dispatch_events()
        texture = image.get_texture()
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture.width = w
        texture.height = h
        texture.blit(0, 0, width=self.win_w, height=self.win_h)
        self.flip()

    def on_key_press(self, key, modifiers):
        self.keys[key] = +1
        if key==pwk.LEFT: self.left_pressed = True
        if key==pwk.RIGHT: self.right_pressed = True

    def on_key_release(self, key, modifiers):
        self.keys[key] = 0
        self.right_pressed = self.left_pressed = False
