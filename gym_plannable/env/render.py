import gym
from IPython.display import display, HTML
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import time

class MplFigEnv(gym.Env):
    def __init__(self, figsize=None, delay=10, dpi=None,
                 notebook=True, display_inline=True, **kwargs):
        super().__init__(**kwargs)
        self.display_handle = None
        self.delay = delay
        self.display_inline = display_inline

        if notebook:
            self.display_handle = display(HTML(""), display_id=True)
        
        if self.display_inline and self.display_handle is None:
            self.render_fig = plt.figure(figsize=figsize, dpi=dpi)
        else:
            self.render_fig = Figure(figsize=figsize, dpi=dpi)

    @staticmethod
    def _draw_pause(fig, delay):
        manager = fig.canvas.manager
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        canvas.start_event_loop(delay)

    def update_display(self, delay=None):
        delay = delay or self.delay

        if self.display_handle is None:
            plt.show(block=False)
            self._draw_pause(self.render_fig, delay / 1000)
        else:
            self.display_handle.update(self.render_fig)
            time.sleep(delay / 1000)

    def render(self, mode='human', close=False,
               fig=None, ax=None, update_display=True,
               **kwargs):
        if fig is None:
            self._render(self.render_fig, mode='human', close=False,
                         ax=ax, **kwargs)

            if self.display_inline and update_display:
                self.update_display()
        else:
            self._render(fig, mode='human', close=False,
                         ax=ax, **kwargs)
        
        if close: self.close()

    def close(self):
        if self.display_inline:
            if self.display_handle is None:
                plt.close(self.render_fig)
            else:
                self.display_handle.update(HTML(""))
