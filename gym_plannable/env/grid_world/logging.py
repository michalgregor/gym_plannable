from .grid_world import TransitionObject
import matplotlib.pyplot as plt
import numpy as np

class PathLogger(TransitionObject):
    def __init__(self, name, actor_name, tile_size=1,
                 show_arrows=True, arrow_color='black', arrow_alpha=1.0,
                 show_visited=False, visited_color='blue', visited_alpha=0.1,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.done = False
        self.actor_name = actor_name
        self.tile_size = tile_size
        
        self.show_arrows = show_arrows
        self.arrow_color = arrow_color
        self.arrow_alpha = arrow_alpha
        
        self.show_visited = show_visited
        self.visited_color = visited_color
        self.visited_alpha = visited_alpha

    def init(self, **params):
        p = getattr(self.world, self.actor_name)
        self.path = [p.position]
        self.done = p.done

    def all_init(self):
        return (({}, 1.0) for _ in range(1))

    def next(self, **params):
        p = getattr(self.world, self.actor_name)
        self.path.append(p.position)
        self.done = p.done

    def all_next(self):
        return (({}, 1.0) for _ in range(1))

    def render(self, ax):
        if self.show_arrows:
            path = np.asarray(self.path)

            X = path[:-1, 1]
            Y = path[:-1, 0]
            U = path[1:, 1] - path[:-1, 1]
            V = path[:-1, 0] - path[1:, 0]

            ax.quiver(
                X, Y, U, V,
                scale_units='xy',
                scale=self.tile_size,
                color=self.arrow_color,
                alpha=self.arrow_alpha,
                zorder=5
            )

        if self.show_visited:
            for r, c in self.path:
                patch = plt.Rectangle(
                    [c - self.tile_size / 2,
                    r - self.tile_size / 2],
                    self.tile_size, self.tile_size,
                    facecolor=self.visited_color,
                    alpha=self.visited_alpha
                )
                ax.add_patch(patch)
