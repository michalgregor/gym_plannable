from .grid_world import TransitionObject
import numpy as np

class PathLogger(TransitionObject):
    def __init__(self, name, actor_name, tile_size=1,
                 color='black', **kwargs):
        super().__init__(name=name, **kwargs)
        self.done = False
        self.actor_name = actor_name
        self.tile_size = tile_size
        self.color = color

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
        path = np.asarray(self.path)
        X = path[:-1, 1]
        Y = path[:-1, 0]
        U = path[1:, 1] - path[:-1, 1]
        V = path[:-1, 0] - path[1:, 0]

        ax.quiver(
            X, Y, U, V,
            scale_units='xy',
            scale=self.tile_size,
            color=self.color,
            zorder=5
        )
