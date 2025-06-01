import numpy as np
from shapely.geometry import Point

class Obstacle:
    def __init__(self, center, radius):
        self.center = center  # a tuple (x, y)
        self.radius = radius
        self.shape = Point(center).buffer(radius)  # Shapely Polygon (circle)

    def contains(self, point):
        """Check if a point (x, y) is inside the obstacle."""
        return self.shape.contains(Point(point))
