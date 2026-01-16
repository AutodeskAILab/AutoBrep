"""
A simple class to visualize point grids using matplotlib.

We could try to avoid explicit dependency on
pytorch and Open Cascade in here
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from autobrep.utils import compute_bbox_center_and_size, plot_3d_bbox


class PointGridVisualizer:
    def __init__(self, max_surfaces, num_constraint_surfaces=0):
        # Set up the colors, so these will be consistent
        # over frames
        self.max_surfaces = max_surfaces

        # Save the red surfaces for the constraints
        self.colors = cm.rainbow(np.linspace(0.3, 1, max_surfaces))
        np.random.shuffle(self.colors)

        # Fix the constraint surfaces to be purple
        for i in range(num_constraint_surfaces):
            self.colors[i] = cm.rainbow(0)

        self.fig = None
        self.ax = None

    def new_figure(self, max_box_coord=1.1):
        """
        Start plotting a new figure
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim([-max_box_coord, max_box_coord])
        self.ax.set_ylim([-max_box_coord, max_box_coord])
        self.ax.set_zlim([-max_box_coord, max_box_coord])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

    def plot_bounding_boxes(self, face_pos):
        """
        Plot the surface bounding boxes.
        This should be called with bounding boxes
        for a single solid.
        """
        assert self.fig is not None, (
            "Image already rendered.   Should call new_figure()"
        )
        assert face_pos.shape[0] <= self.max_surfaces, (
            "Should have less than max surfaces"
        )
        assert face_pos.shape[1] == 6, "Should have 6 values  minpt, max_pt"
        for bbox, color in zip(face_pos, self.colors):
            plot_3d_bbox(self.ax, bbox[0:3], bbox[3:6], color)

    def plot_vertex_point(self, vertex_pos, edge_mask):
        """
        Plot the vertex point.
        """
        for pos, mask in zip(vertex_pos, edge_mask):
            valid_pos = pos[~mask]
            self.ax.scatter(
                valid_pos[:, 0], valid_pos[:, 1], valid_pos[:, 2], c="red", s=5
            )
            self.ax.scatter(
                valid_pos[:, 3], valid_pos[:, 4], valid_pos[:, 5], c="red", s=5
            )

    def plot_surface_point_grids(self, face_pos, face_ncs):
        """
        Plot the surface point grids.
        This should be called with bounding boxes
        and point grids for a single solid.
        """

        for bbox, ncs, color in zip(face_pos, face_ncs, self.colors):
            center, size = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
            ncs = ncs.reshape(-1, 3)
            wcs = ncs * (size / 2) + center
            self.ax.scatter(wcs[:, 0], wcs[:, 1], wcs[:, 2], color=color, s=0.1)

    def plot_edge_point_grids(self, edge_pos, edge_ncs, edge_mask):
        """
        Plot the edge point grids.
        This should be called with bounding boxes
        and point grids for a single solid.
        """
        for bbox, ncs, mask in zip(edge_pos, edge_ncs, edge_mask):
            valid_bbox = bbox[~mask]
            valid_ncs = ncs[~mask]
            for bb, ee in zip(valid_bbox, valid_ncs):
                center, size = compute_bbox_center_and_size(bb[0:3], bb[3:])
                wcs = ee * (size / 2) + center
                self.ax.scatter(wcs[:, 0], wcs[:, 1], wcs[:, 2], c="black", s=5)

    def save_image(self, pathname):
        """
        Render the image to a specified pathname
        """
        self._before_render()
        plt.savefig(pathname, format="png", dpi=300)
        plt.close(self.fig)
        self.fig = None
        self.ax = None

    def _before_render(self):
        """
        Do things we need to do before rendering the image
        """
        plt.tight_layout()
