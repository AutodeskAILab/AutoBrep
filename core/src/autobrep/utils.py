import math
import random
import string
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import smart_open
import torch
import torch.nn as nn
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@contextmanager
def timer(message, stream=print):
    start_time = time.time()
    yield
    end_time = time.time()
    stream(message % (end_time - start_time))


class DotDict(dict):
    """
    Dictionary subclass that supports access via
    dot notation and handles nested dictionaries.
    """

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, attr, value):
        if isinstance(value, dict):
            value = DotDict(value)
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]


def normalize_pc(pc):
    min_vals = np.min(pc.reshape(-1, 3), axis=0)
    max_vals = np.max(pc.reshape(-1, 3), axis=0)
    local_offset = min_vals + (max_vals - min_vals) / 2
    local_scale = max(max_vals - min_vals)
    pnt_ncs = (pc - local_offset[np.newaxis, np.newaxis, :]) / (local_scale * 0.5)
    return pnt_ncs


def compute_bbox_center_and_size(min_corner, max_corner):
    # Calculate the center
    center_x = (min_corner[0] + max_corner[0]) / 2
    center_y = (min_corner[1] + max_corner[1]) / 2
    center_z = (min_corner[2] + max_corner[2]) / 2
    center = np.array([center_x, center_y, center_z])
    # Calculate the size
    size_x = max_corner[0] - min_corner[0]
    size_y = max_corner[1] - min_corner[1]
    size_z = max_corner[2] - min_corner[2]
    size = max(size_x, size_y, size_z)
    return center, size


def ncs2wcs(ncs, bbox):
    """
    Convert normalized coordinates to world coordinates
    """
    min_corner, max_corner = bbox[:3], bbox[3:]
    center, size = compute_bbox_center_and_size(min_corner, max_corner)
    wcs = ncs * (size / 2) + center
    return wcs


def batch_ncs2wcs(ncs_batch, bbox_batch):
    """
    Convert normalized coordinates to world coordinates for a batch of data
    """
    wcs_batch = []
    for ncs, bbox in zip(ncs_batch, bbox_batch):
        wcs = ncs2wcs(ncs, bbox)
        wcs_batch.append(wcs)
    return np.array(wcs_batch)


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = (
            generator.device.type
            if not isinstance(generator, list)
            else generator[0].device.type
        )
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(
                f"Cannot generate a {device} tensor from a generator of type {gen_device_type}."
            )

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(
                shape,
                generator=generator[i],
                device=rand_device,
                dtype=dtype,
                layout=layout,
            )
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(
            shape, generator=generator, device=rand_device, dtype=dtype, layout=layout
        ).to(device)

    return latents


def pad_repeat(x, max_len):
    repeat_times = math.floor(max_len / len(x))
    sep = max_len - repeat_times * len(x)
    sep1 = np.repeat(x[:sep], repeat_times + 1, axis=0)
    sep2 = np.repeat(x[sep:], repeat_times, axis=0)
    x_repeat = np.concatenate([sep1, sep2], 0)
    return x_repeat


def pad_zero(x, max_len, return_mask=False):
    if len(x) > max_len:
        raise ValueError(f"len(x)={len(x)} is greater than max_len={max_len}")
    keys = np.ones(len(x))
    padding = np.zeros((max_len - len(x))).astype(int)
    mask = 1 - np.concatenate([keys, padding]) == 1
    padding = np.zeros((max_len - len(x), *x.shape[1:]))
    x_padded = np.concatenate([x, padding], axis=0)
    if return_mask:
        return x_padded, mask
    else:
        return x_padded


def pad_neg(x, max_len, return_mask=False):
    keys = np.ones(len(x))
    padding = np.zeros((max_len - len(x))).astype(int)
    mask = 1 - np.concatenate([keys, padding]) == 1
    padding = -np.ones((max_len - len(x), *x.shape[1:])).astype(int)
    x_padded = np.concatenate([x, padding], axis=0)
    if return_mask:
        return x_padded, mask
    else:
        return x_padded


def plot_3d_bbox(ax, min_corner, max_corner, color="r"):
    """
    Helper function for plotting 3D bounding boxese
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = [
        (min_corner[0], min_corner[1], min_corner[2]),
        (max_corner[0], min_corner[1], min_corner[2]),
        (max_corner[0], max_corner[1], min_corner[2]),
        (min_corner[0], max_corner[1], min_corner[2]),
        (min_corner[0], min_corner[1], max_corner[2]),
        (max_corner[0], min_corner[1], max_corner[2]),
        (max_corner[0], max_corner[1], max_corner[2]),
        (min_corner[0], max_corner[1], max_corner[2]),
    ]
    # Define the 12 triangles composing the box
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]],
    ]
    ax.add_collection3d(
        Poly3DCollection(
            faces, facecolors="blue", linewidths=1, edgecolors=color, alpha=0
        )
    )
    return


def rotate_vectors(vectors: np.ndarray, angle_degrees: float, axis: np.ndarray):
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Compute rotation matrix based on the specified axis
    if axis == "x":
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_radians), -np.sin(angle_radians)],
                [0, np.sin(angle_radians), np.cos(angle_radians)],
            ],
            dtype=vectors.dtype,
        )
    elif axis == "y":
        rotation_matrix = np.array(
            [
                [np.cos(angle_radians), 0, np.sin(angle_radians)],
                [0, 1, 0],
                [-np.sin(angle_radians), 0, np.cos(angle_radians)],
            ],
            dtype=vectors.dtype,
        )
    elif axis == "z":
        rotation_matrix = np.array(
            [
                [np.cos(angle_radians), -np.sin(angle_radians), 0],
                [np.sin(angle_radians), np.cos(angle_radians), 0],
                [0, 0, 1],
            ],
            dtype=vectors.dtype,
        )
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")
    return np.dot(vectors, rotation_matrix.T)


def rotate_point_cloud(point_cloud, angle_degrees, axis):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """
    # Center the point cloud
    center = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - center

    # Apply rotation
    rotated_point_cloud = rotate_vectors(centered_point_cloud, angle_degrees, axis)

    # Translate back to original position
    rotated_point_cloud += center

    # Find the maximum absolute coordinate value
    max_abs_coord = np.max(np.abs(rotated_point_cloud))

    # Scale the point cloud to fit within the -1 to 1 cube
    normalized_point_cloud = rotated_point_cloud / max_abs_coord

    return normalized_point_cloud


def get_bboxes(pnts):
    """
    Get the tighest fitting 3D (axis-aligned) bounding box giving a set of points
    """
    bbox_corners = [get_bbox(point_cloud) for point_cloud in pnts]
    return np.array(bbox_corners)


def get_bbox(point_cloud):
    """
    Get the tightest fitting 3D (axis-aligned) bounding box around a set of points
    """
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return np.array([min_point, max_point])


def bbox_corners(bboxes):
    """
    Given the bottom-left and top-right corners of the bbox
    Return all eight corners
    """
    bboxes_all_corners = []
    for bbox in bboxes:
        bottom_left, top_right = bbox[:3], bbox[3:]
        # Bottom 4 corners
        bottom_front_left = bottom_left
        bottom_front_right = (top_right[0], bottom_left[1], bottom_left[2])
        bottom_back_left = (bottom_left[0], top_right[1], bottom_left[2])
        bottom_back_right = (top_right[0], top_right[1], bottom_left[2])

        # Top 4 corners
        top_front_left = (bottom_left[0], bottom_left[1], top_right[2])
        top_front_right = (top_right[0], bottom_left[1], top_right[2])
        top_back_left = (bottom_left[0], top_right[1], top_right[2])
        top_back_right = top_right

        # Combine all coordinates
        all_corners = [
            bottom_front_left,
            bottom_front_right,
            bottom_back_left,
            bottom_back_right,
            top_front_left,
            top_front_right,
            top_back_left,
            top_back_right,
        ]
        bboxes_all_corners.append(np.vstack(all_corners))
    bboxes_all_corners = np.array(bboxes_all_corners)
    return bboxes_all_corners


def get_bbox_diag_distance(point_cloud):
    min_point, max_point = get_bbox(point_cloud)
    return np.linalg.norm(max_point - min_point)


def rotate_axis(pnts, angle_degrees, axis, normalized=False):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, ..., 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Convert points to homogeneous coordinates
    shape = list(np.shape(pnts))
    shape[-1] = 1
    pnts_homogeneous = np.concatenate((pnts, np.ones(shape, dtype=pnts.dtype)), axis=-1)

    # Compute rotation matrix based on the specified axis
    if axis == "x":
        rotation_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
                [0, np.sin(angle_radians), np.cos(angle_radians), 0],
                [0, 0, 0, 1],
            ],
            dtype=pnts.dtype,
        )
    elif axis == "y":
        rotation_matrix = np.array(
            [
                [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
                [0, 1, 0, 0],
                [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
                [0, 0, 0, 1],
            ],
            dtype=pnts.dtype,
        )
    elif axis == "z":
        rotation_matrix = np.array(
            [
                [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
                [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=pnts.dtype,
        )
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Apply rotation
    rotated_pnts_homogeneous = np.dot(pnts_homogeneous, rotation_matrix.T)
    rotated_pnts = rotated_pnts_homogeneous[..., :3]

    # Scale the point cloud to fit within the -1 to 1 cube
    if normalized:
        max_abs_coord = np.max(np.abs(rotated_pnts))
        rotated_pnts = rotated_pnts / max_abs_coord

    return rotated_pnts


def rescale_bbox(bboxes, scale):
    # Apply scaling factors to bounding boxes
    scaled_bboxes = bboxes * scale
    return scaled_bboxes


def select_random_offset(min_val, max_val, offset_factor):
    # Calculate the range
    limits = np.array(
        [
            min_val + offset_factor,
            min_val - offset_factor,
            max_val + offset_factor,
            max_val - offset_factor,
        ]
    )
    min_thres, max_thres = limits.min(), limits.max()
    add_range = min(max_thres, 1) - max_val
    subtract_range = max(min_thres, -1) - min_val
    offset = np.random.uniform(subtract_range, add_range)
    return offset


def find_random_bbox_translations(bboxes, offset_factor=0.2):
    """
    Find a set of random translations which are suitable for applying to the given array of boxes
    """
    point_cloud = bboxes.reshape(-1, 3)
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])
    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])
    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])
    x_offset = select_random_offset(min_x, max_x, offset_factor)
    y_offset = select_random_offset(min_y, max_y, offset_factor)
    z_offset = select_random_offset(min_z, max_z, offset_factor)
    random_translation = np.array([x_offset, y_offset, z_offset])
    return random_translation


def find_max_scale(min_val, max_val, scale_factor):
    inc_scale = [1 + scale_factor]
    if max_val != 0:
        inc_scale.append(min(max_val * (1 + scale_factor), 1) / max_val)
    if min_val != 0:
        inc_scale.append(
            max(min_val * (1 + scale_factor), -1) / min_val,
        )
    return np.array(inc_scale).min()


def find_random_bbox_scale(bboxes, scale_factor=0.2):
    """
    Find a random scale which are suitable for applying to the given array of boxes
    """
    point_cloud = bboxes.reshape(-1, 3)
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])
    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])
    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])
    x_scale = find_max_scale(min_x, max_x, scale_factor)
    y_scale = find_max_scale(min_y, max_y, scale_factor)
    z_scale = find_max_scale(min_z, max_z, scale_factor)
    dec_scale = 1 - scale_factor  # assumes always safe to shrink solid
    inc_scale = min(x_scale, y_scale, z_scale)  # minimum increase scale
    random_scale = np.random.uniform(dec_scale, inc_scale)
    return random_scale


def translate_bbox(bboxes, translation_vectors):
    """
    Translate these bodes by a set of vectors
    """
    bboxes_translated = bboxes + translation_vectors.astype(bboxes.dtype)
    return bboxes_translated


def sort_uv_grid(point):
    """
    Sort a N x N x 3 uv grid so that
        (1) top left corner is the smallest after sorting all four corners
        (2) top right corner is smaller than bottom left corner
    """
    num_u, num_v = point.shape[0], point.shape[1]
    corner_pnts = np.stack(
        [
            point[0, 0],
            point[0, num_v - 1],
            point[num_u - 1, 0],
            point[num_u - 1, num_v - 1],
        ]
    )

    # Move smallest corner to top left
    pnt_order = np.lexsort((corner_pnts[:, 2], corner_pnts[:, 1], corner_pnts[:, 0]))
    if pnt_order[0] == 1:
        point = np.fliplr(point)
    elif pnt_order[0] == 2:
        point = np.flipud(point)
    elif pnt_order[0] == 3:
        point = np.flipud(np.fliplr(point))
    else:
        pass

    # Ensure top right corner smaller than bottom left
    corner_pnts = np.stack(
        [
            point[0, 0],
            point[0, num_v - 1],
            point[num_u - 1, 0],
            point[num_u - 1, num_v - 1],
        ]
    )
    pnt_order = np.lexsort((corner_pnts[:, 2], corner_pnts[:, 1], corner_pnts[:, 0]))
    if np.where(pnt_order == 1)[0] > np.where(pnt_order == 2)[0]:
        point = np.transpose(point, axes=(1, 0, 2))
    return point


def sort_uv_grids(points):
    return np.stack([sort_uv_grid(uv) for uv in points])


def sort_u_grid(point):
    """
    Sort a N x 3 uv grid so that
        (1) left corner is the smallest after sorting all two endpoints
    """
    corner_pnts = np.stack([point[0], point[-1]])
    # Move smallest corner to top left
    pnt_order = np.lexsort((corner_pnts[:, 2], corner_pnts[:, 1], corner_pnts[:, 0]))
    if pnt_order[0] == 1:
        point = point[::-1]
    else:
        pass
    return point


def sort_u_grids(points):
    return np.stack([sort_u_grid(u) for u in points])


@dataclass
class BoundingBox:
    """
    This class is used to represent a bounding box transformation.
    It contains the parametric representation of the transformation, including
    a scale, rotation and translation; as well as methods to convert these
    representations into matrices for use in `occwl`'s `Shape.transform` class.
    {
        "origin": (float, float, float),
        "scale": (float, float, float),
        "x_axis": (float, float, float),
        "y_axis": (float, float, float),
        "z_axis": (float, float, float),
    }
    """

    scale: Tuple[float, float, float] = (1, 1, 1)
    origin: Tuple[float, float, float] = (0, 0, 0)
    x_axis: Tuple[float, float, float] = (1, 0, 0)
    y_axis: Tuple[float, float, float] = (0, 1, 0)
    z_axis: Tuple[float, float, float] = (0, 0, 1)

    @property
    def scale_factor(self):
        return max(self.scale)

    @property
    def shift(self):
        return np.array(self.origin).reshape(3, 1)

    @property
    def rotation(self):
        axes = np.array([self.x_axis, self.y_axis, self.z_axis])
        rotation = Rotation.align_vectors(np.eye(3), axes)
        assert rotation[1] < 1e-4, (
            "Failed to align vectors, are the input vectors orthogonal?"
        )
        return rotation[0]

    def as_matrix(self):
        rotation = self.rotation.as_matrix()
        scale_rotate = rotation * self.scale_factor
        return np.hstack([scale_rotate, self.shift])

    def as_inverse_matrix(self):
        inv_rotation = self.rotation.inv().as_matrix()
        inv_scale_rotate = inv_rotation / self.scale_factor
        return np.hstack([inv_scale_rotate, -inv_scale_rotate.dot(self.shift)])


class STModel(nn.Module):
    def __init__(self, num_surf):
        super().__init__()
        self.surf_st = nn.Parameter(
            torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).repeat(num_surf, 1)
        )


def joint_optimize(
    face_ncs: np.ndarray,  # NumPy array for face normalized uv
    edge_ncs: np.ndarray,  # NumPy array for edge normalized uv
    face_pos: np.ndarray,  # NumPy array for face global positions
    unique_vertices: np.ndarray,  # NumPy array for unique vertices
    edge_vertex_adj: np.ndarray,  # NumPy array for edge-vertex adjacency
    face_edge_adj: Union[np.ndarray, List[List[int]]],
    # NumPy array or list of List as face-edge adjacency
    num_surf: int,  # Integer for number of surfaces
    device: torch.device,  # PyTorch device object
):
    """
    Jointly optimize the face/edge/vertex based on topology
    """
    model = STModel(num_surf)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-08,
    )

    # Optimize edges (directly compute)
    edge_ncs_se = edge_ncs[:, [0, -1]]
    edge_vertex_se = unique_vertices[edge_vertex_adj]
    assert len(edge_ncs_se) == len(edge_vertex_se), "edges do not match"

    edge_wcs = []
    for wcs, ncs_se, vertex_se in zip(edge_ncs, edge_ncs_se, edge_vertex_se):
        # scale
        scale_target = np.linalg.norm(vertex_se[0] - vertex_se[1])
        scale_ncs = np.linalg.norm(ncs_se[0] - ncs_se[1])
        edge_scale = scale_target / scale_ncs

        edge_updated = wcs * edge_scale
        edge_se = ncs_se * edge_scale

        # offset
        offset = vertex_se - edge_se
        offset_rev = vertex_se - edge_se[::-1]

        # swap start / end if necessary
        offset_error = np.abs(offset[0] - offset[1]).mean()
        offset_rev_error = np.abs(offset_rev[0] - offset_rev[1]).mean()
        if offset_rev_error < offset_error:
            edge_updated = edge_updated[::-1]
            offset = offset_rev

        edge_updated = edge_updated + offset.mean(0)[np.newaxis, np.newaxis, :]
        edge_wcs.append(edge_updated)

    edge_wcs = np.vstack(edge_wcs)

    # Replace start/end points with corner, and backprop change along curve
    for index in range(len(edge_wcs)):
        start_vec = edge_vertex_se[index, 0] - edge_wcs[index, 0]
        end_vec = edge_vertex_se[index, 1] - edge_wcs[index, -1]
        weight = np.tile((np.arange(32) / 31)[:, np.newaxis], (1, 3))
        weighted_vec = (
            np.tile(start_vec[np.newaxis, :], (32, 1)) * (1 - weight)
            + np.tile(end_vec, (32, 1)) * weight
        )
        edge_wcs[index] += weighted_vec

    # Optimize surfaces
    face_edges = []
    if isinstance(face_edge_adj, list):
        for adj in face_edge_adj:
            all_pnts = edge_wcs[adj]
            face_edges.append(torch.FloatTensor(all_pnts).to(device))
    else:
        offsets = np.concatenate((np.array([0]), np.cumsum(face_edge_adj.sum(1))))
        for offset_idx in range(len(offsets) - 1):
            all_pnts = edge_wcs[offsets[offset_idx] : offsets[offset_idx + 1]]
            face_edges.append(torch.FloatTensor(all_pnts).to(device))

    # Initialize surface in wcs based on surface pos
    surf_wcs_init = []
    bbox_threshold_min = []
    bbox_threshold_max = []
    for edges_perface, ncs, bbox in zip(face_edges, face_ncs, face_pos):
        surf_center, surf_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
        edges_perface_flat = edges_perface.reshape(-1, 3).detach().cpu().numpy()
        min_point, max_point = get_bbox(edges_perface_flat)
        edge_center, edge_scale = compute_bbox_center_and_size(min_point, max_point)
        bbox_threshold_min.append(min_point)
        bbox_threshold_max.append(max_point)

        # increase surface size if does not fully cover the wire bbox
        if surf_scale < edge_scale:
            surf_scale = 1.05 * edge_scale

        wcs = ncs * (surf_scale / 2) + surf_center
        surf_wcs_init.append(wcs)

    surf_wcs_init = np.stack(surf_wcs_init)

    # optimize the surface offset
    surf = torch.FloatTensor(surf_wcs_init).to(device)
    for iters in range(200):
        surf_scale = model.surf_st[:, 0].reshape(-1, 1, 1, 1)
        surf_offset = model.surf_st[:, 1:].reshape(-1, 1, 1, 3)
        surf_updated = surf + surf_offset

        surf_loss = 0
        for surf_pnt, edge_pnts in zip(surf_updated, face_edges):
            surf_pnt = surf_pnt.reshape(-1, 3)
            edge_pnts = edge_pnts.reshape(-1, 3).detach()
            surf_loss += chamfer_distance(surf_pnt, edge_pnts)
        surf_loss /= len(surf_updated)

        optimizer.zero_grad()
        (surf_loss).backward()
        optimizer.step()

        # print(f"Iter {iters} surf:{surf_loss:.5f}")
    surf_wcs = surf_updated.detach().cpu().numpy()

    return (surf_wcs, edge_wcs)


def chamfer_distance(pc1, pc2):
    """
    Compute the Chamfer Distance between two point clouds.

    Parameters:
    pc1: torch.Tensor of shape (M, 3) - Point cloud 1 (learnable)
    pc2: torch.Tensor of shape (N, 3) - Point cloud 2 (target)

    Returns:
    torch.Tensor: Chamfer distance between the two point clouds
    """
    # Compute pairwise distances between points in pc1 and pc2
    diff_pc1_pc2 = torch.cdist(pc1, pc2, p=2)  # Euclidean distance

    # For each point in pc1, find the nearest point in pc2
    min_dist_pc1_to_pc2, _ = torch.min(diff_pc1_pc2, dim=1)

    # For each point in pc2, find the nearest point in pc1
    min_dist_pc2_to_pc1, _ = torch.min(diff_pc1_pc2, dim=0)

    # Chamfer distance is the average of these minimum distances
    chamfer_loss = torch.mean(min_dist_pc1_to_pc2) + torch.mean(min_dist_pc2_to_pc1)

    return chamfer_loss


def chamfer_distance_kdtree(pc1, pc2):
    """
    Compute the Chamfer distance between two point clouds (pc1, pc2) using the
    squared L2 norm and KDTree
    """
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)

    dist_pc1, _ = tree2.query(pc1, k=1)
    dist_pc2, _ = tree1.query(pc2, k=1)

    dist_pc1_squared = dist_pc1**2
    dist_pc2_squared = dist_pc2**2

    return np.mean(dist_pc1_squared) + np.mean(dist_pc2_squared)


def hausdorff_distance_kdtree(pc1, pc2):
    """
    Compute the Hausdorff distance between two point clouds (pc1, pc2) using the
    squared L2 norm and KDTree
    """
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)

    dist_to_B, _ = tree2.query(pc1, k=1)
    dist_to_A, _ = tree1.query(pc2, k=1)

    return max(np.max(dist_to_B), np.max(dist_to_A))


def edge2loop(face_edges):
    face_edges_flatten = face_edges.reshape(-1, 3)

    # connect end points by closest distance
    merged_vertex_id = []
    for edge_idx, startend in enumerate(face_edges):
        self_id = [2 * edge_idx, 2 * edge_idx + 1]
        cur_ids = np.unique(np.array(merged_vertex_id), axis=0).flatten()

        # left endpoint
        if (2 * edge_idx) not in cur_ids:
            distance = np.linalg.norm(face_edges_flatten - startend[0], axis=1)
            min_id = list(np.argsort(distance))
            min_id_noself = [
                x for x in min_id if (x not in self_id and x not in cur_ids)
            ]
            assert len(min_id_noself) > 0
            merged_vertex_id.append(sorted([2 * edge_idx, min_id_noself[0]]))

        # right endpoint
        if (2 * edge_idx + 1) not in cur_ids:
            distance = np.linalg.norm(face_edges_flatten - startend[1], axis=1)
            min_id = list(np.argsort(distance))
            min_id_noself = [
                x for x in min_id if (x not in self_id and x not in cur_ids)
            ]
            assert len(min_id_noself) > 0
            merged_vertex_id.append(sorted([2 * edge_idx + 1, min_id_noself[0]]))

    merged_vertex_id = np.unique(np.array(merged_vertex_id), axis=0)
    return merged_vertex_id


def keep_largelist(int_lists):
    # Initialize a list to store the largest integer lists
    largest_int_lists = []

    # Convert each list to a set for efficient comparison
    sets = [set(lst) for lst in int_lists]

    # Iterate through the sets and check if they are subsets of others
    for i, s1 in enumerate(sets):
        is_subset = False
        for j, s2 in enumerate(sets):
            if i != j and s1.issubset(s2) and s1 != s2:
                is_subset = True
                break
        if not is_subset:
            largest_int_lists.append(list(s1))

    # Initialize a set to keep track of seen tuples
    seen_tuples = set()

    # Initialize a list to store unique integer lists
    unique_int_lists = []

    # Iterate through the input list
    for int_list in largest_int_lists:
        # Convert the list to a tuple for hashing
        int_tuple = tuple(sorted(int_list))

        # Check if the tuple is not in the set of seen tuples
        if int_tuple not in seen_tuples:
            # Add the tuple to the set of seen tuples
            seen_tuples.add(int_tuple)

            # Add the original list to the list of unique integer lists
            unique_int_lists.append(int_list)

    return unique_int_lists


def detect_shared_vertex(
    edge_mask_cad: Optional[np.ndarray],
    edge_vertex: np.ndarray,
    dist_threshold: float = 0.05,
    face_edge_adj: Optional[np.ndarray] = None,
):
    """
    Find the shared vertices
    """
    # Detect shared-vertex on seperate face loop
    used_vertex = []
    face_sep_merges = []
    # BrepGen
    if face_edge_adj is None:
        assert edge_mask_cad is not None, "edge_mask must be provided"
        edge_id_offset = (
            2
            * np.concatenate(
                [np.array([0]), np.cumsum(np.logical_not(edge_mask_cad).sum(1))]
            )[:-1]
        )
        valid = True

        for face_idx, (face_vertex, edge_mask) in enumerate(
            zip(edge_vertex, edge_mask_cad)
        ):
            face_vertex = face_vertex[~edge_mask]
            face_vertex = face_vertex.reshape(len(face_vertex), 2, 3)
            face_start_id = edge_id_offset[face_idx]

            # Connect vertices by closest distance (per-face)
            merged_vertex_id = edge2loop(face_vertex)
            if len(merged_vertex_id) == len(face_vertex):
                merged_vertex_id = face_start_id + merged_vertex_id
                face_sep_merges.append(merged_vertex_id)
                used_vertex.append(face_vertex)
                continue

            valid = False
            break

    else:
        edge_id_offset = (
            2 * np.concatenate([np.array([0]), np.cumsum(face_edge_adj.sum(1))])[:-1]
        )
        valid = True

        for face_idx, face_vertex in enumerate(edge_vertex):
            face_vertex = face_vertex[np.abs(face_vertex).sum(1) > 0]
            face_vertex = face_vertex.reshape(len(face_vertex), 2, 3)
            face_start_id = edge_id_offset[face_idx]

            # Connect vertices by closest distance (per-face)
            merged_vertex_id = edge2loop(face_vertex)
            if len(merged_vertex_id) == len(face_vertex):
                merged_vertex_id = face_start_id + merged_vertex_id
                face_sep_merges.append(merged_vertex_id)
                used_vertex.append(face_vertex)
                continue

            valid = False
            break

    # Invalid
    if not valid:
        assert False

    # (1) Detect shared-vertex across faces
    total_pnts = np.vstack(used_vertex)
    total_pnts = total_pnts.reshape(len(total_pnts), 2, 3)
    total_pnts_flatten = total_pnts.reshape(-1, 3)

    total_ids = []
    for face_idx, face_merge in enumerate(face_sep_merges):
        # non-self merge centers
        nonself_face_idx = list(set(np.arange(len(face_sep_merges))) - set([face_idx]))
        nonself_face_merges = [face_sep_merges[x] for x in nonself_face_idx]
        nonself_face_merges = np.vstack(nonself_face_merges)
        nonself_merged_centers = total_pnts_flatten[nonself_face_merges].mean(1)

        # connect end points by closest distance
        across_merge_id = []
        for merge_id in face_merge:
            merged_center = total_pnts_flatten[merge_id].mean(0)
            distance = np.linalg.norm(nonself_merged_centers - merged_center, axis=1)
            nonself_match_id = nonself_face_merges[np.argsort(distance)[0]]
            joint_merge_id = list(nonself_match_id) + list(merge_id)
            across_merge_id.append(joint_merge_id)
        total_ids += across_merge_id

    # (2) Merge T-junctions
    while True:
        no_merge = True
        final_merge_id = []

        # iteratelly merge until no changes happen
        for i in range(len(total_ids)):
            perform_merge = False

            for j in range(i + 1, len(total_ids)):
                # check if vertex can be further merged
                max_num = max(len(total_ids[i]), len(total_ids[j]))
                union = set(total_ids[i]).union(set(total_ids[j]))
                common = set(total_ids[i]).intersection(set(total_ids[j]))
                if len(union) > max_num and len(common) > 0:
                    final_merge_id.append(list(union))
                    perform_merge = True
                    no_merge = False
                    break

            if not perform_merge:
                final_merge_id.append(total_ids[i])  # no-merge

        total_ids = final_merge_id
        if no_merge:
            break

    # remove subsets
    total_ids = keep_largelist(total_ids)

    # (3) merge again base on absolute coordinate value, required for >3 T-junction
    tobe_merged_centers = [total_pnts_flatten[x].mean(0) for x in total_ids]
    tobe_centers = np.array(tobe_merged_centers)
    distances = np.linalg.norm(tobe_centers[:, np.newaxis, :] - tobe_centers, axis=2)
    close_points = distances < dist_threshold
    mask = np.tril(np.ones_like(close_points, dtype=bool), k=-1)
    non_diagonal_indices = np.where(close_points & mask)
    row_indices, column_indices = non_diagonal_indices

    # update the total_ids
    total_ids_updated = []
    for row, col in zip(row_indices, column_indices):
        total_ids_updated.append(total_ids[row] + total_ids[col])
    for index, ids in enumerate(total_ids):
        if index not in list(row_indices) and index not in list(column_indices):
            total_ids_updated.append(ids)
    total_ids = total_ids_updated

    # (4) Post-process one last time, merge based on index
    new_vertex_dict = {}
    for new_id, old_ids in enumerate(total_ids):
        new_vertex_dict[new_id] = old_ids

    total_ids_final = total_ids.copy()
    vertex_dict_final = new_vertex_dict.copy()

    while True:
        # Find unmerged cluster
        new_total_ids = None
        for old_id in np.arange(np.hstack(total_ids).max() + 1):
            new_id = []
            for key, value in vertex_dict_final.items():
                # Check if the desired number is in the associated list
                if old_id in value:
                    new_id.append(key)
            if len(new_id) > 1:
                # Update merged index cluster
                merge_idx = [vertex_dict_final[x] for x in new_id]
                new_total_ids = [
                    item
                    for idx, item in enumerate(total_ids_final)
                    if idx not in new_id
                ]
                merged = []
                [
                    merged.append(val)
                    for val in np.concatenate(merge_idx)
                    if val not in merged
                ]
                new_total_ids.append(merged)
                break

        if new_total_ids is None:
            break
        else:
            total_ids_final = new_total_ids
            # Recompute vertex_dict
            vertex_dict_final = {}
            for new_id, old_ids in enumerate(total_ids_final):
                vertex_dict_final[new_id] = old_ids
            print("Merged...")

    # check if two array are identical
    if not np.all(np.concatenate(total_ids_final) == np.concatenate(total_ids)):
        total_ids = total_ids_final
        new_vertex_dict = vertex_dict_final

    # merged vertices
    unique_vertices = []
    for center_id in total_ids:
        center_pnts = total_pnts_flatten[center_id].mean(0)
        unique_vertices.append(center_pnts)
    unique_vertices = np.vstack(unique_vertices)

    return [unique_vertices, new_vertex_dict]


def detect_shared_edge(
    shared_vertex_dict: Dict[int, List[int]],
    edge_ncs_cad: np.ndarray,
    edge_z_cad: np.ndarray,
    edge_mask_cad: np.ndarray,
    z_threshold: float = 0.2,
):
    """
    Find the shared edges
    """
    edge_se = edge_ncs_cad[~edge_mask_cad]
    init_edges = edge_z_cad[~edge_mask_cad]

    # re-assign edge start/end to unique vertices
    new_ids = []
    for old_id in np.arange(2 * len(init_edges)):
        new_id = []
        for key, value in shared_vertex_dict.items():
            # Check if the desired number is in the associated list
            if old_id in value:
                new_id.append(key)
        assert len(new_id) == 1  # should only return one unique value
        new_ids.append(new_id[0])

    edge_vertex_adj = np.array(new_ids).reshape(-1, 2)

    # find edges assigned to the same start/end
    similar_edges = []
    for i, s1 in enumerate(edge_vertex_adj):
        for j, s2 in enumerate(edge_vertex_adj):
            if i != j and set(s1) == set(s2):  # same start/end
                z1 = init_edges[i]
                z2 = init_edges[j]
                z_diff = np.abs(z1 - z2).mean()
                if z_diff < z_threshold:  # check z difference
                    similar_edges.append(sorted([i, j]))
    similar_edges = np.unique(np.array(similar_edges), axis=0)

    # should reduce total edges by two
    if not 2 * len(similar_edges) == len(edge_vertex_adj):
        assert False, "edge not reduced by 2"

    # unique edges
    unique_edge_id = similar_edges[:, 0]
    edge_vertex_adj = edge_vertex_adj[unique_edge_id]
    unique_edges = edge_se[unique_edge_id]

    # face-edge adjacency matrix
    face_edge_adj = []
    ranges = np.concatenate(
        [np.array([0]), np.cumsum(np.logical_not(edge_mask_cad).sum(1))]
    )
    for index in range(len(ranges) - 1):
        adj_ids = np.arange(ranges[index], ranges[index + 1])
        new_ids = []
        for id in adj_ids:
            new_id = np.where(similar_edges == id)[0]
            assert len(new_id) == 1  # should only return one unique value
            new_ids.append(new_id[0])
        face_edge_adj.append(new_ids)

    return [unique_edges, face_edge_adj, edge_vertex_adj]


def generate_random_string(length):
    characters = (
        string.ascii_letters + string.digits
    )  # You can include other characters if needed
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


def contains_segment(query, segments):
    return any(segment in query for segment in segments)


def load_torch_weights(
    ckpt,
    model,
    freeze_weights=None,
):
    with smart_open.open(ckpt, "rb") as f:
        weights = torch.load(f, map_location="cpu")["state_dict"]

    # Load full / partial weights
    if freeze_weights is None:
        weights = {k.removeprefix("model."): v for k, v in weights.items()}
    else:
        weights = {
            k.removeprefix("model."): v
            for k, v in weights.items()
            if contains_segment(k, freeze_weights)
        }
    model.load_state_dict(weights, strict=False)

    # Freeze the weights
    if freeze_weights is not None:
        for name, param in model.named_parameters():
            if contains_segment(name, freeze_weights):
                param.requires_grad = False

    trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    return model, trainable_parameters


def convert_solid_to_mesh(
    solid,
    triangle_tol=0.4,
    angle_tol_radians=0.2,
):
    """
    Convert an OCCWL Solid into a mesh via triangulation
    """
    solid.triangulate_all_faces(
        triangle_face_tol=triangle_tol,
        tol_relative_to_face=True,
        angle_tol_rads=angle_tol_radians,
    )
    verts, faces = solid.get_triangles()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh


def normalize_mesh(mesh):
    """
    Normalize a mesh so its bounding box lies within [-0.5,+0.5]^3.
    """
    bounds_min, bounds_max = mesh.bounds
    bbox_size = bounds_max - bounds_min
    max_dim = bbox_size.max()

    # Avoid division by zero if the mesh is degenerate in some dimension
    if max_dim < 1e-6:
        raise ValueError("Mesh bounding box is too small or degenerate to normalize.")

    scale_factor = 1.0 / max_dim
    mesh.apply_scale(scale_factor)

    centroid = (mesh.bounds[0] + mesh.bounds[1]) / 2
    mesh.apply_translation(-centroid)

    return mesh


def sample_surface_pointcloud(
    mesh,
    n_points=10000,
    normalize=True,
):
    """
    Sample a pointcloud from a mesh using surface sampling.
    """

    if normalize:
        normalize_mesh(mesh)

    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(points)


def sample_voxelized_pointcloud(
    mesh,
    voxel_size=0.02,
    max_points=10000,
    normalize=True,
):
    """
    Sample a point cloud from a mesh using voxelization.
    """

    if normalize:
        normalize_mesh(mesh)

    voxels = mesh.voxelized(pitch=voxel_size, method="subdivide")
    points = voxels.points

    if max_points is not None and len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    return np.asarray(points)


def quantize(
    data: np.ndarray,
    n_bits: int = 8,
    min_range: int = -1,
    max_range: int = 1,
    apply_round: bool = False,
):
    """
    Convert vertices in the [-1., 1.] range to
    discrete values in [0, n_bits**2 - 1]
    """
    range_quantize = 2**n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize)  # clip values
    if apply_round:
        data_quantize = data_quantize.round()
    return data_quantize.astype(int)


def dequantize(
    data: np.ndarray, n_bits: int = 8, min_range: int = -1, max_range: int = 1
):
    """
    Convert quantized discrete value back into floats
    """
    range_quantize = 2**n_bits - 1
    data = data.astype("float32")
    data = data * (max_range - min_range) / range_quantize + min_range
    return data


def reconstruct(
    x: np.ndarray,
    bit: int,
    min_range: int = -1,
    max_range: int = 1,
    apply_round: bool = False,
):
    """Quantize and dequantize the input data.

    Args:
        x (np.ndarray): input data
        bit (int): quantization bit
        apply_round (bool): whether to round the quantized data before casting to int

    Returns:
        x_recon (np.ndarray): reconstructed data
    """
    x_bit = quantize(x, bit, min_range, max_range, apply_round=apply_round)
    x_recon = dequantize(x_bit, bit, min_range, max_range)
    return x_recon


def quantize_pos(face_pos, edge_pos, bit=10):
    """
    Qunatize bbox positions
    """
    face_pos = quantize(
        face_pos,
        n_bits=bit,
        min_range=-1.0,
        max_range=1.0,
        apply_round=True,
    )
    edge_pos = quantize(
        edge_pos,
        n_bits=bit,
        min_range=-1.0,
        max_range=1.0,
        apply_round=True,
    )
    return face_pos, edge_pos
