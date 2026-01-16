import os
from pathlib import Path

import numpy as np

from autobrep.utils import (
    compute_bbox_center_and_size,
)


def convert_face_edge_adj_to_index_arrays(face_edge_adj: np.ndarray) -> np.ndarray:
    """
    This function is used to convert a face-edge adjacency matrix into
    the edge_face_incidence array
    Args:
        face_edge_adj: (np.ndarray)
            A boolean matrix with shape (num_faces, num_edges)
            Values are True if an edge is adjacent to a face and False if not

    Returns:
        (np.ndarray)  A matrix of integer indices with shape (num_edges, 2)
        Each row the the matrix contains the indices of the faces adjacent to the
        edge.   If an edge has only one adjacent face then -1 is used to pad
    """
    assert isinstance(face_edge_adj, np.ndarray), "Should be a numpy array"
    assert face_edge_adj.ndim == 2, "Should be (num_faces, num_edges)"
    num_faces_attached_to_each_edge = np.sum(face_edge_adj, axis=0)
    assert num_faces_attached_to_each_edge.min() > 0, (
        "Should have at least one face attached to each edge"
    )
    assert num_faces_attached_to_each_edge.max() <= 2, (
        "Warning!  More than two faces attached to edge (non-manifold solid generated)"
    )

    num_faces = face_edge_adj.shape[0]
    num_edges = face_edge_adj.shape[1]

    edge_face_incidence = []
    for face_attached_to_edge_flags in np.transpose(face_edge_adj):
        assert face_attached_to_edge_flags.shape == (num_faces,), (
            "Should be a flag for each face"
        )

        # Get the indices of the faces
        face_indices = np.where(face_attached_to_edge_flags)[0]
        if face_indices.shape[0] > 2:
            face_indices = face_indices[0:2]
        if face_indices.shape[0] < 2:
            assert face_indices.shape[0] == 1, (
                "All edges should be attached to at least one face"
            )
            face_indices = np.concatenate([face_indices, np.array([-1])])
        edge_face_incidence.append(face_indices)

    edge_face_incidence = np.stack(edge_face_incidence)
    assert edge_face_incidence.shape == (num_edges, 2), "Check shape of output"
    assert edge_face_incidence.max() < num_faces, "Check indices in range"
    assert edge_face_incidence.min() >= -1, (
        "Check indices in range.  -1 may indicate an open edge"
    )
    return edge_face_incidence


def convert_normalized_face_points_to_world(
    face_bbox_world: np.ndarray, face_points_normalized: np.ndarray
):
    """
    Convert normalized face points to the world coordinate
    system by scaling and translating based on the bounding
    box
    """
    assert face_bbox_world.ndim == 2, "Expect (num_faces, 6)"
    num_faces = face_bbox_world.shape[0]
    assert face_points_normalized.shape[0] == num_faces, (
        "Check the number of faces is consistent"
    )
    assert face_points_normalized.ndim == 4, "Expect (num_faces, num_u, num_v, 3)"

    num_u = face_points_normalized.shape[1]
    num_v = face_points_normalized.shape[2]

    face_points_world = []

    # Loop over the faces
    for face_bbox, face_points_ncs in zip(face_bbox_world, face_points_normalized):
        center, size = compute_bbox_center_and_size(face_bbox[0:3], face_bbox[3:])
        ncs = face_points_ncs.reshape(-1, 3)
        wcs = ncs * (size / 2) + center
        wcs = wcs.reshape(num_u, num_v, 3)
        face_points_world.append(wcs)

    face_points_world = np.stack(face_points_world)
    assert face_points_world.shape == (
        num_faces,
        num_u,
        num_v,
        3,
    ), "Check shape of result"
    return face_points_world


def convert_normalized_edge_points_to_world(
    edge_bbox_world: np.ndarray, edge_points_normalized: np.ndarray
):
    """
    Convert normalized edge points to the world coordinate
    system by scaling and translating based on the bounding
    box
    """
    assert edge_bbox_world.ndim == 2, "Expect (num_edges, 6)"
    num_edges = edge_bbox_world.shape[0]
    assert edge_points_normalized.shape[0] == num_edges, "Check num edges consistent"
    assert edge_points_normalized.ndim == 3, "Expect (num_edges, num_u, 3)"

    edge_points_world = []
    for edge_bbox, edge_points_ncs in zip(edge_bbox_world, edge_points_normalized):
        center, size = compute_bbox_center_and_size(edge_bbox[0:3], edge_bbox[3:])
        wcs = edge_points_ncs * (size / 2) + center
        edge_points_world.append(wcs)
    edge_points_world = np.stack(edge_points_world)
    assert edge_points_world.shape[0] == num_edges, "Check num edges"
    return edge_points_world


def convert_v2_network_output_to_ragged_array(
    face_bbox_world: np.ndarray,
    face_points_normalized: np.ndarray,
    edge_bbox_world: np.ndarray,
    edge_points_normalized: np.ndarray,
    face_edge_adj: np.ndarray,
):
    """
    This function is used to convert V2 network output to the
    ragged arrays used by legacy OCC rebuilders like the
    TolerantBrepBuilder.

        Args:
            face_bbox_world.shape = (num_faces, 6)
                global face position bounding boxes

            face_points_normalized.shape = (num_faces, num_points_u, num_points_v, 3)
                normalized local face uv points

            edge_bbox_world.shape = (num_edges, 6)
                global edge position bounding boxes

            edge_points_normalized.shape = (num_edges, num_points_u, 3)
                normalized local edge uv points

            face_edge_adj.shape = (num_faces, num_edges)
                Adjacency matrix between faces and edges

        Returns:
            face_points_normalized.shape = (num_faces, num_u, num_v, 3)
                The face points in world coordinates

            coedges_for_faces: List[np.ndarray]   len(coedges_for_faces) == num_faces
                This is a list of arrays.  Each element in the list corresponds
                to the trimming curves for a face.   The trimming curve arrays have
                shape = (num_coedges_on_face, num_u, 3)
    """
    assert isinstance(face_bbox_world, np.ndarray), "Should be a numpy array"
    assert isinstance(face_points_normalized, np.ndarray), "Should be a numpy array"
    assert isinstance(edge_bbox_world, np.ndarray), "Should be a numpy array"
    assert isinstance(edge_points_normalized, np.ndarray), "Should be a numpy array"
    assert isinstance(face_edge_adj, np.ndarray), "Should be a numpy array"
    assert face_bbox_world.ndim == 2, "Should be (num_faces, 6)"
    assert face_points_normalized.ndim == 4, (
        "Should be (num_faces, num_pts_u, num_pts_v, 3)"
    )
    num_faces = face_bbox_world.shape[0]
    assert num_faces == face_points_normalized.shape[0], "Check num faces"
    assert edge_bbox_world.ndim == 2, "Should be (num_edges, 6)"
    assert edge_points_normalized.ndim == 3, "Should be (num_edges, num_pts, 3)"
    assert face_edge_adj.ndim == 2, "Should be (num_faces, num_edges)"
    num_edges = edge_bbox_world.shape[0]
    assert edge_points_normalized.shape[0] == num_edges, "Check num edges"
    assert face_edge_adj.shape[0] == num_faces, "Check num faces"
    assert face_edge_adj.shape[1] == num_edges, "Check num edges"

    face_points_world = convert_normalized_face_points_to_world(
        face_bbox_world, face_points_normalized
    )
    edge_points_world = convert_normalized_edge_points_to_world(
        edge_bbox_world, edge_points_normalized
    )

    coedges_for_faces = []
    for coedge_flags in face_edge_adj:
        # This gives an array of indices into the edge list
        coedge_indices = np.where(coedge_flags)
        coedges_for_face = edge_points_world[coedge_indices]
        coedges_for_faces.append(coedges_for_face)
    assert len(coedges_for_faces) == num_faces, (
        "Check the number of faces is consistent"
    )

    return face_points_world, coedges_for_faces


class SimplePointGridDataWriter:
    """
    Saves the generated data in the npz format described at the top
    of the file.
    """

    def write_v2_point_grids(
        self,
        pathname: Path,
        face_bbox_world: np.ndarray,
        face_points_normalized: np.ndarray,
        edge_bbox_world: np.ndarray,
        edge_points_normalized: np.ndarray,
        face_edge_adj: np.ndarray,
    ):
        """
        Save point grid data to npz file

        Args:
            pathname: saved npz path

            face_bbox_world.shape = (num_faces, 6)
                global face position bounding boxes

            face_points_normalized.shape = (num_faces, num_points_u, num_points_v, 3)
                normalized local face uv points

            edge_bbox_world.shape = (num_edges, 6)
                global edge position bounding boxes

            edge_points_normalized.shape = (num_edges, num_points_u, 3)
                normalized local edge uv points

            face_edge_adj.shape = (num_faces, num_edges)
                Adjacency matrix between faces and edges
        """
        assert isinstance(face_bbox_world, np.ndarray), "Should be a numpy array"
        assert isinstance(face_points_normalized, np.ndarray), "Should be a numpy array"
        assert isinstance(edge_bbox_world, np.ndarray), "Should be a numpy array"
        assert isinstance(edge_points_normalized, np.ndarray), "Should be a numpy array"
        assert isinstance(face_edge_adj, np.ndarray), "Should be a numpy array"
        assert face_bbox_world.ndim == 2, "Should be (num_faces, 6)"
        assert face_points_normalized.ndim == 4, (
            "Should be (num_faces, num_pts_u, num_pts_v, 3)"
        )
        num_faces = face_bbox_world.shape[0]
        assert num_faces == face_points_normalized.shape[0], "Check num faces"
        assert edge_bbox_world.ndim == 2, "Should be (num_edges, 6)"
        assert edge_points_normalized.ndim == 3, "Should be (num_edges, num_pts, 3)"
        assert face_edge_adj.ndim == 2, "Should be (num_faces, num_edges)"
        num_edges = edge_bbox_world.shape[0]
        assert edge_points_normalized.shape[0] == num_edges, "Check num edges"
        assert face_edge_adj.shape[0] == num_faces, "Check num faces"
        assert face_edge_adj.shape[1] == num_edges, "Check num edges"

        face_points_world = convert_normalized_face_points_to_world(
            face_bbox_world, face_points_normalized
        )
        edge_points_world = convert_normalized_edge_points_to_world(
            edge_bbox_world, edge_points_normalized
        )
        edge_face_incidence = convert_face_edge_adj_to_index_arrays(face_edge_adj)
        self.save_world_point_grid_data(
            pathname, face_points_world, edge_points_world, edge_face_incidence
        )

    def save_world_point_grid_data(
        self,
        pathname: Path,
        face_points_world: np.ndarray,
        edge_points_world: np.ndarray,
        edge_face_incidence: np.ndarray,
    ):
        """
        Save point grids which have already been converted to world
        coordinates.

        face_points_world:
            shape = (num_faces, num_u, num_v, 3)   dtype=np.float32
            These are the main point grids defining surfaces in the model.
            The grids have size (num_u, num_v) and the last dimension represents 3d points.

        edge_points_world:
            shape = (num_edges, num_u, 3)  dtype=np.float32
            These are grids representing *unique* edges.
            For V2 these will not be duplicated for each edge in the input data.

        edge_face_incidence:
            shape = (num_edges, 2)  dtype=np.int64
            These are indices into the array of faces, indicated the faces
            which share the given edge.
            Linked and manifold edges will have two valid faces with face indices
            in the range 0 <= i < num_faces.
            Open edges, which have only one adjacent face, will have one valid index in the first position
            and -1 in the second position.
        """
        num_faces = face_points_world.shape[0]
        num_edges = edge_points_world.shape[0]
        assert edge_face_incidence.max() < num_faces, "Check face indices in range"
        assert edge_face_incidence.min() >= -1, (
            "Check face indices in range (-1 indicates an open edge)"
        )
        assert edge_face_incidence.shape == (
            num_edges,
            2,
        ), "Check edge_face_incidence has shape (num_edges, 2)"

        face_points_world = face_points_world.astype(np.float32)
        edge_points_world = edge_points_world.astype(np.float32)
        edge_face_incidence = edge_face_incidence.astype(np.int64)
        np.savez(
            str(pathname),
            face_points_world=face_points_world,
            edge_points_world=edge_points_world,
            edge_face_incidence=edge_face_incidence,
        )


class SimplePointGridDataReader:
    """
    Reads the NPZ data into the members

    Total number of faces, edges
        self.face_count, self.edge_count

    A tensor of face points
        self.face_points_world.shape = (num_faces, num_points_u, num_points_v, 3) dtype=np.float32

    A tensor of edge points
        self.edge_points_world.shape = (num_edges, num_u, 3)  dtype=np.float32

    The topology as an edge-face incidence index list
        self.edge_face_incidence. shape = (num_edges, 2)  dtype=np.int64
    """

    def __init__(self, pathname: Path):
        # check if file exist
        assert os.path.exists(pathname), f"File {pathname} does not exist"
        with np.load(pathname, allow_pickle=False) as npz_data:
            self.face_points_world = npz_data["face_points_world"]
            self.edge_points_world = npz_data["edge_points_world"]
            self.edge_face_incidence = npz_data["edge_face_incidence"]

            self.face_count = len(self.face_points_world)
            self.edge_count = len(self.edge_points_world)

    def get_coedges_of_faces(self):
        """
        Get a list containing the coedges which trim each face

        Returns
            List[np.ndarray]   len(coedges_for_faces) == num_faces
                This is a list of arrays.  Each element in the list corresponds
                to the trimming curves for a face.   The trimming curve arrays have
                shape = (num_coedges_on_face, num_u, 3)
        """
        num_u = self.edge_points_world.shape[1]
        coedges_for_faces = [[] for i in range(self.face_count)]
        for edge_points, edge_data in zip(
            self.edge_points_world, self.edge_face_incidence
        ):
            face1_index = edge_data[0]
            face2_index = edge_data[1]
            coedges_for_faces[face1_index].append(edge_points)
            if face2_index >= 0:
                coedges_for_faces[face2_index].append(edge_points)

        for index, edge_points in enumerate(coedges_for_faces):
            if len(edge_points) > 0:
                edge_points = np.stack(edge_points)
            else:
                edge_points = np.zeros((0, num_u, 3))
            coedges_for_faces[index] = edge_points

        assert len(coedges_for_faces) == self.face_count, "Check number of faces"
        return coedges_for_faces
