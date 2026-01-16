from typing import Optional, Tuple

import numpy as np

from autobrep.models.dataclass import BrepGenCAD
from autobrep.utils import (
    batch_ncs2wcs,
    detect_shared_edge,
    detect_shared_vertex,
    joint_optimize,
)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class PostProcess:
    """
    Parent class to support some common functionality between post processors.
    """

    def __init__(
        self,
        device: str,
        eval_mode: bool = False,
    ):
        """
        Initialize the optimizer.
        Args:
            device: The device to run the optimizer on.
        """
        self.device = device
        self.eval_mode = eval_mode

    def __str__(self):
        return (
            f"{self.__class__.__name__}(device={self.device},"
            f" eval_mode={self.eval_mode})"
        )

    @staticmethod
    def eval_optimize_bypass(data: BrepGenCAD):
        face_wcs = batch_ncs2wcs(data.face_ncs_cad, data.face_pos_cad)
        edge_wcs = batch_ncs2wcs(
            data.edge_ncs_cad[~data.edge_mask_cad],
            data.edge_pos_cad[~data.edge_mask_cad],
        )
        return face_wcs, edge_wcs


class SimplePostProcess(PostProcess):
    """
    Class for running post-process optimization algorithms for brepgen.
    """

    def __init__(
        self,
        device: str,
        dist_threshold: float = 0.05,
        z_threshold: float = 0.2,
        eval_mode: bool = False,
    ):
        """
        Initialize the optimizer.
        Args:
            device: The device to run the optimizer on.
            z_threshold: The threshold for shared edge detection.
            dist_threshold: The threshold for shared vertex detection.
        """
        super().__init__(device, eval_mode=eval_mode)
        self.dist_threshold = dist_threshold
        self.z_threshold = z_threshold

    def compute_shared_edge(
        self, data: BrepGenCAD, shared_vertex_dict: dict
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Find shared edges between faces.
        Returns:
            unique_edges: Unique edges between faces.
            face_edge_adjacency: Adjacency matrix between faces and edges.
            edge_vertex_adjacency: Adjacency matrix between edges and vertices.
        """
        unique_edges, face_edge_adjacency, edge_vertex_adjacency = detect_shared_edge(
            shared_vertex_dict,
            data.edge_ncs_cad,
            data.edge_z_cad,
            data.edge_mask_cad,
            z_threshold=self.z_threshold,
        )
        return unique_edges, face_edge_adjacency, edge_vertex_adjacency

    def compute_shared_vertex(self, data: BrepGenCAD) -> Tuple[np.array, dict]:
        """
        Find shared vertices between faces.
        Returns:
            unique_vertices: Unique vertices between faces.
            shared_vertex_dict: Dictionary of shared vertices.
        """
        unique_vertices, shared_vertex_dict = detect_shared_vertex(
            data.edge_mask_cad, data.vertex_cad, self.dist_threshold
        )
        return unique_vertices, shared_vertex_dict

    def optimize(
        self,
        data: BrepGenCAD,
        unique_vertices: np.array,
        unique_edges: np.array,
        edge_vertex_adjacency: np.array,
        face_edge_adjacency: np.array,
    ) -> Tuple[np.array, np.array]:
        """
        Perform gradient based optimization.
        """
        face_count = len(data.face_ncs_cad)
        if self.eval_mode:
            print(f"{bcolors.OKGREEN}[Eval Mode No Optimization]{bcolors.ENDC}")
            return self.eval_optimize_bypass(data)

        print(f"{bcolors.OKGREEN}[joint optimization]{bcolors.ENDC}")
        return joint_optimize(
            data.face_ncs_cad,
            unique_edges,
            data.face_pos_cad,
            unique_vertices,
            edge_vertex_adjacency,
            face_edge_adjacency,
            face_count,
            self.device,
        )

    def data_to_str(
        self,
        data: BrepGenCAD,
        unique_edges: Optional[np.array] = None,
        unique_vertices: Optional[np.array] = None,
    ) -> str:
        return f"SimplePostProcess: F-{len(data.vertex_cad)} E-{len(unique_edges)} V-{len(unique_vertices)}"


class AutoBrepPostProcess(PostProcess):
    """
    Class for running post-process optimization algorithms for brepformer
    """

    def __init__(
        self, device: str, dist_threshold: float = 0.05, eval_mode: bool = False
    ):
        super().__init__(device, eval_mode=eval_mode)
        self.dist_threshold = dist_threshold

    def compute_shared_vertex(
        self, data: BrepGenCAD, face_edge_adjacency: np.array
    ) -> Tuple[np.array, dict, np.array]:
        """
        Find shared vertices between faces.

        Returns:
            unique_vertices: Unique vertices between faces.
            shared_vertex_dict: Dictionary of shared vertices.
            edge_vertex_adjacency: Adjacency matrix between edges and vertices.
        """
        unique_vertices, shared_vertex_dict = detect_shared_vertex(
            None, data.vertex_cad_legacy, self.dist_threshold, face_edge_adjacency
        )
        # re-assign edge start/end to unique vertices
        new_ids = []
        for old_id in np.arange(2 * face_edge_adjacency.sum(1).sum()):
            new_id = []
            for key, value in shared_vertex_dict.items():
                # Check if the desired number is in the associated list
                if old_id in value:
                    new_id.append(key)
            assert len(new_id) == 1, "should only return one unique value"
            new_ids.append(new_id[0])
        edge_vertex_adjacency = np.array(new_ids).reshape(-1, 2)
        return unique_vertices, shared_vertex_dict, edge_vertex_adjacency

    def optimize(
        self,
        data: BrepGenCAD,
        unique_vertices: np.array,
        edge_vertex_adjacency: np.array,
        face_edge_adjacency: np.array,
    ) -> Tuple[np.array, np.array]:
        """
        Perform gradient based optimization.

        Returns:
            face_wcs: World coordinates of faces.
            edge_wcs: World coordinates of edges.
        """
        face_count = len(data.face_ncs_cad)
        if self.eval_mode:
            # Directly compute wcs from ncs and pos in eval mode
            print(f"{bcolors.OKGREEN}[Eval Mode No Optimization]{bcolors.ENDC}")
            return self.eval_optimize_bypass(data)

        print(f"{bcolors.OKGREEN}[joint optimization]{bcolors.ENDC}")
        return joint_optimize(
            data.face_ncs_cad,
            data.edge_ncs_cad_legacy[~data.edge_mask_cad],
            data.face_pos_cad,
            unique_vertices,
            edge_vertex_adjacency,
            face_edge_adjacency,
            face_count,
            self.device,
        )

    def data_to_str(
        self, data: BrepGenCAD, unique_vertices: Optional[np.array] = None
    ) -> str:
        return f"BrepFormerPostProcess: F-{len(data.face_ncs_cad)} E-{len(data.edge_ncs_cad)} V-{len(unique_vertices)}"
