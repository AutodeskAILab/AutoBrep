import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers import DDPMScheduler, PNDMScheduler

from occwl.compound import Compound
from occwl.io import save_step
from occwl.solid import Solid

from autobrep.data.simple_point_grid_data_file import (
    SimplePointGridDataWriter,
    convert_face_edge_adj_to_index_arrays,
)
from autobrep.data.token_mapping import MMTokenIndex
from autobrep.inference.post_process import AutoBrepPostProcess
from autobrep.models.dataclass import BrepGenCAD
from autobrep.point_grid_visualizer import PointGridVisualizer
from autobrep.utils import pad_zero, quantize


def load_eval_args(mode: str):
    # Load evaluation config
    with open("configs/inference/eval_config.yaml", "r") as fp:
        config = yaml.safe_load(fp)
    eval_args = config[mode]
    return eval_args


def save_debug_images(cad_data: BrepGenCAD, face_path: Path, edge_path: Path):
    viz = PointGridVisualizer(max_surfaces=len(cad_data.face_pos_cad))
    viz.new_figure()
    viz.plot_surface_point_grids(cad_data.face_pos_cad, cad_data.face_ncs_cad)
    viz.plot_edge_point_grids(
        cad_data.edge_pos_cad_legacy,
        cad_data.edge_ncs_cad_legacy,
        cad_data.edge_mask_cad,
    )
    viz.save_image(str(face_path))
    viz.new_figure()
    viz.plot_edge_point_grids(
        cad_data.edge_pos_cad_legacy,
        cad_data.edge_ncs_cad_legacy,
        cad_data.edge_mask_cad,
    )
    viz.plot_vertex_point(cad_data.vertex_cad_legacy, cad_data.edge_mask_cad)
    viz.save_image(str(edge_path))


def save_point_grid(output_path: Path, cad_data: BrepGenCAD):
    point_grid_writer = SimplePointGridDataWriter()
    point_grid_writer.write_v2_point_grids(
        str(output_path),
        cad_data.face_pos_cad,
        cad_data.face_ncs_cad,
        cad_data.edge_pos_cad,
        cad_data.edge_ncs_cad,
        np.logical_not(cad_data.edge_mask_cad),
    )


def save_point_grid_joint_optimized(
    output_path: Path, cad_data: BrepGenCAD, dist_threshold: float
):
    # Copy the points before aplying any processing
    cad_data = deepcopy(cad_data)
    device = cad_data.face_ncs_cad.device
    post_process_agent = AutoBrepPostProcess(
        device=device, dist_threshold=dist_threshold, eval_mode=False
    )
    face_edge_adjacency = ~cad_data.edge_mask_cad
    try:
        unique_vertices, shared_vertex_dict, edge_vertex_adjacency = (
            post_process_agent.compute_shared_vertex(cad_data, face_edge_adjacency)
        )
    except Exception as ex:
        print(f"Joint optimize failed with excpetion {type(ex).__name__}: {ex}")
        return

    # Optimize
    face_wcs, coedge_wcs = post_process_agent.optimize(
        cad_data,
        unique_vertices,
        edge_vertex_adjacency,
        face_edge_adjacency,
    )
    point_grid_writer = SimplePointGridDataWriter()

    num_faces = face_wcs.shape[0]
    num_edges = face_edge_adjacency.shape[1]
    num_coedges = coedge_wcs.shape[0]
    assert face_edge_adjacency.shape[0] == num_faces, (
        "Check number of faces is consistent"
    )
    assert num_coedges == np.sum(face_edge_adjacency), (
        "Number of coedges must equal the total number of face-edge adjacencies"
    )

    # The following double loop is essentially trying to undo the process
    # which was used to duplicate the edges to build the coedges
    #
    # We loop over the rows of the matrix
    coedge_offset = 0
    deduplicated_edge_indices = np.zeros((num_edges), dtype=np.int64)
    for coedge_row in face_edge_adjacency:
        # These are the indices of the edges around a given face
        # The coedge array was created by concatenating the edge
        # data from these indices
        edge_indices = np.where(coedge_row)[0]
        for offset, edge_index in enumerate(edge_indices):
            # The coedge offset keeps track of the previous coedges
            # which were added to list when the data was duplicated
            coedge_index = offset + coedge_offset

            # Here we are doing the deduplication by overwriting the
            # elements in the array.  i.e. we will write a coedge
            # index for each edge twice
            deduplicated_edge_indices[edge_index] = coedge_index

        # Finally we update the coedge offset.
        coedge_offset += len(edge_indices)

    assert deduplicated_edge_indices.ndim == 1, "Should have one index for each edge"
    assert deduplicated_edge_indices.shape[0] == num_edges, (
        "Check we have the correct number of edges"
    )
    assert deduplicated_edge_indices.max() < num_coedges, "Check no index out of bounds"

    # Now extract the deduplicated grids for the edges
    edge_wcs = coedge_wcs[deduplicated_edge_indices]
    edge_face_incidence = convert_face_edge_adj_to_index_arrays(face_edge_adjacency)

    point_grid_writer.save_world_point_grid_data(
        output_path, face_wcs, edge_wcs, edge_face_incidence
    )


def reconstruct_compound(
    cad_data: BrepGenCAD,
    builders: List,
) -> Tuple[Optional[Solid], List[str]]:
    """
    Reconstruct a compound from the CAD data using the provided builders.
    Each builder is tried in turn until one is successful.

    Arguments:
        cad_data: The CAD data to reconstruct.
        builders: An ordered list of builders to try.

    Returns:
        compound: The reconstructed compound if successful, otherwise None
        errors: A list of error messages from each failed builder.
    """
    errors = []
    for builder in builders:
        return builder.rebuild_brep(cad_data)