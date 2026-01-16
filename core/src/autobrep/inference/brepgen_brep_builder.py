
from typing import List

import numpy as np
import torch
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.ShapeFix import ShapeFix_Edge, ShapeFix_Face, ShapeFix_Wire
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TopoDS import (
    TopoDS_Compound,
    TopoDS_Shell,
    TopoDS_Solid,
)
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer

from occwl.compound import Compound
from occwl.solid import Solid

from autobrep.inference.post_process import (
    AutoBrepPostProcess,
    SimplePostProcess,
)
from autobrep.models.dataclass import BrepGenCAD
from autobrep.utils import get_bbox_diag_distance


def convert_to_topods_face(bspline_surface):
    face = BRepBuilderAPI_MakeFace(bspline_surface, 1e-6).Face()
    return face


def convert_to_topods_edge(bspline_edge):
    edge = BRepBuilderAPI_MakeEdge(bspline_edge).Edge()
    return edge


def create_shell(faces):
    # Initialize a BRep_Builder and a TopoDS_Shell
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    # Add each face to the shell
    for face in faces:
        builder.Add(shell, face)
    return shell


def create_compound(faces):
    # Initialize a BRep_Builder and a TopoDS_Compound
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    # Add faces to the compound
    for face in faces:
        builder.Add(compound, face)
    return compound


def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)


def fix_wires(face, debug=False):
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        if debug:
            wire_checker = ShapeAnalysis_Wire(wire, face, 0.01)
            print(f"Check order 3d {wire_checker.CheckOrder()}")
            print(f"Check 3d gaps {wire_checker.CheckGaps3d()}")
            print(f"Check closed {wire_checker.CheckClosed()}")
            print(f"Check connected {wire_checker.CheckConnected()}")
        wire_fixer = ShapeFix_Wire(wire, face, 0.01)

        # wire_fixer.SetClosedWireMode(True)
        # wire_fixer.SetFixConnectedMode(True)
        # wire_fixer.SetFixSeamMode(True)

        assert wire_fixer.IsReady()
        ok = wire_fixer.Perform()  # noqa
        # assert ok


def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()  # noqa
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face


class BrepGenBrepBuilder():
    def __init__(
        self,
        device: torch.device,
        z_threshold: float,
        sewing_tolerance: float = 0.01,
        eval_mode: bool = False,
        surface_degree_min: int = 3,
        surface_degree_max: int = 8,
        surface_tolerance: float = 5e-2,
        edge_degree_min: int = 0,
        edge_degree_max: int = 8,
    ):
        """
        Create the B-Rep builder

        Args:

        """
        self.sewing_tolerance = sewing_tolerance
        self.post_process_agent = SimplePostProcess(
            device, z_threshold=z_threshold, eval_mode=eval_mode
        )
        self.surface_degree_min = surface_degree_min
        self.surface_degree_max = surface_degree_max
        self.surface_tolerance = surface_tolerance
        self.edge_degree_min = edge_degree_min
        self.edge_degree_max = edge_degree_max
        self.device = device

    def rebuild_surfaces(
        self, face_wcs: np.array
    ) -> List[GeomAPI_PointsToBSplineSurface]:
        """
        Reconstruct brep surfaces.
        """
        # Fit surface bspline
        rebuilt_faces = []
        for points in face_wcs:
            num_u_points, num_v_points = points.shape[0], points.shape[1]
            uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
            for u_index in range(1, num_u_points + 1):
                for v_index in range(1, num_v_points + 1):
                    pt = points[u_index - 1, v_index - 1]
                    point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                    uv_points_array.SetValue(u_index, v_index, point_3d)
            approx_face = GeomAPI_PointsToBSplineSurface(
                uv_points_array,
                self.surface_degree_min,
                self.surface_degree_max,
                GeomAbs_C2,
                self.surface_tolerance,
            ).Surface()
            rebuilt_faces.append(approx_face)
        return rebuilt_faces

    def rebuild_curves(self, edge_wcs: np.array) -> List[GeomAPI_PointsToBSpline]:
        """
        Reconstruct brep cruves.
        """
        rebuilt_edges = []
        for points in edge_wcs:
            num_u_points = points.shape[0]
            u_points_array = TColgp_Array1OfPnt(1, num_u_points)
            for u_index in range(1, num_u_points + 1):
                pt = points[u_index - 1]
                point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                u_points_array.SetValue(u_index, point_2d)
            try:
                approx_edge = GeomAPI_PointsToBSpline(
                    u_points_array,
                    self.edge_degree_min,
                    self.edge_degree_max,
                    GeomAbs_C2,
                    5e-3,
                ).Curve()
            except Exception:
                print("high precision failed, trying mid precision...")
                try:
                    approx_edge = GeomAPI_PointsToBSpline(
                        u_points_array,
                        self.edge_degree_min,
                        self.edge_degree_max,
                        GeomAbs_C2,
                        8e-3,
                    ).Curve()
                except Exception:
                    print("mid precision failed, trying low precision...")
                    approx_edge = GeomAPI_PointsToBSpline(
                        u_points_array,
                        self.edge_degree_min,
                        self.edge_degree_max,
                        GeomAbs_C2,
                        5e-2,
                    ).Curve()
            rebuilt_edges.append(approx_edge)
        return rebuilt_edges

    def linker(self, corner_indices):
        loops = []
        ordered = [0]
        seen_corners = [corner_indices[0, 0], corner_indices[0, 1]]
        next_index = corner_indices[0, 1]

        while len(ordered) < len(corner_indices):
            while True:
                next_row = [
                    idx
                    for idx, edge in enumerate(corner_indices)
                    if next_index in edge and idx not in ordered
                ]
                if len(next_row) == 0:
                    break
                ordered += next_row
                next_index = list(set(corner_indices[next_row][0]) - set(seen_corners))
                if len(next_index) == 0:
                    break
                else:
                    next_index = next_index[0]
                seen_corners += [
                    corner_indices[next_row][0][0],
                    corner_indices[next_row][0][1],
                ]

            cur_len = int(
                np.array([len(x) for x in loops]).sum()
            )  # add to inner / outer loops
            loops.append(ordered[cur_len:])

            # Swith to next loop
            next_corner = list(set(np.arange(len(corner_indices))) - set(ordered))
            if len(next_corner) == 0:
                break
            else:
                next_corner = next_corner[0]
            next_index = corner_indices[next_corner][0]
            ordered += [next_corner]
            seen_corners += [
                corner_indices[next_corner][0],
                corner_indices[next_corner][1],
            ]
            next_index = corner_indices[next_corner][1]
        return loops, ordered

    def rebuild_face_edge(
        self,
        face_wcs: np.array,
        edge_wcs: np.array,
        face_edge_adjacency: np.array,
        edge_vertex_adjacency: np.array,
    ):
        """
        Trim and cut the faces by edge wire
        rebuild the faces and edges
        """
        # rebuild faces and edges
        rebuilt_faces = self.rebuild_surfaces(face_wcs)
        rebuilt_edges = self.rebuild_curves(edge_wcs)
        # Create edges from the curve list
        edge_list = []
        for curve in rebuilt_edges:
            edge = BRepBuilderAPI_MakeEdge(curve).Edge()
            edge_list.append(edge)

        # Cut surface by wire
        post_faces = []
        post_edges = []
        for surface, edge_incides in zip(rebuilt_faces, face_edge_adjacency):
            face_occ, edge_post = self._cut_surface_by_wire(
                surface,
                edge_list,
                edge_wcs,
                edge_incides,
                edge_vertex_adjacency,
            )
            post_faces.append(face_occ)
            post_edges += edge_post
        return (post_faces, post_edges)

    def _cut_surface_by_wire(
        self,
        surface,
        edge_list,
        edge_wcs,
        edge_incides,
        edge_vertex_adjacency,
    ):
        corner_indices = edge_vertex_adjacency[edge_incides]

        # ordered loop
        loops, ordered = self.linker(corner_indices)

        # Determine the outer loop by bounding box length (?)
        bbox_spans = [get_bbox_diag_distance(edge_wcs[x].reshape(-1, 3)) for x in loops]

        # Create wire from ordered edges
        _edge_incides_ = [edge_incides[x] for x in ordered]
        edge_post = [edge_list[x] for x in _edge_incides_]

        out_idx = np.argmax(np.array(bbox_spans))
        inner_idx = list(set(np.arange(len(loops))) - set([out_idx]))

        # Outer wire
        wire_builder = BRepBuilderAPI_MakeWire()
        for edge_idx in loops[out_idx]:
            wire_builder.Add(edge_list[edge_incides[edge_idx]])
        outer_wire = wire_builder.Wire()

        # Inner wires
        inner_wires = []
        for idx in inner_idx:
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_idx in loops[idx]:
                wire_builder.Add(edge_list[edge_incides[edge_idx]])
            inner_wires.append(wire_builder.Wire())

        # Cut by wires
        face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire)
        for wire in inner_wires:
            face_builder.Add(wire)
        face_occ = face_builder.Shape()
        fix_wires(face_occ)
        add_pcurves_to_edges(face_occ)
        fix_wires(face_occ)
        face_occ = fix_face(face_occ)
        return face_occ, edge_post

    def rebuild_solid(
        self,
        edge_wcs: np.array,
        face_wcs: np.array,
        edge_vertex_adjacency: np.array,
        face_edge_adjacency: np.array,
    ) -> TopoDS_Solid:
        """
        Trim and cut the faces by edge wire, rebuild the solid
        """
        # rebuild faces and edges
        faces, edges = self.rebuild_face_edge(
            face_wcs, edge_wcs, face_edge_adjacency, edge_vertex_adjacency
        )
        # sew faces into solid
        sewing = BRepBuilderAPI_Sewing(self.sewing_tolerance)
        for face in faces:
            sewing.Add(face)
        sewing.Perform()
        sewn_shape = sewing.SewedShape()
        if isinstance(sewn_shape, TopoDS_Shell):
            maker = BRepBuilderAPI_MakeSolid()
            maker.Add(sewn_shape)
            maker.Build()
            compound = maker.Solid()
        else:
            # Typically if we don't sew to make a shell
            # then we will create a compound
            compound = sewn_shape
        return compound

    def rebuild_brep(self, data: BrepGenCAD):
        """
        Rebuild the B-Rep
        """
        # Merge shared vertex and edge
        unique_vertices, shared_vertex_dict = (
            self.post_process_agent.compute_shared_vertex(data)
        )

        unique_edges, face_edge_adjacency, edge_vertex_adjacency = (
            self.post_process_agent.compute_shared_edge(data, shared_vertex_dict)
        )
      
        # Joint optimize
        face_wcs, edge_wcs = self.post_process_agent.optimize(
            data,
            unique_vertices,
            unique_edges,
            edge_vertex_adjacency,
            face_edge_adjacency,
        )

        solid = self.rebuild_solid(
            edge_wcs, face_wcs, edge_vertex_adjacency, face_edge_adjacency
        )
     
        return Solid(solid)


class AutoBrepBuilder(BrepGenBrepBuilder):
    def __init__(
        self,
        device: torch.device,
        vertex_threshold: float,
        z_threshold: float = None,
        sewing_tolerance: float = 0.01,
        surface_degree_min: int = 3,
        surface_degree_max: int = 8,
        surface_tolerance: float = 5e-2,
        edge_degree_min: int = 0,
        edge_degree_max: int = 8,
        eval_mode: bool = False,
    ):
        super().__init__(
            device,
            eval_mode=eval_mode,
            z_threshold=0.0,  # dummy value because post_process_agent is overwritten
            sewing_tolerance=sewing_tolerance,
            surface_degree_min=surface_degree_min,
            surface_degree_max=surface_degree_max,
            surface_tolerance=surface_tolerance,
            edge_degree_min=edge_degree_min,
            edge_degree_max=edge_degree_max,
        )
        self.post_process_agent = AutoBrepPostProcess(
            device=device, dist_threshold=vertex_threshold, eval_mode=eval_mode
        )

    def rebuild_face_edge(
        self,
        face_wcs: np.array,
        edge_wcs: np.array,
        face_edge_adjacency: np.array,
        edge_vertex_adjacency: np.array,
    ):
        """
        Trim and cut the faces by edge wire
        rebuild the faces and edges
        """
        # rebuild faces and edges
        rebuilt_faces = self.rebuild_surfaces(face_wcs)
        rebuilt_edges = self.rebuild_curves(edge_wcs)
        # Create edges from the curve list
        edge_list = []
        for curve in rebuilt_edges:
            edge = BRepBuilderAPI_MakeEdge(curve).Edge()
            edge_list.append(edge)

        # Cut surface by wire
        post_faces = []
        post_edges = []
        offsets = np.concatenate((np.array([0]), np.cumsum(face_edge_adjacency.sum(1))))

        for idx, surface in enumerate(rebuilt_faces):
            edge_incides = np.arange(offsets[idx], offsets[idx + 1])

            face_occ, edge_post = self._cut_surface_by_wire(
                surface,
                edge_list,
                edge_wcs,
                edge_incides,
                edge_vertex_adjacency,
            )
            post_faces.append(face_occ)
            post_edges += edge_post
        return (post_faces, post_edges)

    def rebuild_brep(self, data: BrepGenCAD):
        """
        Rebuild the B-Rep
        """
        face_edge_adjacency = ~data.edge_mask_cad
        try:
            unique_vertices, shared_vertex_dict, edge_vertex_adjacency = (
                self.post_process_agent.compute_shared_vertex(data, face_edge_adjacency)
            )
        except Exception:
            return None

        # Optimize
        face_wcs, edge_wcs = self.post_process_agent.optimize(
            data,
            unique_vertices,
            edge_vertex_adjacency,
            face_edge_adjacency,
        )

        try:
            shape = self.rebuild_solid(
                edge_wcs, face_wcs, edge_vertex_adjacency, face_edge_adjacency
            )
        except RuntimeError:
            return None

        if isinstance(shape, TopoDS_Solid):
            occwl_shape = Solid(shape)
        else:
            occwl_shape = Compound(shape)

        return occwl_shape
