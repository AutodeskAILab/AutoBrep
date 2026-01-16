from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class BrepGenCAD:
    vertex_cad_legacy: np.ndarray
    edge_mask_cad: np.ndarray
    edge_pos_cad: np.ndarray
    edge_ncs_cad: np.ndarray
    face_ncs_cad: np.ndarray
    face_pos_cad: np.ndarray

    # Legacy data.   Should be removed when possible!
    edge_pos_cad_legacy: Optional[np.ndarray] = None
    edge_ncs_cad_legacy: Optional[np.ndarray] = None
    edge_z_cad: Optional[np.ndarray] = None

    def mask_data(self, face_mask_cad):
        return BrepGenCAD(
            vertex_cad_legacy=self.vertex_cad_legacy[~face_mask_cad],
            edge_mask_cad=self.edge_mask_cad[~face_mask_cad],
            edge_pos_cad=self.edge_pos_cad[~face_mask_cad],
            edge_ncs_cad=self.edge_ncs_cad[~face_mask_cad],
            face_ncs_cad=self.face_ncs_cad[~face_mask_cad],
            face_pos_cad=self.face_pos_cad[~face_mask_cad],
            edge_pos_cad_legacy=self.edge_pos_cad_legacy[~face_mask_cad]
            if self.edge_pos_cad_legacy is not None
            else None,
            edge_ncs_cad_legacy=self.edge_ncs_cad_legacy[~face_mask_cad]
            if self.edge_ncs_cad_legacy is not None
            else None,
            edge_z_cad=self.edge_z_cad[~face_mask_cad]
            if self.edge_z_cad is not None
            else None,
        )


@dataclass
class CheckpointPaths:
    @classmethod
    def from_folder(
        cls, folder: Union[str, Path], checkpoints: Optional["CheckpointPaths"] = None
    ):
        if checkpoints is None:
            checkpoints = cls()

        protocol = ""
        if str(folder).startswith("s3://"):
            protocol = "s3://"
            folder = str(folder)[len(protocol) :]

        folder = Path(folder)
        return cls(*(protocol + str(folder / checkpoint) for checkpoint in checkpoints))

    def __iter__(self):
        return iter(tuple())
