import abc
import functools
import random
from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union

import networkx as nx
import numpy as np
import pyarrow.compute as pc
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import AutoTokenizer

from autobrep import utils
from autobrep.data.serialize import deserialize_array
from autobrep.data.token_mapping import MMTokenIndex
import copy

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from torch.utils.data import DataLoader, IterableDataset


class ParquetRowIterable(IterableDataset):
    """
    Streams rows from a Parquet directory/file via pyarrow.dataset.

    Pipeline per row:
      pre_filter -> unpickle -> post_filter -> map_func(aug=...) -> yield item

    Supports:
      - pushdown filter (pyarrow Expression)
      - optional local shuffle buffer (approximate shuffle)
      - optional limit (stop after N yielded items)
    """

    def __init__(
        self,
        paths: str,
        columns: List[str],
        filter_expr: Optional[pc.Expression],
        pre_filter,
        unpickle,
        post_filter,
        map_func,
        aug: bool,
        limit: Optional[int] = None,
        shuffle_buffer_size: Optional[int] = None,
        shuffle_seed: int = 9876,
        batch_rows_read: int = 4096,
    ):
        super().__init__()
        self.paths = paths
        self.columns = columns
        self.filter_expr = filter_expr
        self.pre_filter = pre_filter
        self.unpickle = unpickle
        self.post_filter = post_filter
        self.map_func = map_func
        self.aug = aug
        self.limit = limit
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed
        self.batch_rows_read = batch_rows_read

    def _iter_rows(self) -> Iterator[Dict[str, Any]]:
        dataset = ds.dataset(self.paths, format="parquet")

        scanner = dataset.scanner(
            columns=self.columns,
            filter=self.filter_expr,
            batch_size=self.batch_rows_read,
        )

        yielded = 0
        for record_batch in scanner.to_batches():
            # Convert each batch to columns once, then yield row dicts.
            cols = {name: record_batch.column(i) for i, name in enumerate(record_batch.schema.names)}
            n = record_batch.num_rows
            for idx in range(n):
                row = {k: cols[k][idx].as_py() for k in cols.keys()}

                if not self.pre_filter(row):
                    continue

                row = self.unpickle(row)

                if not self.post_filter(row):
                    continue

                row = self.map_func(row, aug=self.aug)

                yield row
                yielded += 1
                if self.limit is not None and yielded >= self.limit:
                    return

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Make per-worker RNG distinct but deterministic-ish.
        worker_info = torch.utils.data.get_worker_info()
        base_seed = self.shuffle_seed
        if worker_info is not None:
            base_seed = base_seed + worker_info.id * 1000003

        rng = random.Random(base_seed)

        if not self.shuffle_buffer_size or self.shuffle_buffer_size <= 0:
            yield from self._iter_rows()
            return

        # Approximate "local shuffle buffer"
        buf: List[Dict[str, Any]] = []
        for item in self._iter_rows():
            if len(buf) < self.shuffle_buffer_size:
                buf.append(item)
                continue

            j = rng.randrange(len(buf))
            yield buf[j]
            buf[j] = item

        # Drain buffer
        rng.shuffle(buf)
        yield from buf


class BaseDataModule(abc.ABC, LightningDataModule):
    def __init__(
        self,
        data_root: str,
        aug: bool = True,
        fast_dev_run: bool = False,
        min_face: int = 0,
        max_face: int = 30,
        max_edge: int = 20,          # kept for compatibility (unused here)
        bbox_scaled: int = 3,        # kept for compatibility (unused here)
        buffer_size: int = 100,
        batch_size: int = 512,
        drop_last: bool = False,
        prefetch_batches: int = 1,   # mapped to DataLoader prefetch_factor
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
        materialize: bool = False,   # streaming dataset => no materialize
        disable_progress_bars: bool = True,  # irrelevant without Ray
        scaled_unique: bool = True,
        num_workers: int = 0,        # add: standard torch DataLoader parallelism
        pin_memory: bool = True,     # add: common perf toggle
        persistent_workers: bool = False,  # add: common perf toggle
        rows_per_arrow_batch: int = 4096,  # add: scanner batch size
    ):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.fast_dev_run:
            self.hparams.prefetch_batches = 1

        print(f"using data from: {data_root}")
        print("max_face:", self.hparams.max_face)

        # holders created in setup()
        self._train_ds = None
        self._val_ds = None

    @property
    @abc.abstractmethod
    def columns(self) -> List[str]:
        ...

    @staticmethod
    @abc.abstractmethod
    def unpickle(row: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def post_filter(self, row: Dict[str, Any]) -> bool:
        return True

    def pre_filter(self, row: Dict[str, Any]) -> bool:
        return True

    @abc.abstractmethod
    def map_func(self, row: Dict[str, Any], aug: bool) -> Dict[str, Any]:
        ...

    def _filter_expr(self) -> pc.Expression:
        expr = (pc.field("num_faces_after_splitting") >= self.hparams.min_face) & (
            pc.field("num_faces_after_splitting") <= self.hparams.max_face
        )
        if self.hparams.scaled_unique:
            expr = expr & pc.field("scaled_unique")
        return expr

    def _dataset_columns(self) -> List[str]:
        extra = ["num_faces_after_splitting"]
        if self.hparams.scaled_unique:
            extra.append("scaled_unique")
        # ensure required filter columns are present for pushdown and (optional) for your filters
        return list(dict.fromkeys(self.columns + extra))

    def setup(self, stage: Optional[str] = None) -> None:
        # Build streaming datasets once.
        cols = self._dataset_columns()
        expr = self._filter_expr()

        if stage in (None, "fit"):
            self._train_ds = ParquetRowIterable(
                paths=f"{self.hparams.data_root}/train",
                columns=cols,
                filter_expr=expr,
                pre_filter=self.pre_filter,
                unpickle=self.unpickle,
                post_filter=self.post_filter,
                map_func=self.map_func,
                aug=self.hparams.aug,
                limit=self.hparams.limit_train,
                shuffle_buffer_size=self.hparams.buffer_size,
                shuffle_seed=9876,
                batch_rows_read=self.hparams.rows_per_arrow_batch,
            )

            self._val_ds = ParquetRowIterable(
                paths=f"{self.hparams.data_root}/val",
                columns=cols,
                filter_expr=expr,
                pre_filter=self.pre_filter,
                unpickle=self.unpickle,
                post_filter=self.post_filter,
                map_func=self.map_func,
                aug=False,  # no aug in val
                limit=self.hparams.limit_val,
                shuffle_buffer_size=None,
                shuffle_seed=9876,
                batch_rows_read=self.hparams.rows_per_arrow_batch,
            )

        if self.hparams.materialize:
            print("WARNING: materialize=True requested, but this implementation streams Parquet and does not materialize.")

    def _dl_kwargs(self, train: bool) -> Dict[str, Any]:
        # Map Ray prefetch_batches -> DataLoader prefetch_factor (only matters if num_workers>0)
        prefetch_factor = None
        if self.hparams.num_workers and self.hparams.num_workers > 0:
            prefetch_factor = max(1, int(self.hparams.prefetch_batches))

        return {
            "batch_size": self.hparams.batch_size,
            "drop_last": self.hparams.drop_last,
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
            "persistent_workers": bool(self.hparams.persistent_workers) and self.hparams.num_workers > 0,
            "prefetch_factor": prefetch_factor,
            "collate_fn": getattr(self, "collate_fn", None),
        }

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            self.setup("fit")
        return DataLoader(self._train_ds, **self._dl_kwargs(train=True))

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            self.setup("fit")
        return DataLoader(self._val_ds, **self._dl_kwargs(train=False))


class ARDataModule(BaseDataModule):
    def __init__(
        self,
        max_seq: int,
        bit: int,
        load_geom: bool = False,
        geom_tokenizer: str = "constraint",
        geom_ratio: float = 0.5,
        load_meta: bool = False,
        meta_ratio: float = 0.5,
        uv_invariant: bool = True,
        surf_codebook_size: int = 1000,
        edge_codebook_size: int = 1000,
        **kwargs,
    ):
        # Pass only the parent arguments to the Parent class
        super().__init__(**kwargs)
        self.FLAG_PAD = len(MMTokenIndex.__members__)
        self.ID_PAD = self.hparams.max_face
        self.POS_PAD = 2**self.hparams.bit

    @property
    def columns(self) -> List[str]:
        """
        Returns the list of column names.

        Returns:
            List[str]: List of column names.
        """
        cols = [
            "face_points_normalized",
            "edge_points_normalized",
            "face_bbox_world",
            "edge_bbox_world",
            "face_edge_incidence",
        ]
        if self.hparams.load_geom and self.hparams.geom_tokenizer != "random":
            cols += ["constraint_faces"]
        return cols

    def pre_filter(self, row: Dict[str, Any]) -> bool:
        """

        Filters the input row based on a set of conditions.
        Pre-Fitler check for following conditions:
            [1]: empty face edge incidence
            [2]: adjacency array of dimension 0
            [3]: face with no adjacency to an edge
            [4]: non-manifold solid
            [5]: tiny face
            [6]: tiny edge
            [7]: data beyond maximum sequence length
        Args:
            row (Dict[str, Any]): Input row data.

        Returns:
            bool: True if the row passes the filter, False otherwise.

        """
        TOL = 1 / (
            2 ** (self.hparams.bit - 1)
        )  # Our data is quantized to 10 bits, but the range is from [-1, 1].

        # check for empty face_edge_incidence
        face_edge_adj = deserialize_array(row["face_edge_incidence"])
        if len(face_edge_adj) == 0:
            return False

        # check for array of dimension 0
        if len(face_edge_adj.shape) == ():
            return False

        # Checks if there are any faces with no adjacency to an edge. If not, skips the data point.
        if np.any(np.all(np.logical_not(face_edge_adj), axis=1)):
            return False

        # Skip non-manifold shapes
        if np.any(np.sum(face_edge_adj.sum(0) != 2)):
            return False

        # check for max edges 
        if face_edge_adj.shape[1] > self.hparams.max_edge:
            return False

        # Filter tiny face (original and reconstructed)
        # must have at least one dimension larger than tol_diff
        face_pos = deserialize_array(row["face_bbox_world"])
        xyz_diff = np.abs(face_pos[:, 0:3] - face_pos[:, 3:6])
        if np.any(np.all(xyz_diff < TOL, axis=-1)):
            return False

        # Filter tiny edge (original and reconstructed)
        # must have at least one dimension larger than tol_diff
        edge_pos = deserialize_array(row["edge_bbox_world"])
        xyz_diff = np.abs(edge_pos[:, 0:3] - edge_pos[:, 3:6])
        if np.any(np.all(xyz_diff < TOL, axis=-1)):
            return False

        num_faces, num_edges = face_edge_adj.shape
        
        # # Check if sequence is too long with dummy tokens
        # total_seq_len = 2 * 4 + 1  # sequence s/e. cad s/e. meta s/e.  geom s/e
        # total_seq_len += 2*num_faces  # face start / end
        # total_seq_len += num_edges  # <edge id>
        # total_seq_len += num_faces * (6 + 4)  # <face pos, face geom>
        # total_seq_len += num_edges * (6 + 2)  # <edge pos, edge geom>
        # # level start / end, assuming won't be more than half the faces
        # total_seq_len += int(num_faces//2) 
        # if total_seq_len > self.hparams.max_seq - 1000:  # subtract 1000 for user input (sft)
        #     return False

        # Decrease ratio of easy samples
        if num_faces < 25 and random.random() < 0.9:
            return False

        return True

    @staticmethod
    def unpickle(row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unpickles the input row data.

        Args:
            row (Dict[str, Any]): Input row data after filtering.

        Returns:
            Dict[str, Any]: Unpickled row data.
        """
        data = {
            "face_points_normalized": deserialize_array(row["face_points_normalized"]),
            "edge_points_normalized": deserialize_array(row["edge_points_normalized"]),
            "face_edge_incidence": deserialize_array(row["face_edge_incidence"]).astype(
                np.float32
            ),
            "face_bbox_world": deserialize_array(row["face_bbox_world"]),
            "edge_bbox_world": deserialize_array(row["edge_bbox_world"]),
        }
        if "constraint_faces" in row:
            data["constraint_faces"] = deserialize_array(
                row["constraint_faces"]
            ).astype(np.float32)
        return data

    def argument_cad_data(self, face_pos, edge_pos, face_ncs, edge_ncs):
        # Get all eight corners
        face_pos_corners = utils.bbox_corners(face_pos)
        edge_pos_corners = utils.bbox_corners(edge_pos)
        # Random rotation
        axis = random.choice(["x", "y", "z"])
        angle = random.choice([90, 180, 270])
        face_pos_corners = utils.rotate_axis(face_pos_corners, angle, axis)
        edge_pos_corners = utils.rotate_axis(edge_pos_corners, angle, axis)
        face_ncs = utils.rotate_axis(face_ncs, angle, axis)
        edge_ncs = utils.rotate_axis(edge_ncs, angle, axis)
        # Re-compute the bottom left and top right corners
        face_pos = utils.get_bboxes(face_pos_corners)
        face_pos = face_pos.reshape(len(face_pos), 6)
        edge_pos = utils.get_bboxes(edge_pos_corners)
        edge_pos = edge_pos.reshape(len(edge_pos), 6)
        return face_pos, edge_pos, face_ncs, edge_ncs

    def convert2seq(
        self, face_pos_bit, edge_pos_bit, faces_sorted, face_edge_adj, levels
    ):
        # CAD tokenization
        levels_sum = np.cumsum(levels)
        data_seq = [MMTokenIndex.BOC.value]

        # First face
        data_seq += (
            [MMTokenIndex.BOL.value]  # start of level
            + [MMTokenIndex.BOF.value]  # start of face
            + [faces_sorted[0] + self.FLAG_PAD]  # face id
            + (
                face_pos_bit[faces_sorted[0]] + self.ID_PAD + self.FLAG_PAD
            ).tolist()  # face position
            + [
                faces_sorted[0] + self.POS_PAD + self.ID_PAD + self.FLAG_PAD
            ]  # face z index
            + [MMTokenIndex.EOF.value]  # end of face
        )

        # Rest of the faces
        for index in range(1, len(faces_sorted)):
            if index in levels_sum:
                data_seq += [
                    MMTokenIndex.EOL.value,
                    MMTokenIndex.BOL.value,
                ]  # end / start of line

            data_seq += (
                [MMTokenIndex.BOF.value]  # start of face
                + [faces_sorted[index] + self.FLAG_PAD]  # face id
                + (
                    face_pos_bit[faces_sorted[index]]
                    + self.ID_PAD
                    + self.FLAG_PAD  # face position
                ).tolist()  # face position
                + [
                    faces_sorted[index] + self.POS_PAD + self.ID_PAD + self.FLAG_PAD
                ]  # face z index
            )

            # Find all the edges connected to previously genearted faces
            for x in faces_sorted[:index]:
                connected_edges = np.where(
                    face_edge_adj[[faces_sorted[index], x]].sum(0) == 2
                )[0]

                if len(connected_edges) > 0:
                    # Sort by edge bbox
                    edge_xyz_order = np.lexsort(
                        (
                            edge_pos_bit[connected_edges][:, 5],
                            edge_pos_bit[connected_edges][:, 4],
                            edge_pos_bit[connected_edges][:, 3],
                            edge_pos_bit[connected_edges][:, 2],
                            edge_pos_bit[connected_edges][:, 1],
                            edge_pos_bit[connected_edges][:, 0],
                        )
                    ).tolist()
                    connected_edges = connected_edges[edge_xyz_order]

                    # Add all the edges
                    for edge_index in connected_edges:
                        data_seq += (
                            [x + self.FLAG_PAD]  # face id
                            + (
                                edge_pos_bit[edge_index]
                                + self.ID_PAD
                                + self.FLAG_PAD  # edge position
                            ).tolist()  # edge position
                            + [
                                edge_index
                                + self.POS_PAD
                                + 2 * self.ID_PAD
                                + self.FLAG_PAD
                            ]  # edge z index
                        )

            data_seq += [MMTokenIndex.EOF.value]  # end of face

        data_seq += [
            MMTokenIndex.EOL.value,
            MMTokenIndex.EOC.value,
        ]  # end of line & cad
        return data_seq

    def cad_tokenization(
        self, 
        face_pos, 
        edge_pos, 
        face_edge_adj, 
        face_graph, 
        seen_faces,
    ):
        """
        CAD tokenization
        """
        # Qunatize bbox positions
        face_pos_bit, edge_pos_bit = utils.quantize_pos(
            face_pos, edge_pos, self.hparams.bit
        )

        # Sort faces by xyz order
        xyz_order = np.lexsort(
            (
                face_pos_bit[:, 5],
                face_pos_bit[:, 4],
                face_pos_bit[:, 3],
                face_pos_bit[:, 2],
                face_pos_bit[:, 1],
                face_pos_bit[:, 0],
            )
        ).tolist()

        # Sort Faces by Ordered-BFS
        if len(seen_faces)>0:
            faces_sorted = copy.deepcopy(seen_faces)
        else:
            faces_sorted = [xyz_order[0]]
        levels = [len(faces_sorted)]

        for _ in range(self.hparams.max_face):
            # Unseen neighbor faces
            neighbors = []
            for face in faces_sorted:
                neighbors += [
                    x for x in face_graph.neighbors(face) if x not in faces_sorted
                ]
            if len(neighbors) == 0:
                break  # complete
            # Sort neighbors by xyz order
            neighbors = list(set(neighbors))  # remove duplicates
            neighbors_order = [xyz_order.index(x) for x in neighbors]
            neighbors_sorted = [neighbors[x] for x in np.argsort(neighbors_order)]
            # Add to existing faces
            faces_sorted += neighbors_sorted
            levels.append(len(neighbors_sorted))

        # Parse sequence
        data_seq = self.convert2seq(
            face_pos_bit, edge_pos_bit, faces_sorted, face_edge_adj, levels
        )

        # Remove user input 
        if len(seen_faces)>0:
            faces_sorted = faces_sorted[len(seen_faces):]
            user_end_idx = np.where(np.array(data_seq)==MMTokenIndex.EOL.value)[0][0]            
            data_seq = [MMTokenIndex.BOC.value] + list(data_seq[user_end_idx+1:])

        return data_seq, faces_sorted

    def geom_tokenization(
        self,
        face_pos,
        edge_pos,
        face_edge_adj,
        face_graph,
        constraint_mask,
    ):
        """
        Partial user geometry for autocomplete
        """
        # Select user face indices
        min_sample_face_count = 2 #minimum two faces 
        max_sample_face_count = min(int(len(face_pos) * 0.5), 20) # maximum cap at 50% or 20 faces
        num_user_input = random.randint(min_sample_face_count, max_sample_face_count)  
        random_face_indices = random.sample(range(len(face_pos)), num_user_input)

        if self.hparams.geom_tokenizer == "random":
            user_face_indices = random_face_indices
        elif self.hparams.geom_tokenizer == "constraint":
            if constraint_mask.sum() == 0:
                user_face_indices = random_face_indices
            else:
                user_face_indices = np.where(constraint_mask)[0]
        else:
            if constraint_mask.sum() == 0 or random.random() > 0.7: # 30% random - 70% constraint
                user_face_indices = random_face_indices
            else:
                user_face_indices = np.where(constraint_mask)[0]
                if len(user_face_indices)>20:
                    user_face_indices = np.random.choice(user_face_indices, size=20, replace=False)
            

        # Qunatize bbox positions
        face_pos_bit, edge_pos_bit = utils.quantize_pos(
            face_pos, edge_pos, self.hparams.bit
        )

        # Sort faces by xyz order
        user_face_pos_bit = face_pos_bit[user_face_indices]
        xyz_order = np.lexsort(
            (
                user_face_pos_bit[:, 5],
                user_face_pos_bit[:, 4],
                user_face_pos_bit[:, 3],
                user_face_pos_bit[:, 2],
                user_face_pos_bit[:, 1],
                user_face_pos_bit[:, 0],
            )
        ).tolist()
        faces_sorted = [user_face_indices[x] for x in xyz_order]

        data_seq = [
            MMTokenIndex.BOGEOM.value,
            MMTokenIndex.BOL.value,
        ]  # start of level & cad

        for index, face_idx in enumerate(faces_sorted):
            data_seq += (
                [MMTokenIndex.BOF.value]  # start of face
                + [face_idx + self.FLAG_PAD]  # face id
                + (
                    face_pos_bit[face_idx]
                    + self.ID_PAD
                    + self.FLAG_PAD  # face position
                ).tolist()  # face position
                + [
                    face_idx + self.POS_PAD + self.ID_PAD + self.FLAG_PAD
                ]  # face z index
            )

            # Find all the edges connected to previously genearted faces
            prev_face_edges = []
            prev_face_ids = []
            for x in faces_sorted[:index]:
                connected_edges = np.where(face_edge_adj[[face_idx, x]].sum(0) == 2)[0]
                if len(connected_edges) > 0:
                    prev_face_edges += list(connected_edges)
                    prev_face_ids += [x] * len(connected_edges)

            # Dangling edges
            co_edges = []
            edges_on_face = np.where(face_edge_adj[face_idx] == 1)[0]
            neighbor_faces = [
                x for x in face_graph.neighbors(face_idx) if x in faces_sorted
            ]
            if len(neighbor_faces) > 0:
                co_edges = edges_on_face[
                    np.where(
                        face_edge_adj[:, edges_on_face][np.array(neighbor_faces)].sum(0)
                        == 1
                    )[0]
                ]
            dangling_edges = list(set(edges_on_face) - set(co_edges))

            all_edges = dangling_edges + prev_face_edges
            all_face_ids = [face_idx] * len(dangling_edges) + prev_face_ids

            # Sort by edge bbox
            xyz_order = np.lexsort(
                (
                    edge_pos_bit[all_edges][:, 5],
                    edge_pos_bit[all_edges][:, 4],
                    edge_pos_bit[all_edges][:, 3],
                    edge_pos_bit[all_edges][:, 2],
                    edge_pos_bit[all_edges][:, 1],
                    edge_pos_bit[all_edges][:, 0],
                )
            ).tolist()
            all_edges = [all_edges[x] for x in xyz_order]
            all_face_ids = [all_face_ids[x] for x in xyz_order]

            # Replace dangling edges with dummyID
            all_face_ids = [
                MMTokenIndex.DUMMYID.value if (x==face_idx) 
                else (x+self.FLAG_PAD) 
                for x in all_face_ids
            ]

            # Add the edges
            for edge_index, connect_face_index in zip(all_edges, all_face_ids):
                data_seq += (
                    [connect_face_index]  # face id
                    + (
                        edge_pos_bit[edge_index]
                        + self.ID_PAD
                        + self.FLAG_PAD  # edge position
                    ).tolist()  # edge position
                    + [
                        edge_index + self.POS_PAD + 2 * self.ID_PAD + self.FLAG_PAD
                    ]  # edge z index
                )

            data_seq += [MMTokenIndex.EOF.value]  # end of face

        data_seq += [
            MMTokenIndex.EOL.value,
            MMTokenIndex.EOGEOM.value,
        ]  # end of level & geometry

        return data_seq, faces_sorted

    def map_func(self, row: Dict[str, Any], aug: bool) -> Dict[str, Any]:
        """
        Tokenized and augments the input row data to breformer format.

        Args:
            row (Dict[str, Any]): Input row data.
            aug (bool): Flag indicating whether to augment the data.

        Returns:
            Dict[str, Any]: Processed and tokenized row data.
        """
        # do aug here as we need to pad the data afterward whether
        # aug was done or not
        face_ncs = row["face_points_normalized"]
        edge_ncs = row["edge_points_normalized"]
        face_pos = row["face_bbox_world"]
        edge_pos = row["edge_bbox_world"]
        face_edge_adj = row["face_edge_incidence"].astype(bool)
        constraint_mask = None
        if "constraint_faces" in row:
            constraint_mask = row["constraint_faces"].astype(bool)

        # Build face graph for fast finding neighbors
        face_graph = nx.Graph()
        face_graph.add_nodes_from(np.arange(len(face_ncs)))
        for col in face_edge_adj.T:
            face_graph.add_edge(np.where(col)[0][0], np.where(col)[0][1])

        # Augment CAD data
        if aug and random.random() > 0.9:
            face_pos, edge_pos, face_ncs, edge_ncs = self.argument_cad_data(
                face_pos, edge_pos, face_ncs, edge_ncs
            )

        # Apply uv-invariant (must be after argumentation)
        if self.hparams.uv_invariant:
            face_ncs = utils.sort_uv_grids(face_ncs)
            edge_ncs = utils.sort_u_grids(edge_ncs)

        # Geometry Tokenization
        geom_tokens, geom_face_indices = [], []
        if self.hparams.load_geom and random.random() > (1 - self.hparams.geom_ratio):

            # Randomly select scale (within 15% of the original size)
            threshold = 0.15
            face_pos = face_pos.reshape(len(face_pos), 2, 3)
            edge_pos = edge_pos.reshape(len(edge_pos), 2, 3)
            random_s = utils.find_random_bbox_scale(face_pos, threshold)
            user_faces_s = utils.rescale_bbox(face_pos, random_s)
            # Randomly select xyz translation (within 0.1 of the original location)
            random_t = utils.find_random_bbox_translations(user_faces_s, threshold)
            # Apply rotation and translation to bbox
            face_pos = utils.rescale_bbox(face_pos, random_s)
            edge_pos = utils.rescale_bbox(edge_pos, random_s)
            face_pos = utils.translate_bbox(face_pos, random_t)
            edge_pos = utils.translate_bbox(edge_pos, random_t)
            face_pos = face_pos.reshape(len(face_pos), 6)
            edge_pos = edge_pos.reshape(len(edge_pos), 6)

            geom_tokens, geom_face_indices = self.geom_tokenization(
                face_pos,
                edge_pos,
                face_edge_adj,
                face_graph,
                constraint_mask,
            )

        # Meta data (complexity)
        meta_tokens = []
        if self.hparams.load_meta:
            meta_tokens = [MMTokenIndex.BOM.value]
            if random.random() > (1 - self.hparams.meta_ratio):
                if len(face_pos) < 25:
                    meta_tokens += [MMTokenIndex.GEN_EASY.value]
                elif len(face_pos) < 50:
                    meta_tokens += [MMTokenIndex.GEN_MID.value]
                else:
                    meta_tokens += [MMTokenIndex.GEN_HARD.value]
            else:
                meta_tokens += [MMTokenIndex.GEN_UNCOND.value]
            meta_tokens += [MMTokenIndex.EOM.value]

        # CAD Tokenization
        cad_tokens, cad_face_indices = self.cad_tokenization(
            face_pos, edge_pos, face_edge_adj, face_graph, geom_face_indices
        )

        # TODO add point cloud condition
        full_seq = (
            [MMTokenIndex.BOS.value]
            + meta_tokens
            # + geom_tokens
            + cad_tokens
            + [MMTokenIndex.EOS.value]
        )
        
        #  Reorder Face ID
        cad_face_indices_unseen = [x for x in cad_face_indices if x not in geom_face_indices]
        face_indices = geom_face_indices + cad_face_indices_unseen
        remap = {x+self.FLAG_PAD: y+self.FLAG_PAD for x, y in zip(face_indices, np.arange(len(face_indices)))}
        full_seq = np.array([remap[x] if x in remap.keys() else x for x in full_seq])

        ######################################################
        # [OPTIONAL] Convert global edge ID to local edge ID #
        ######################################################
        level_idx = np.where(full_seq==MMTokenIndex.BOL.value)[0]
        level_tokens = np.split(full_seq, level_idx)
        updated_tokens = []
        for i, (pre_level, cur_level) in enumerate(zip(level_tokens[:-1], level_tokens[1:])):
            cur_face_ids = cur_level[(cur_level >= self.FLAG_PAD) & (cur_level < self.FLAG_PAD+self.ID_PAD)]
            prev_face_ids = pre_level[(pre_level >= self.FLAG_PAD) & (pre_level < self.FLAG_PAD+self.ID_PAD)]
            # Remove edge id from prev face
            if i >= 1:
                ppre_level = level_tokens[i-1]
                pprev_face_ids = ppre_level[(ppre_level >= self.FLAG_PAD) & (ppre_level < self.FLAG_PAD+self.ID_PAD)]
                prev_face_ids = np.array([x for x in prev_face_ids if x not in pprev_face_ids])
            # Re-assign edge ids
            level_faces = np.sort(
                np.unique(
                    np.concatenate((prev_face_ids,cur_face_ids))
                )
            )
            remap = {
                x : y + self.FLAG_PAD
                for x, y in zip(level_faces, np.arange(len(level_faces)))
            }
            cur_level = np.array([remap[x] if x in remap.keys() else x for x in cur_level])
            # Remove face ID (set to -1)
            cur_level[np.where(cur_level == MMTokenIndex.BOF.value)[0]+1] = -1
            updated_tokens += list(cur_level[cur_level>0])
        # add back meta tokens
        updated_tokens = list(level_tokens[0]) + updated_tokens
        full_seq = np.array(updated_tokens)

        # Randomly sample context window
        if len(full_seq) > self.hparams.max_seq:
            start_idx = np.random.randint(0, len(full_seq) - self.hparams.max_seq + 1)
            full_seq = full_seq[start_idx:start_idx + self.hparams.max_seq]

        # Add padding
        seq_tokens = utils.pad_neg(full_seq, self.hparams.max_seq)
        face_ncs = utils.pad_zero(face_ncs, self.hparams.max_face)
        edge_ncs = utils.pad_zero(edge_ncs, self.hparams.max_edge)

        output = {
            "seq": seq_tokens,
            "face_ncs": face_ncs,
            "edge_ncs": edge_ncs,
        }
        return output

    @property
    def dtypes(self):
        output = {
            "seq": torch.int64,
            "face_ncs": torch.float32,
            "edge_ncs": torch.float32,
        }
        return output
