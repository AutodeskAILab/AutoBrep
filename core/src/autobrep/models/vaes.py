
from typing import List, Literal, Optional, Union

import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_dc import (
    Decoder as DCDecoder,
)
from diffusers.models.autoencoders.autoencoder_dc import (
    Encoder as DCEncoder,
)
from diffusers.models.autoencoders.vae import (
    Decoder,
    DiagonalGaussianDistribution,
    Encoder,
)
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from autobrep.models.fsq import FSQ
from autobrep.network import (
    AutoencoderKL1D,
    DCDecoder1D,
    DCEncoder1D,
    Decoder1D,
    DecoderOutput,
    Encoder1D,
)


def multiplyList(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result


class SurfaceFSQVAE(LightningModule):
    def __init__(
        self,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        num_down_blocks: int = 4,
        num_up_blocks: int = 4,
        block_out_channels: Union[str, List[int]] = "128,256,512,512",
        layers_per_block: int = 2,
        latent_channels: int = 3,
        sync_dist_train: bool = True,
        fsq_levels: list = [8, 5, 5, 5],
        z_dim: int = 4,
        max_face: int = 0,  # Saving as a hyperparameter and copying over from datamodule args
        max_edge: int = 0,  # Saving as a hyperparameter and copying over from datamodule args
        use_dcae: bool = False,  # use deepcompression autorencoder
    ):
        """
        Initialize the SurfaceFSQVAE model.

        Parameters
        ----------
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay in the AdamW optimizer.
        num_down_blocks : int
            Number of down blocks in the encoder that controls the latent grid spatial dimensions.
        num_up_blocks : int
            Number of up blocks in the decoder that controls the output grid spatial dimensions.
        block_out_channels : Union[str, List[int]]
            Number of output channels for each block in the encoder and decoder.
        layers_per_block : int
            Number of layers in each block.
        latent_channels : int
            Number of latent grid channels.
        sync_dist_train : bool
            Whether to synchronize distributed training.
        fsq_levels : list
            List of levels for the FSQ quantizer. The codebook size is the product of this.
        """
        super().__init__()
        self.save_hyperparameters()
        if isinstance(block_out_channels, str):
            self.hparams.block_out_channels = list(
                map(int, block_out_channels.split(","))
            )

        in_channels = 3  # We always use the point coordinate grids

        if use_dcae:
            self.encoder = DCEncoder(
                in_channels=in_channels,
                latent_channels=z_dim,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=[layers_per_block] * num_up_blocks,
            )
        else:
            self.encoder = Encoder(
                in_channels=in_channels,
                out_channels=z_dim,
                down_block_types=["DownEncoderBlock2D"] * num_down_blocks,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=layers_per_block,
                act_fn="silu",
                norm_num_groups=32,
                double_z=False,
            )

        # pass init params to Decoder
        out_channels = 3  # Points are always predicted

        if use_dcae:
            self.decoder = DCDecoder(
                in_channels=out_channels,
                latent_channels=z_dim,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=[layers_per_block] * num_up_blocks,
                act_fn="silu",
            )
        else:
            self.decoder = Decoder(
                in_channels=z_dim,
                out_channels=out_channels,
                up_block_types=["UpDecoderBlock2D"] * num_up_blocks,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=layers_per_block,
                norm_num_groups=32,
                act_fn="silu",
            )

        self.quantizer = FSQ(dim=z_dim, levels=fsq_levels)

        self.codebook_size = multiplyList(fsq_levels)

        self.downsample = torch.nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1
        )
        self.upsample = torch.nn.Linear(16, 64)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def prepare_input(self, batch):
        features = [batch["face_points_normalized"]]
        input_grid = torch.cat(features, dim=-1).permute(0, 3, 1, 2)
        return input_grid

    def prepare_output(self, decoded_output):
        output = {
            "face_points_normalized": decoded_output[:, :3],
        }
        return output

    def common_step(self, batch):
        face_uv = self.prepare_input(batch)
        # Encode
        logits = self.encoder(face_uv)

        # further shrink it down
        logits_downsample = self.downsample(logits)

        # FSQ
        quant, id = self.quantizer(logits_downsample)

        quant_upsample = self.upsample(quant.reshape(len(quant), -1)).reshape(
            len(quant), 4, 4, 4
        )

        # Decode
        dec = self.decoder(quant_upsample)
        output = {
            "face_uv": face_uv,
            "z": quant,
            "dec": dec,
            "id": id,
        }
        return output

    def common_step_and_loss(self, batch, stage: str):
        assert stage in ("train", "val", "test")
        output = self.common_step(batch)
        dec = output["dec"]
        dec = self.prepare_output(dec)

        points_mse_loss = torch.nn.functional.mse_loss(
            dec["face_points_normalized"],
            batch["face_points_normalized"].permute(0, 3, 1, 2),
        )
        total_loss = points_mse_loss

        loss = {
            f"{stage}/points_mse_loss": points_mse_loss,
        }

        loss[f"{stage}/loss"] = total_loss
        self.log_dict(
            loss,
            on_step=stage == "train",  # Log per-step for training
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.hparams.sync_dist_train,
        )
        return total_loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.common_step_and_loss(batch, "train")

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        output = self.common_step(batch)
        dec = output["dec"]
        dec = self.prepare_output(dec)

        points_mse_loss = torch.nn.functional.mse_loss(
            dec["face_points_normalized"],
            batch["face_points_normalized"].permute(0, 3, 1, 2),
        )
        total_loss = points_mse_loss

        loss = {
            "val/points_mse_loss": points_mse_loss,
            "val/loss": total_loss,
        }

        self.log_dict(
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.hparams.sync_dist_train,
        )
        return total_loss

    def drop_decoder(self):
        self.decoder = None
        return self

    def drop_encoder(self):
        self.encoder = None
        return self

    def encode(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            x (`torch.FloatTensor`): Input sample.
            posterior_sample_generator (`torch.Generator`, *optional*, defaults to None):
                The generator to use to sample the posterior. Returns the mode if undefined.
        """
        logits = self.encoder(x)
        logits_downsample = self.downsample(logits)
        quant, id = self.quantizer(logits_downsample)
        return quant, id

    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            z (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        z_upsample = self.upsample(z.reshape(len(z), -1)).reshape(len(z), 4, 4, 4)
        dec = self.decoder(z_upsample)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)


class EdgeFSQVAE(LightningModule):
    def __init__(
        self,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        num_down_blocks: int = 3,
        num_up_blocks: int = 3,
        block_out_channels: Union[str, List[int]] = "128,256,512",
        layers_per_block: int = 2,
        latent_channels: int = 3,
        sync_dist_train: bool = True,
        fsq_levels: list = [8, 5, 5, 5],
        z_dim: int = 4,
        max_face: int = 0,  # Saving as a hyperparameter and copying over from datamodule args
        max_edge: int = 0,  # Saving as a hyperparameter and copying over from datamodule args
        use_dcae: bool = False,  # use deepcompression autoencoder
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if isinstance(block_out_channels, str):
            self.hparams.block_out_channels = list(
                map(int, block_out_channels.split(","))
            )

        in_channels = 3  # TODO always use the point coordinate for edges
        if use_dcae:
            self.encoder = DCEncoder1D(
                in_channels=in_channels,
                latent_channels=z_dim,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=[layers_per_block] * num_down_blocks,
            )
        else:
            self.encoder = Encoder1D(
                in_channels=in_channels,
                out_channels=z_dim,
                down_block_types=["DownBlock1D"] * num_down_blocks,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=layers_per_block,
                act_fn="silu",
                norm_num_groups=32,
                double_z=False,
            )

        out_channels = 3  # TODO always use the point coordinate for edges
        if use_dcae:
            self.decoder = DCDecoder1D(
                in_channels=out_channels,
                latent_channels=z_dim,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=[layers_per_block] * num_up_blocks,
                act_fn="silu",
            )
        else:
            self.decoder = Decoder1D(
                in_channels=z_dim,
                out_channels=out_channels,
                up_block_types=["UpBlock1D"] * num_up_blocks,
                block_out_channels=self.hparams.block_out_channels,
                layers_per_block=layers_per_block,
                act_fn="silu",
                norm_num_groups=32,
            )

        self.quantizer = FSQ(dim=z_dim, levels=fsq_levels)

        self.codebook_size = multiplyList(fsq_levels)
        self.downsample = torch.nn.Conv1d(
            in_channels=z_dim, out_channels=z_dim, kernel_size=3, stride=2, padding=1
        )
        self.upsample = torch.nn.Linear(2 * z_dim, 4 * z_dim)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def prepare_input(self, batch):
        features = [batch["edge_points_normalized"]]
        input_grid = torch.cat(features, dim=-1).permute(0, 2, 1)
        return input_grid

    def prepare_output(self, decoded_output):
        output = {
            "edge_points_normalized": decoded_output[:, :3],
        }
        return output

    def common_step(self, batch):
        edge_u = self.prepare_input(batch)
        # Encode
        z = self.encoder(edge_u)

        z_downsample = self.downsample(z)

        # FSQ
        quant, id = self.quantizer(z_downsample.permute(0, 2, 1))
        quant = quant.permute(0, 2, 1)

        quant_upsample = self.upsample(quant.reshape(len(quant), -1)).reshape(
            len(quant), self.hparams.z_dim, 4
        )

        # Decode
        dec = self.decoder(quant_upsample)

        return {
            "edge_u": edge_u,
            "z": quant,
            "dec": dec,
        }

    def common_step_and_loss(self, batch, stage: str):
        assert stage in ("train", "val", "test")
        output = self.common_step(batch)
        dec = self.prepare_output(output["dec"])

        # Point grid loss: first 3 channels are always point grids
        points_mse_loss = torch.nn.functional.mse_loss(
            dec["edge_points_normalized"],
            batch["edge_points_normalized"].permute(0, 2, 1),
        )
        total_loss = points_mse_loss

        loss = {
            f"{stage}/points_mse_loss": points_mse_loss,
        }

        loss[f"{stage}/loss"] = total_loss.item()
        self.log_dict(
            loss,
            on_step=stage == "train",  # Log per-step for training
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.hparams.sync_dist_train,
        )
        return total_loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.common_step_and_loss(batch, "train")

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.common_step_and_loss(batch, "val")

    def drop_decoder(self):
        self.decoder = None
        return self

    def drop_encoder(self):
        self.encoder = None
        return self

    def encode(
        self,
        edge_u: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            x (`torch.FloatTensor`): Input sample.
            posterior_sample_generator (`torch.Generator`, *optional*, defaults to None):
                The generator to use to sample the posterior. Returns the mode if undefined.
        """
        # Encode
        z = self.encoder(edge_u)
        z_downsample = self.downsample(z)

        # FSQ
        quant, id = self.quantizer(z_downsample.permute(0, 2, 1))
        quant = quant.permute(0, 2, 1)
        return quant, id

    def decode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            z (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        quant_upsample = self.upsample(x.reshape(len(x), -1)).reshape(
            len(x), self.hparams.z_dim, 4
        )
        dec = self.decoder(quant_upsample)
        return DecoderOutput(sample=dec)