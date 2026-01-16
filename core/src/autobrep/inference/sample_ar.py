import json
from pathlib import Path
from typing import List, Optional

import torch
from jsonargparse import CLI
from pytorch_lightning import seed_everything

from autobrep.inference.inference_common import (
    reconstruct_compound,
    save_debug_images,
    save_point_grid,
)
from autobrep.models.autoregressive import (
    ARGenCheckpointPaths,
    AutoRegressiveSampler,
)
from autobrep.utils import DotDict, generate_random_string, timer
from autobrep.inference.brepgen_brep_builder import AutoBrepBuilder
from occwl.io import save_step as save_step_func


def sample_batch(
    config: dict,
    model: AutoRegressiveSampler,
    device: torch.device,
    debug_dir: Optional[str] = None,
):
    """
    Autoregressively sample a batch of CAD solids

    Args:
        config: configuration dictionary
        model: model dictionary with brepformer, surface and edge fsq vaes
        device: default to use gpu
        debug_dir: debug directory to save debug images
    """
    if config.debug_mode:
        assert debug_dir is not None, "Debug directory required if debug_mode is set."
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(exist_ok=True, parents=True)

    stem = generate_random_string(20)

    log = {
        "status": "TBD",
        "seed": config.seed.seed_value,
        "errors": [],
        "valid": 0,
        "invalid": 0,
        "stem": stem,
    }
    builders = []

    # Unconditional generation.
    builders = [
        AutoBrepBuilder(
            device=device, 
            z_threshold=config.hyper_parameters.z_threshold, 
            vertex_threshold=config.hyper_parameters.vertex_threshold, 
            sewing_tolerance=config.hyper_parameters.sewing_tolerance
        )
    ]

    import time 
    start_time = time.time()

    with timer(
        "Total Time to generate "
        + str(config.hyper_parameters.num_samples_to_generate)
        + " B-Reps: %s seconds",
    ):
        samples = model.sample_tokens(
            config=config,
            batch_size=config.hyper_parameters.batch_size,
        )

        # Decode B-rep tokens
        batch_decoded = model.decode_tokens(samples.detach().cpu().numpy())

        # Convert to cad data format
        batch_cad_data = model.convert_to_cad_data(batch_decoded)

        # Post process and rebuild to STEP file
        with timer("Time to rebuild: %s seconds"):
            for sample_idx, cad_data in enumerate(batch_cad_data):
                sample_stem = f"{stem}_{str(sample_idx).zfill(3)}"

                if config.debug_mode:
                    # Save point grid data
                    save_point_grid(
                        debug_dir / f"{sample_stem}_before_joint_optimize.npz", cad_data
                    )

                    # Save debug images
                    save_debug_images(
                        cad_data,
                        debug_dir / f"{sample_stem}_face.png",
                        debug_dir / f"{sample_stem}_edge.png",
                    )

                # Rebuild
                result = reconstruct_compound(cad_data, builders)

                if result is not None:
                    save_step_func(
                        [result], (debug_dir / sample_stem).with_suffix(".step")
                    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    return


def main(
    input_json: str,
    output_dir: str,
    checkpoint_names: ARGenCheckpointPaths = ARGenCheckpointPaths(),
    debug_dir: Optional[str] = None,
):
    """
    Run the diffusion model to build a solid.

    Save the resulting solid to output
    """
    # Load JSON config
    with open(input_json, "r") as file:
        config = json.load(file)
    config = DotDict(config)

    checkpoints = ARGenCheckpointPaths.from_folder(
        folder=config.weight_folder, checkpoints=checkpoint_names
    )

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    if debug_dir is not None:
        Path(debug_dir).mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained weights
    model = AutoRegressiveSampler(
        checkpoint_paths=checkpoints,
        device=device,
    )

    # Seed everything
    if config.seed.use_seed:
        seed_everything(config.seed.seed_value, workers=True)

    # Run unconditional generation
    for i in range(config.hyper_parameters.num_batches_to_sample):
        print(f"------------ {i} ------------")
        sample_batch(
            config,
            model,
            device,
            debug_dir=debug_dir,
        )


if __name__ == "__main__":
    CLI(main, as_positional=False)
