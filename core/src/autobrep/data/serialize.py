import io
import json
import random

import numpy as np


def serialize_array(array: np.ndarray) -> bytes:
    """
    Serialize a numpy array to a byte string.
    Taken from: https://stackoverflow.com/a/30699208
    """
    memfile = io.BytesIO()
    np.save(memfile, array)
    serialized = memfile.getvalue()
    return serialized


def deserialize_array(serialized: str) -> np.ndarray:
    """
    Deserialize a byte string to a numpy array.
    Taken from: https://stackoverflow.com/a/30699208
    """
    memfile = io.BytesIO()
    memfile.write(serialized)
    memfile.seek(0)
    return np.load(memfile)