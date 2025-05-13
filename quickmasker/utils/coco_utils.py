"""
Utils for working with COCO annotation formats
"""
import numpy as np
from pycocotools import mask as mask_util
import logging
from typing import List, Dict, Any

log = logging.getLogger(__name__)

def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """
    Converts a binary mask (NumPy array) to COCO Run-Length Encoding (RLE) format.

    Args:
        binary_mask (np.ndarray): A 2D NumPy array where non-zero values indicate the mask.

    Returns:
        Dict[str, Any]: A dictionary containing 'counts' and 'size' in RLE format.
    """
    try:
        # Ensure mask is Fortran-contiguous, required by pycocotools
        fortran_binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
        # Encode to RLE
        encoded_mask = mask_util.encode(fortran_binary_mask)
        # Ensure 'counts' is utf-8 decoded if it's bytes
        if isinstance(encoded_mask['counts'], bytes):
            encoded_mask['counts'] = encoded_mask['counts'].decode('utf-8')
        return encoded_mask
    except Exception as e:
        log.error(f"Error converting mask to RLE: {e}", exc_info=True)
        raise # Re-raise after logging

def get_bbox_from_mask(binary_mask: np.ndarray) -> List[int]:
    """
    Calculates the bounding box [x_min, y_min, width, height] from a binary mask.

    Args:
        binary_mask (np.ndarray): A 2D NumPy array representing the mask.

    Returns:
        List[int]: The bounding box coordinates [x, y, w, h]. Returns [0, 0, 0, 0] for empty masks.
    """
    if binary_mask.sum() == 0:
        return [0, 0, 0, 0] # Handle empty masks

    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
         return [0, 0, 0, 0]

    try:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        width = int(xmax - xmin + 1)
        height = int(ymax - ymin + 1)

        return [int(xmin), int(ymin), width, height]
    except Exception as e:
         log.error(f"Error calculating bounding box from mask: {e}", exc_info=True)
         return [0, 0, 0, 0] # Return empty box on error

