"""
Wrapper class for the Segment Anything Model (SAM) predictor.
Handles model loading, setting images, and running predictions.
"""

import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from .utils.download import download_with_progress

log = logging.getLogger(__name__)

# Model URLs
SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

class SamModelWrapper:
    """Encapsulates SAM model loading and prediction."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the wrapper, loads the model checkpoint, and creates the predictor.

        Args:
            config (Dict[str, Any]): The application configuration dictionary.
        """
        self.config = config
        self.model_type = config['sam_model']['model_type']
        self.checkpoint_dir = Path(config['paths']['sam_checkpoint_dir'])
        self.device = torch.device(config['sam_model']['device'])
        self.predictor: Optional[SamPredictor] = None
        self._load_model()

    def _load_model(self):
        """Downloads checkpoint if needed and loads the SAM model."""
        checkpoint_url = SAM_CHECKPOINT_URLS.get(self.model_type)
        if not checkpoint_url:
            log.error(f"Invalid SAM model type specified: {self.model_type}")
            return # Predictor remains None

        checkpoint_filename = Path(os.path.basename(checkpoint_url))
        checkpoint_path = self.checkpoint_dir / checkpoint_filename

        log.info(f"Attempting to load SAM model '{self.model_type}' on device '{self.device}'")
        effective_checkpoint_path = download_with_progress(checkpoint_url, checkpoint_path)
        if not effective_checkpoint_path or not effective_checkpoint_path.exists():
             log.error("Failed to obtain SAM model checkpoint.")
             return

        log.info(f"Loading SAM model from {effective_checkpoint_path}...")
        try:
            sam_model = sam_model_registry[self.model_type](checkpoint=str(effective_checkpoint_path))
            sam_model.to(device=self.device)
            sam_model.eval()
            self.predictor = SamPredictor(sam_model)
            log.info("SAM model loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load SAM model: {e}", exc_info=True)
            if "CUDA out of memory" in str(e):
                log.critical("CUDA out of memory. Try smaller model or 'cpu' device in config.")
            self.predictor = None

    def set_image(self, image_rgb: np.ndarray) -> bool:
        """
        Sets the image for the SAM predictor. Handles potential memory errors.

        Args:
            image_rgb (np.ndarray): The image in RGB format.

        Returns:
            bool: True if successful, False otherwise (e.g., OOM error).
        """
        if not self.predictor:
            log.error("Cannot set image: SAM predictor not initialized.")
            return False
        if image_rgb is None:
             log.error("Cannot set image: Provided image is None.")
             return False

        log.info("Setting image in SAM predictor...")
        start_time = time.time()
        try:
            self.predictor.set_image(image_rgb)
            log.info(f"Image set successfully (took {time.time() - start_time:.2f} seconds).")
            return True
        except torch.cuda.OutOfMemoryError as e:
            log.error(f"CUDA out of memory while setting image: {e}")
            log.critical("GPU ran out of memory setting image. Try smaller model or 'cpu' device.")
            return False
        except Exception as e:
            log.error(f"Unexpected error setting image in predictor: {e}", exc_info=True)
            return False

    def predict(self, points: List[Tuple[int, int]], labels: List[int]) -> Optional[np.ndarray]:
        """
        Runs SAM prediction with the given points and labels.

        Args:
            points (List[Tuple[int, int]]): List of (x, y) point coordinates.
            labels (List[int]): List of corresponding labels (1 for foreground, 0 for background).

        Returns:
            Optional[np.ndarray]: The predicted mask (boolean array), or None if prediction fails.
        """
        if not self.predictor:
            log.error("Cannot predict: SAM predictor not initialized.")
            return None
        if not points:
            log.debug("Prediction skipped: No points provided.")
            return None # Return None if no points, so caller knows no mask exists

        log.info(f"Predicting mask with {len(points)} points...")
        start_time = time.time()
        input_point = np.array(points)
        input_label = np.array(labels)

        try:
            # Predict - using multimask_output=False for simplicity
            masks, scores, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            # masks[0] is highest score mask when multimask_output=False
            log.info(f"Prediction complete. Score: {scores[0]:.3f} (took {time.time() - start_time:.2f} seconds)")
            return masks[0] # Return boolean mask
        except torch.cuda.OutOfMemoryError as e:
            log.error(f"CUDA out of memory during prediction: {e}")
            log.critical("GPU ran out of memory during prediction. Try smaller model or 'cpu' device.")
            return None
        except Exception as e:
            log.error(f"Unexpected error during SAM prediction: {e}", exc_info=True)
            return None

