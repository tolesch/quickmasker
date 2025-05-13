"""
Main controller class for the QuickMasker application.
Orchestrates the interaction between UI, SAM model, state management,
and handles the main application loop.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import logging
import time
from typing import List, Tuple, Dict, Any, Optional

from .config import load_and_validate_config
from .sam_model import SamModelWrapper
from .state_manager import StateManager
from .ui import OpenCVUI
from .event_handler import handle_mouse_event, handle_key_press
from .utils.coco_utils import mask_to_rle, get_bbox_from_mask

log = logging.getLogger(__name__)

class AnnotationController:
    """
    Orchestrates the annotation workflow, connecting UI, model, and state.
    """
    def __init__(self, config: Dict[str, Any]):
        log.info("Initializing AnnotationController...")
        self.config = config
        self.should_quit = False
        self.awaiting_delete_confirmation = False

        try:
            self.state_manager = StateManager(config)
            self.sam_model = SamModelWrapper(config)
            self.ui = OpenCVUI(config['ui'])
        except Exception as e:
            log.exception("Failed to initialize core components.")
            raise

        self.image_files: List[Path] = []
        self.current_image_rgb: Optional[np.ndarray] = None
        self.current_image_bgr: Optional[np.ndarray] = None
        self.points: List[Tuple[int, int]] = []
        self.point_labels: List[int] = []
        self.current_mask: Optional[np.ndarray] = None
        self.force_redraw: bool = True

        self._load_initial_data()

    def _load_initial_data(self):
        log.info("Loading initial application data...")
        self._get_image_files()
        self.state_manager.load_state()
        self.state_manager.sync_coco_images(self.image_files)
        if not (0 <= self.state_manager.get_current_index() < len(self.image_files)) and self.image_files:
             self.state_manager.set_current_index(0)
        self._load_image_by_index(self.state_manager.get_current_index())

    def _get_image_files(self):
        image_folder = Path(self.config['paths']['image_folder'])
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        log.info(f"Scanning for images in: {image_folder}")
        self.image_files = sorted([
            p for p in image_folder.glob('*')
            if p.is_file() and p.suffix.lower() in supported_extensions
        ])
        log.info(f"Found {len(self.image_files)} images.")

    def _load_image_by_index(self, index: int):
        self._reset_points_and_mask()
        self.current_image_rgb = None
        self.current_image_bgr = None

        if not self.image_files:
            log.warning("No images to load.")
            self.force_redraw = True
            return

        # Clamp index to valid range
        if index >= len(self.image_files): index = len(self.image_files) - 1
        if index < 0: index = 0
        if not self.image_files:
            self.force_redraw = True
            return


        self.state_manager.set_current_index(index)
        img_path = self.image_files[index]
        log.info(f"\nLoading image {index + 1}/{len(self.image_files)}: {img_path.name}")

        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: raise IOError(f"OpenCV failed to load image: {img_path}")
            self.current_image_bgr = img_bgr
            self.current_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.state_manager.update_image_dimensions(
                img_path.name, self.current_image_rgb.shape[0], self.current_image_rgb.shape[1]
            )
            if not self.sam_model.set_image(self.current_image_rgb):
                 self.current_image_rgb = None; self.current_image_bgr = None
        except Exception as e:
            log.error(f"Failed loading/processing image {img_path}: {e}", exc_info=True)
            self.current_image_rgb = None; self.current_image_bgr = None
            if isinstance(e, torch.cuda.OutOfMemoryError):
                 log.critical("GPU OOM loading image. Try smaller model or 'cpu'.")
        self.force_redraw = True

    def _reset_points_and_mask(self):
        self.points = []
        self.point_labels = []
        self.current_mask = None
        self.force_redraw = True

    def _handle_mouse_event(self, event: int, x: int, y: int, flags: int, param: Any):
        if self.current_image_rgb is None or self.awaiting_delete_confirmation: return
        point_added = handle_mouse_event(event, x, y, flags, self.points, self.point_labels)
        if point_added:
            self._run_prediction()

    def _run_prediction(self):
        if self.current_image_rgb is None or not self.points:
             self.current_mask = None; self.force_redraw = True; return
        predicted_mask = self.sam_model.predict(self.points, self.point_labels)
        self.current_mask = predicted_mask
        self.force_redraw = True

    def _save_current_annotation(self):
        if self.current_mask is None or not self.image_files: return
        current_filename = self.image_files[self.state_manager.get_current_index()].name
        self.state_manager.add_annotation(current_filename, self.current_mask)
        # Optionally, save state immediately after adding an annotation
        # self.state_manager.save_state()

    def _navigate(self, direction: int):
        if not self.image_files: return
        log.debug("Saving state before navigation...")
        self.state_manager.save_state() # Save before changing index

        current_index = self.state_manager.get_current_index()
        new_index = current_index + direction
        # Clamping is handled by _load_image_by_index if list is not empty
        self._load_image_by_index(new_index)

    def _initiate_delete_current_image(self):
        """Sets state to await delete confirmation."""
        if self.current_image_rgb is None or not self.image_files:
            log.warning("Delete initiated, but no image loaded or image list empty.")
            return
        self.awaiting_delete_confirmation = True
        self.force_redraw = True
        log.info(f"Awaiting confirmation to delete image: {self.image_files[self.state_manager.get_current_index()].name}")

    def _delete_current_image_confirmed(self):
        """Performs the actual deletion after user confirmation."""
        if not self.image_files or not (0 <= self.state_manager.get_current_index() < len(self.image_files)):
            log.error("Cannot delete: Invalid image index or empty image list.")
            self.awaiting_delete_confirmation = False
            self.force_redraw = True
            return

        current_idx = self.state_manager.get_current_index()
        img_path_to_delete = self.image_files[current_idx]
        filename_to_delete = img_path_to_delete.name
        coco_image_id_to_delete = self.state_manager.filename_to_coco_id.get(filename_to_delete)

        log.warning(f"DELETING image file: {img_path_to_delete}")
        try:
            img_path_to_delete.unlink(missing_ok=True) # Delete from filesystem
            log.info(f"Successfully deleted file: {img_path_to_delete}")
        except OSError as e:
            log.error(f"Error deleting file {img_path_to_delete} from filesystem: {e}")
        
        # Remove from internal image list
        self.image_files.pop(current_idx)

        # Update COCO data: Remove image entry
        if coco_image_id_to_delete is not None:
            self.state_manager.coco_data['images'] = [
                img for img in self.state_manager.coco_data.get('images', [])
                if img.get('id') != coco_image_id_to_delete
            ]
            self.state_manager.coco_data['annotations'] = [
                ann for ann in self.state_manager.coco_data.get('annotations', [])
                if ann.get('image_id') != coco_image_id_to_delete
            ]
        # Re-sync COCO images and filename_to_id map
        self.state_manager.sync_coco_images(self.image_files)


        # Adjust current index
        new_idx = current_idx
        if new_idx >= len(self.image_files) and self.image_files: # If deleted last, point to new last
            new_idx = len(self.image_files) - 1

        self.awaiting_delete_confirmation = False
        self.state_manager.save_state() # Save immediately after structural change
        log.info(f"Image '{filename_to_delete}' and its annotations removed from state.")

        if not self.image_files:
            log.info("All images have been deleted.")
            self._load_image_by_index(0) # Will show "no images"
        else:
            self._load_image_by_index(new_idx) # Load the image now at the adjusted index

        self.force_redraw = True


    def run(self):
        log.info("Starting QuickMasker main loop...")
        if not self.image_files:
            log.error("No images found. Please check 'image_folder' in config.yaml.")
            self.ui.setup_window()
            self._draw_frame()
            self.ui.get_key_press(5000)
            self.ui.destroy_window()
            return
        if not self.sam_model.predictor:
             log.error("SAM predictor not initialized. Cannot continue.")
             return

        self.ui.setup_window()
        self.ui.set_mouse_callback(self._handle_mouse_event)

        while not self.should_quit:
            if self.force_redraw:
                self._draw_frame()
                self.force_redraw = False

            key = self.ui.get_key_press(delay_ms=20)
            if key != -1 and key != 255:
                self.should_quit = handle_key_press(key, self) # event_handler calls methods on self

        self.ui.destroy_window()
        log.info("QuickMasker finished.")

    def _draw_frame(self):
         status_info = {}
         keys_hint = "Keys: [N]ext|[P]rev|[S]ave|[C]lear|[D]el|[Q]uit"
         message = None

         if self.awaiting_delete_confirmation:
             img_name = "N/A"
             if self.image_files and 0 <= self.state_manager.get_current_index() < len(self.image_files):
                 img_name = self.image_files[self.state_manager.get_current_index()].name
             confirmation_message = f"DELETE '{img_name}'?\nThis cannot be undone.\n\nConfirm (Y/N)"
             status_info["confirmation_message"] = confirmation_message
         elif self.current_image_rgb is None:
             if self.image_files and self.state_manager.get_current_index() >= len(self.image_files):
                 message = "End of image list."
                 keys_hint = "[P]rev | [Q]uit"
             elif not self.image_files:
                 message = "No images found in folder."
                 keys_hint = "[Q]uit"
             else: # Image failed to load
                 message = "Error Loading Image (See Logs)"
             status_info["message"] = message
             status_info["keys_hint"] = keys_hint
             status_info["status_line"] = ""
         elif self.image_files:
              idx = self.state_manager.get_current_index()
              status_info["status_line"] = f"Image: {idx + 1}/{len(self.image_files)} ({self.image_files[idx].name})"
              status_info["keys_hint"] = keys_hint
              status_info["message"] = None

         self.ui.display(
             self.current_image_bgr,
             self.points,
             self.point_labels,
             self.current_mask,
             status_info,
             show_confirmation=self.awaiting_delete_confirmation,
             confirmation_message=status_info.get("confirmation_message", "")
         )
