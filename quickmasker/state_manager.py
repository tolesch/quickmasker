"""
Manages the application state, including loading/saving progress
and handling the COCO annotation data structure.
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from .utils.coco_utils import mask_to_rle, get_bbox_from_mask

log = logging.getLogger(__name__)

class StateManager:
    """Handles loading, saving, and managing annotation state and COCO data."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the StateManager.

        Args:
            config (Dict[str, Any]): The application configuration dictionary.
        """
        self.paths_config = config['paths']
        self.annotation_config = config['annotation']
        self.state_file = Path(self.paths_config['state_file'])
        self.output_coco_file = Path(self.paths_config['output_coco_file'])

        self.current_image_index: int = 0
        self.coco_data: Dict[str, Any] = self._initialize_coco_data()
        self.filename_to_coco_id: Dict[str, int] = {}

    def _initialize_coco_data(self) -> Dict[str, Any]:
        """Creates the basic COCO data structure."""
        log.info("Initializing new COCO data structure.")
        return {
            "info": {"description": "Annotations created with QuickMasker"},
            "licenses": [], "images": [], "annotations": [],
            "categories": [{"id": self.annotation_config["default_category_id"],
                            "name": self.annotation_config["default_category_name"],
                            "supercategory": ""}]
        }

    def load_state(self):
        """Loads the application state (index and COCO data) from the state file."""
        if self.state_file.exists():
            log.info(f"Loading state from {self.state_file}...")
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                # Load COCO data
                self.coco_data = state.get("coco_data", self._initialize_coco_data())
                log.info(f"Loaded {len(self.coco_data.get('annotations', []))} existing annotations.")
                # Load index after potentially syncing images (done by controller)
                self.current_image_index = state.get("current_image_index", 0)
                log.info(f"Loaded starting image index: {self.current_image_index}")

            except json.JSONDecodeError as e:
                log.error(f"Error decoding JSON from state file {self.state_file}: {e}. Starting fresh.")
                self._initialize_state_and_sync([]) # Need image files to sync properly
            except Exception as e:
                log.error(f"Unexpected error loading state file: {e}. Starting fresh.", exc_info=True)
                self._initialize_state_and_sync([])
        else:
            log.info("No state file found. Starting fresh.")
            self._initialize_state_and_sync([])

    def _initialize_state_and_sync(self, image_files: List[Path]):
        """Helper for fresh starts, requires image file list."""
        self.current_image_index = 0
        self.coco_data = self._initialize_coco_data()
        self.sync_coco_images(image_files) # Sync immediately

    def save_state(self):
        """Saves the current image index and COCO data to the state file."""
        if not self.state_file:
            log.error("State file path not configured. Cannot save state.")
            return
        state = {"current_image_index": self.current_image_index, "coco_data": self.coco_data}
        log.debug(f"Saving state: Index={self.current_image_index}, Annotations={len(self.coco_data.get('annotations',[]))}")
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
            # Also save main annotations file as primary output
            self.save_coco_file()
        except Exception as e:
            log.error(f"Error saving state to {self.state_file}: {e}", exc_info=True)

    def save_coco_file(self):
         """Saves the current COCO annotations to the final output file."""
         if not self.output_coco_file:
             log.error("Output COCO file path not configured. Cannot save annotations.")
             return
         log.info(f"Saving annotations to {self.output_coco_file}...")
         try:
             self.output_coco_file.parent.mkdir(parents=True, exist_ok=True)
             with open(self.output_coco_file, 'w', encoding='utf-8') as f:
                 json.dump(self.coco_data, f, indent=4)
             log.info("Annotations saved successfully.")
         except Exception as e:
             log.error(f"Error saving COCO file to {self.output_coco_file}: {e}", exc_info=True)

    def sync_coco_images(self, current_image_files: List[Path]):
        """
        Ensures the COCO 'images' list matches the provided list of image files found on disk.
        Updates the internal filename_to_coco_id mapping.
        Removes annotations associated with images that are no longer present.

        Args:
            current_image_files (List[Path]): The up-to-date list of image Paths found.
        """
        log.info("Syncing COCO image list with discovered files...")
        if not current_image_files:
            log.warning("No image files provided for syncing. Clearing COCO image list.")
            self.coco_data['images'] = []
            self.filename_to_coco_id = {}
            return

        existing_coco_images = self.coco_data.get('images', [])
        existing_ids = {img['id'] for img in existing_coco_images}
        max_id = max(existing_ids) if existing_ids else 0
        filename_map = {img['file_name']: img for img in existing_coco_images}

        updated_list = []
        processed_files = set() # Keep track of filenames actually found

        for idx, img_path in enumerate(current_image_files):
            filename = img_path.name
            processed_files.add(filename)

            if filename in filename_map:
                # Existing image: update index, keep ID and data
                entry = filename_map[filename]
                entry['file_index'] = idx # Ensure index matches current file list order
                updated_list.append(entry)
            else:
                # New image: assign new ID, add basic entry
                log.info(f"New image file detected: {filename}. Adding to COCO.")
                max_id += 1
                new_entry = {
                    "id": max_id, "file_name": filename,
                    "height": -1, "width": -1, # Placeholders
                    "license": None, "flickr_url": None, "coco_url": None, "date_captured": None,
                    "file_index": idx
                }
                updated_list.append(new_entry)

        # Filter list to only include images currently present
        final_list = [img for img in updated_list if img['file_name'] in processed_files]

        # Sort by file_index to maintain consistent order
        self.coco_data['images'] = sorted(final_list, key=lambda x: x.get('file_index', float('inf')))

        # Update the filename -> ID mapping
        self.filename_to_coco_id = {img['file_name']: img['id'] for img in self.coco_data['images']}

        ### Annotation Cleanup
        valid_image_ids = set(self.filename_to_coco_id.values())
        original_annotations = self.coco_data.get('annotations', [])
        filtered_annotations = [ann for ann in original_annotations if ann.get('image_id') in valid_image_ids]

        removed_count = len(original_annotations) - len(filtered_annotations)
        if removed_count > 0:
            log.warning(f"Removed {removed_count} annotations linked to images no longer found in the folder.")
        self.coco_data['annotations'] = filtered_annotations

        ### Final Index Validation
        # Ensure current_image_index is valid after sync
        if not current_image_files:
             self.current_image_index = 0
        elif self.current_image_index >= len(current_image_files):
             log.warning(f"Current index {self.current_image_index} out of bounds after sync. Resetting to last image.")
             self.current_image_index = len(current_image_files) - 1
        elif self.current_image_index < 0:
             log.warning("Current index is negative after sync. Resetting to first image.")
             self.current_image_index = 0


    def add_annotation(self, image_filename: str, binary_mask: np.ndarray) -> bool:
        """
        Creates and adds a COCO annotation for the given mask and image.

        Args:
            image_filename (str): The filename of the image the mask belongs to.
            binary_mask (np.ndarray): The boolean mask array.

        Returns:
            bool: True if annotation was added successfully, False otherwise.
        """
        if binary_mask is None or binary_mask.sum() == 0:
            log.warning("Attempted to save an empty or non-existent mask. Skipping.")
            return False

        image_id = self.filename_to_coco_id.get(image_filename)
        if image_id is None:
            log.error(f"Cannot save annotation: No COCO ID found for image '{image_filename}'.")
            return False

        log.info(f"Creating annotation for image ID {image_id} ('{image_filename}')...")
        try:
            rle = mask_to_rle(binary_mask)       # From utils
            bbox = get_bbox_from_mask(binary_mask) # From utils
            area = int(binary_mask.sum())

            # Determine next annotation ID
            current_annotations = self.coco_data.setdefault('annotations', [])
            next_ann_id = max((ann['id'] for ann in current_annotations), default=0) + 1

            annotation = {
                "id": next_ann_id, "image_id": image_id,
                "category_id": self.annotation_config["default_category_id"],
                "segmentation": rle, "area": area, "bbox": bbox, "iscrowd": 0
            }
            current_annotations.append(annotation)
            log.info(f"Annotation {next_ann_id} added successfully.")
            return True
        except Exception as e:
            log.error(f"Error creating COCO annotation: {e}", exc_info=True)
            return False

    def get_coco_data(self) -> Dict[str, Any]:
        """Returns the current COCO data dictionary."""
        return self.coco_data

    def get_current_index(self) -> int:
        """Returns the current image index."""
        return self.current_image_index

    def set_current_index(self, index: int):
        """Sets the current image index (validation happens in sync)."""
        self.current_image_index = index

    def update_image_dimensions(self, filename: str, height: int, width: int):
         """Updates the height/width for a specific image entry in COCO data."""
         image_id = self.filename_to_coco_id.get(filename)
         if image_id is not None:
             for img_entry in self.coco_data.get('images', []):
                 if img_entry['id'] == image_id:
                     if img_entry.get('height', -1) == -1 or img_entry.get('width', -1) == -1:
                         img_entry['height'] = height
                         img_entry['width'] = width
                         log.debug(f"Updated COCO dimensions for image {filename} (ID: {image_id}).")
                     break # Found image
         else:
             log.warning(f"Could not update dimensions: Image '{filename}' not found in COCO index.")