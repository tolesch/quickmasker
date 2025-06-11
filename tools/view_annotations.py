"""
Annotation Viewer for the QuickMasker application.
This script uses the main application's OpenCVUI class to display images and annotations.
"""
import cv2
import logging
import json
from pathlib import Path
import yaml
from enum import Enum, auto
from quickmasker.ui import OpenCVUI
from pycocotools import mask as mask_util


class DisplayMode(Enum):
    BOTH = auto()
    MASK = auto()
    BOX = auto()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> dict:
    log.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class AnnotationViewer:
    """
    Orchestrates the annotation viewing workflow, connecting the data
    to the shared OpenCVUI class.
    """
    def __init__(self, config: dict):
        log.info("Initializing AnnotationViewer...")
        self.config = config
        self.should_quit = False
        
        # pass 'ui' section and 'paths' of config
        self.ui = OpenCVUI(config['ui'])
        self.image_folder = Path(config['paths']['image_folder'])
        self.coco_file_path = Path(config['paths']['output_coco_file'])

        self.images_info: list = []
        self.annotations_by_image_id: dict = {}
        self.current_index = 0
        
        self.display_mode = DisplayMode.BOTH
        self.display_modes = [DisplayMode.BOTH, DisplayMode.MASK, DisplayMode.BOX]
    
    def _load_data(self) -> bool:
        """Loads the COCO annotation file and prepares data structures."""
        if not self.coco_file_path.exists():
            log.error(f"Annotation file not found at: {self.coco_file_path}")
            return False
        with open(self.coco_file_path, 'r') as f:
            coco_data = json.load(f)
        self.images_info = sorted(coco_data.get('images', []), key=lambda img: img['file_name'])
        annotations = coco_data.get('annotations', [])
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image_id:
                self.annotations_by_image_id[img_id] = []
            self.annotations_by_image_id[img_id].append(ann)
        log.info(f"Loaded {len(self.images_info)} images and {len(annotations)} total annotations.")
        return True

    def _draw_frame(self):
        """Prepares data and tells the UI class to draw it."""
        image_to_show = None
        status_info = {}
        annotations_to_show = []
        
        if not self.images_info:
            status_info["message"] = "No Images Found"
            status_info["keys_hint"] = "[Q]uit"
        else:
            image_info = self.images_info[self.current_index]
            image_path = self.image_folder / image_info['file_name']
            
            if not image_path.exists():
                status_info["message"] = f"MISSING: {image_info['file_name']}"
                status_info["status_line"] = f"Image {self.current_index + 1}/{len(self.images_info)}"
            else:
                image_to_show = cv2.imread(str(image_path))
                annotations_to_show = self.annotations_by_image_id.get(image_info['id'], [])
            
            # Prepare status info for the side panel
            status_info["status_line"] = f"Image: {self.current_index + 1}/{len(self.images_info)} ({image_info['file_name']})"
            mode_text = f"Mode: {self.display_mode.name}"
            keys_text = "N:Next|P:Prev|B:Toggle|Q:Quit"
            status_info["keys_hint"] = f"{mode_text}\n{keys_text}"
            
            ## Pack viewer-specific data into the status_info dictionary
            status_info["viewer_annotations"] = annotations_to_show
            status_info["viewer_display_mode"] = self.display_mode.name

        self.ui.display(
            original_image_bgr=image_to_show,
            points_original_coords=[], # Not used by viewer
            point_labels=[],          # Not used by viewer
            mask_original=None,       # Not used by viewer
            status_info=status_info
        )

    def _navigate(self, direction: int):
        """Changes the current image index."""
        new_index = self.current_index + direction
        if 0 <= new_index < len(self.images_info):
            self.current_index = new_index
            self._draw_frame()

    def _toggle_display_mode(self):
        """Cycles to the next display mode."""
        current_idx = self.display_modes.index(self.display_mode)
        next_idx = (current_idx + 1) % len(self.display_modes)
        self.display_mode = self.display_modes[next_idx]
        log.info(f"Display mode changed to: {self.display_mode.name}")
        self._draw_frame()

    def run(self):
        """Starts the main viewer loop, delegating UI to the ui object."""
        if not self._load_data():
            self._draw_frame()
            self.ui.get_key_press(5000)
            return

        self.ui.setup_window()
        self._draw_frame()

        while not self.should_quit:
            # Use the UI class to get key presses
            key = self.ui.get_key_press(0) # Wait indefinitely

            if key == ord('q'):
                self.should_quit = True
            elif key == ord('n') or key == 83:
                self._navigate(1)
            elif key == ord('p') or key == 81:
                self._navigate(-1)
            elif key == ord('b'):
                self._toggle_display_mode()

        self.ui.destroy_window()
        log.info("Annotation Viewer closed.")

if __name__ == '__main__':
    try:
        app_config = load_config('config.yaml')
        viewer = AnnotationViewer(app_config)
        viewer.run()
    except Exception as e:
        log.exception("An unhandled error occurred.")