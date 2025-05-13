"""
Handles loading and validation of the YAML configuration file for QuickMasker
"""

import yaml
from pathlib import Path
import torch
import cv2 
import logging
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

### Default config structure
# mirror structure of config.yaml
DEFAULT_CONFIG = {
    "paths": {
        "image_folder": "./data/images_to_annotate/",
        "output_coco_file": "./data/outputs/output_annotations.json",
        "state_file": "./data/outputs/annotation_state.json",
        "sam_checkpoint_dir": "./models/",
    },
    "sam_model": {
        "model_type": "vit_b", # Default to smallest
        "device": "cuda",
    },
    "annotation": {
        "default_category_id": 1,
        "default_category_name": "object",
    },
    "ui": {
        "window_name": "QuickMasker",
        "window_width": 1280,
        "window_height": 720,
        "font_face_str": "FONT_HERSHEY_SIMPLEX",
        "font_scale_large": 0.7,
        "font_scale_medium": 0.6,
        "font_scale_small": 0.5,
        "text_main_color": [220, 220, 220],
        "text_secondary_color": [170, 170, 170],
        "info_panel_bg_color": [20, 20, 20],
        "info_panel_alpha": 0.85,
        "info_panel_padding": 15,
        "info_panel_margin": 10,
        "point_radius": 6,
        "positive_color": [50, 200, 50],
        "negative_color": [50, 50, 220],
        "mask_alpha": 0.45,
        "mask_color": [130, 130, 255],
        "confirmation_bg_color": [40, 40, 70],
        "confirmation_text_color": [255, 255, 180],
        "confirmation_panel_alpha": 0.9,
    }
}

# Map font names from config string to OpenCV integer constants
FONT_MAP = {
    "FONT_HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
    "FONT_HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
    "FONT_HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
    "FONT_HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
    "FONT_HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
    "FONT_HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "FONT_HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "FONT_HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    "FONT_ITALIC": cv2.FONT_ITALIC
}


def merge_configs(default: Dict, user: Dict) -> Dict:
    """Recursively merge user config into default config."""
    merged = default.copy()
    for key, value in user.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            # Only update if key exists in default or if new top-level key
            merged[key] = value
    return merged

def load_and_validate_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads configuration from YAML, merges with defaults, validates,
    and performs adjustments (like device detection)
    """
    config = DEFAULT_CONFIG.copy()

    config_file = Path(config_path)
    if config_file.is_file():
        log.info(f"Loading user configuration from {config_path}")
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            if user_config:
                config = merge_configs(config, user_config)
            log.info("User configuration merged with defaults.")
        except yaml.YAMLError as e:
            log.error(f"Failed to parse user configuration file '{config_path}': {e}")
            return None
        except Exception as e:
            log.error(f"An unexpected error occurred loading user config: {e}")
            return None
    else:
        log.warning(f"User configuration file '{config_path}' not found. Using default settings.")
        log.warning("It is recommended to create a config.yaml to customize paths and settings.")

    try:
        # Path validation
        img_folder_path = Path(config['paths']['image_folder'])
        if not img_folder_path.is_dir():
            log.error(f"'paths.image_folder' is not a valid directory: '{img_folder_path}'")
            if str(img_folder_path) == DEFAULT_CONFIG['paths']['image_folder']:
                try:
                    img_folder_path.mkdir(parents=True, exist_ok=True)
                    log.info(f"Created default image folder: {img_folder_path}")
                except OSError as e:
                    log.error(f"Could not create default image folder {img_folder_path}: {e}")
                    return None
            else:
                return None # Fail if user-specified path is invalid

        # Ensure other output directories exist
        Path(config['paths']['output_coco_file']).parent.mkdir(parents=True, exist_ok=True)
        Path(config['paths']['state_file']).parent.mkdir(parents=True, exist_ok=True)
        Path(config['paths']['sam_checkpoint_dir']).mkdir(parents=True, exist_ok=True)

        # Font validation
        font_name = config['ui']['font_face_str']
        if font_name not in FONT_MAP:
            log.error(f"Invalid font name '{font_name}' in config. Available: {list(FONT_MAP.keys())}")
            return None
        config['ui']['font_face'] = FONT_MAP[font_name] # Add integer version

        # Device validation
        sam_config = config['sam_model']
        req_device = sam_config['device']
        if req_device == 'cuda':
            if torch.cuda.is_available():
                sam_config['device'] = 'cuda'
            else:
                log.warning("Config requested 'cuda' but not available. Falling back to 'cpu'.")
                sam_config['device'] = 'cpu'
        elif req_device != 'cpu':
            log.warning(f"Invalid device '{req_device}' specified. Using 'cpu'.")
            sam_config['device'] = 'cpu'

        # SAM model type validation
        if sam_config['model_type'] not in ["vit_h", "vit_l", "vit_b"]:
             log.error(f"Invalid SAM model_type: {sam_config['model_type']}. Choose vit_h, vit_l, or vit_b.")
             return None

        log.info("Configuration validated successfully.")
        log.info(f"Effective SAM device: {sam_config['device']}")
        log.info(f"Effective SAM model type: {sam_config['model_type']}")
        return config

    except KeyError as e:
        log.error(f"Missing a required configuration key: {e}. Please ensure your config.yaml has all necessary sections (paths, sam_model, annotation, ui).")
        return None
    except Exception as e:
        log.error(f"Unexpected error during configuration validation: {e}", exc_info=True)
        return None

