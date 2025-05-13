"""
Handles user input events (mouse clicks and key presses) for QuickMasker
"""

import cv2
import logging
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .main_controller import AnnotationController

log = logging.getLogger(__name__)

# Define key constants
KEY_Q = ord('q')
KEY_ESC = 27
KEY_N = ord('n')
KEY_P = ord('p')
KEY_S = ord('s')
KEY_C = ord('c')
KEY_Y = ord('y')
KEY_D = ord('d')

def handle_mouse_event(event: int, x: int, y: int, flags: int,
                       controller_points: List[Tuple[int, int]],
                       controller_point_labels: List[int]) -> bool:
    """
    Processes mouse events, updating the controllers points and labels lists directly.
    Returns True if a point was added.
    """
    point_added = False
    if event == cv2.EVENT_LBUTTONDOWN:
        controller_points.append((x, y))
        controller_point_labels.append(1)
        log.debug(f"Added positive point: ({x}, {y})")
        point_added = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        controller_points.append((x, y))
        controller_point_labels.append(0)
        log.debug(f"Added negative point: ({x}, {y})")
        point_added = True
    return point_added

def handle_key_press(key: int, tool: 'AnnotationController') -> bool:
    """
    Processes key presses and triggers corresponding actions on the tool instance.
    Returns True if the application should quit.
    """
    should_quit = False

    if tool.awaiting_delete_confirmation:
        if key == KEY_Y:
            log.info("Deletion confirmed by user.")
            tool._delete_current_image_confirmed()
            tool.awaiting_delete_confirmation = False
        elif key == KEY_N or key == KEY_ESC:
            log.info("Deletion cancelled by user.")
            tool.awaiting_delete_confirmation = False
            tool.force_redraw = True # Redraw to remove confirmation
            # ignore other keys while waiting for confirmation
        return should_quit

    # Normal key handling
    if key == KEY_Q or key == KEY_ESC:
        log.info("Quit key pressed.")
        tool._save_state()
        should_quit = True
    elif key == KEY_N:
        log.debug("Key 'n' pressed for next image.")
        tool._navigate(1)
    elif key == KEY_P:
        log.debug("Key 'p' pressed for previous image.")
        tool._navigate(-1)
    elif key == KEY_S:
        log.debug("Key 's' pressed to save annotation.")
        tool._save_current_annotation()
    elif key == KEY_C:
        if tool.points:
            log.info("Key 'c' pressed to clear points/mask.")
            tool._reset_points_and_mask()
        else:
            log.info("Key 'c' pressed, but no points to clear.")
    elif key == KEY_D:
        log.info("Key 'd' pressed to initiate delete.")
        if tool.current_image_rgb is not None: # Only if an image is loaded
            tool.awaiting_delete_confirmation = True
            tool.force_redraw = True # To show confirmation dialog
            log.info("Awaiting confirmation to delete current image.")
        else:
            log.warning("Delete key pressed, but no image is currently loaded.")
    else:
        if key != 255 and key != -1: # Ignore "no key"
            log.debug(f"Unhandled key press: {key}")

    return should_quit
