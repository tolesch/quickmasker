"""
Handles the OpenCV user interface for QuickMasker.
Manages window creation, drawing, and event capturing with an updated design.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Callable, Optional, List, Tuple
from .config import FONT_MAP

log = logging.getLogger(__name__)

class OpenCVUI:
    """Manages the OpenCV window and drawing with an enhanced design."""

    def __init__(self, ui_config: Dict[str, Any]):
        self.config = ui_config
        self.window_name = self.config.get("window_name", "QuickMasker")

        # Font settings
        self.font_face_str = self.config.get("font_face_str", "FONT_HERSHEY_SIMPLEX")
        self.font_face = FONT_MAP.get(self.font_face_str, cv2.FONT_HERSHEY_SIMPLEX)
        self.font_scale_large = self.config.get("font_scale_large", 0.7)
        self.font_scale_medium = self.config.get("font_scale_medium", 0.6)
        self.font_scale_small = self.config.get("font_scale_small", 0.5)
        self.text_main_color = tuple(self.config.get("text_main_color", [220, 220, 220]))
        self.text_secondary_color = tuple(self.config.get("text_secondary_color", [170, 170, 170]))

        # Info Panel settings
        self.info_panel_bg_color = tuple(self.config.get("info_panel_bg_color", [20, 20, 20]))
        self.info_panel_alpha = self.config.get("info_panel_alpha", 0.85)
        self.info_panel_padding = self.config.get("info_panel_padding", 15)
        self.info_panel_margin = self.config.get("info_panel_margin", 10)

        # Points & Mask settings
        self.point_radius = self.config.get("point_radius", 6)
        self.positive_color = tuple(self.config.get("positive_color", [50, 200, 50]))
        self.negative_color = tuple(self.config.get("negative_color", [50, 50, 220]))
        self.mask_alpha = self.config.get("mask_alpha", 0.45)
        self.mask_color = tuple(self.config.get("mask_color", [130, 130, 255])) # Accent for mask

        # Confirmation Dialog settings
        self.confirmation_bg_color = tuple(self.config.get("confirmation_bg_color", [40, 40, 70]))
        self.confirmation_text_color = tuple(self.config.get("confirmation_text_color", [255, 255, 180]))
        self.confirmation_panel_alpha = self.config.get("confirmation_panel_alpha", 0.9)

        self.window_width = self.config.get("window_width", 1600)
        self.window_height = self.config.get("window_height", 900)

        self._mouse_callback_func: Optional[Callable] = None

    def setup_window(self):
        log.info(f"Setting up OpenCV window: {self.window_name}")
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        except Exception as e:
            log.error(f"Failed to create or resize OpenCV window: {e}", exc_info=True)
            raise

    def set_mouse_callback(self, callback_func: Callable):
        self._mouse_callback_func = callback_func
        try:
            cv2.setMouseCallback(self.window_name, self._on_mouse)
            log.info("Mouse callback set successfully.")
        except Exception as e:
            log.error(f"Failed to set mouse callback: {e}", exc_info=True)

    def _on_mouse(self, event, x, y, flags, param):
        if self._mouse_callback_func:
            try:
                self._mouse_callback_func(event, x, y, flags, param)
            except Exception as e:
                log.error(f"Error in registered mouse callback: {e}", exc_info=True)

    def _draw_text_with_background(self, image: np.ndarray, text: str,
                                   origin: Tuple[int, int], font_face: int,
                                   font_scale: float, text_color: Tuple[int, int, int],
                                   bg_color: Tuple[int, int, int], bg_alpha: float,
                                   padding: int = 5, line_thickness: int = 1):
        """Helper to draw text with a semi-transparent background."""
        if not text: # Do not draw if text is empty
            return 0

        (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, line_thickness)
        
        # Background rectangle coordinates
        bg_x1 = origin[0] - padding
        bg_y1 = origin[1] - text_height - padding - baseline // 2 
        bg_x2 = origin[0] + text_width + padding
        bg_y2 = origin[1] + padding + baseline // 2

        h, w = image.shape[:2]
        bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1)
        bg_x2, bg_y2 = min(w, bg_x2), min(h, bg_y2)

        if bg_x2 > bg_x1 and bg_y2 > bg_y1: 
            sub_img = image[bg_y1:bg_y2, bg_x1:bg_x2]
            bg_rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 1 - bg_alpha, bg_rect, bg_alpha, 1.0)
            image[bg_y1:bg_y2, bg_x1:bg_x2] = res

        text_origin = (origin[0], origin[1]) 
        cv2.putText(image, text, text_origin, font_face, font_scale, text_color, line_thickness, cv2.LINE_AA)
        return text_height + baseline + (2 * padding)


    def display(self,
                image_bgr: Optional[np.ndarray],
                points: List[Tuple[int, int]],
                point_labels: List[int],
                mask: Optional[np.ndarray],
                status_info: Dict[str, Any],
                show_confirmation: bool = False,
                confirmation_message: str = ""):
        """Draws the UI with the new design."""
        draw_canvas: Optional[np.ndarray] = None

        if image_bgr is None and not show_confirmation:
            h, w = self.window_height, self.window_width
            draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.rectangle(draw_canvas, (0,0), (w,h), self.info_panel_bg_color, -1)

            status_text = status_info.get("message", "No Image Loaded")
            (tw, th), _ = cv2.getTextSize(status_text, self.font_face, self.font_scale_large, 2)
            text_x = (w - tw) // 2
            text_y = (h + th) // 2
            cv2.putText(draw_canvas, status_text, (text_x, text_y), self.font_face, self.font_scale_large, self.text_main_color, 2, cv2.LINE_AA)

            keys_text = status_info.get("keys_hint", "[Q]uit")
            (tw_keys, th_keys), _ = cv2.getTextSize(keys_text, self.font_face, self.font_scale_medium, 1)
            # Use text_secondary_color here as intended
            cv2.putText(draw_canvas, keys_text, (self.info_panel_margin, h - self.info_panel_margin - th_keys),
                        self.font_face, self.font_scale_medium, self.text_secondary_color, 1, cv2.LINE_AA)

        elif image_bgr is not None:
            draw_canvas = image_bgr.copy()

            if mask is not None and not show_confirmation:
                try:
                    mask_bool = mask > 0
                    roi = draw_canvas[mask_bool]
                    if roi.size > 0:
                        mask_color_overlay = np.full_like(roi, self.mask_color)
                        blended_roi = cv2.addWeighted(roi, 1 - self.mask_alpha, mask_color_overlay, self.mask_alpha, 0)
                        draw_canvas[mask_bool] = blended_roi
                except Exception as e: log.warning(f"Error applying mask overlay: {e}")

            if not show_confirmation:
                for i, point in enumerate(points):
                    if i < len(point_labels):
                        color = self.positive_color if point_labels[i] == 1 else self.negative_color
                        cv2.circle(draw_canvas, point, self.point_radius, color, -1)
                        cv2.circle(draw_canvas, point, self.point_radius, (0,0,0), 1)

            ### Draw Info Panel (Status & Instructions)
            panel_x = self.info_panel_margin
            current_y = self.info_panel_margin + self.info_panel_padding # Start y inside padding

            status_line = status_info.get("status_line", "")
            if status_line:
                current_y += self._draw_text_with_background(
                    draw_canvas, status_line, (panel_x + self.info_panel_padding, current_y),
                    self.font_face, self.font_scale_large, self.text_main_color,
                    self.info_panel_bg_color, self.info_panel_alpha, padding=5
                ) + 5

            points_line = f"Points: {len(points)} (LMB: Pos [+] | RMB: Neg [-])"
            current_y += self._draw_text_with_background(
                draw_canvas, points_line, (panel_x + self.info_panel_padding, current_y),
                self.font_face, self.font_scale_medium, self.text_secondary_color, # Using secondary color
                self.info_panel_bg_color, self.info_panel_alpha, padding=5
            ) + 5

            keys_line = status_info.get("keys_hint", "N:Next|P:Prev|S:Save|C:Clear|D:Del|Q:Quit")
            current_y += self._draw_text_with_background(
                draw_canvas, keys_line, (panel_x + self.info_panel_padding, current_y),
                self.font_face, self.font_scale_medium, self.text_secondary_color, # Using secondary color
                self.info_panel_bg_color, self.info_panel_alpha, padding=5
            )
        elif image_bgr is None and show_confirmation:
             h, w = self.window_height, self.window_width
             draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
             cv2.rectangle(draw_canvas, (0,0), (w,h), self.info_panel_bg_color, -1)
             log.warning("Attempting to show confirmation dialog without a background image.")


        if show_confirmation and draw_canvas is not None:
            h_canvas, w_canvas = draw_canvas.shape[:2]
            msg_lines = confirmation_message.split('\n')
            max_text_width = 0
            total_text_height = (len(msg_lines) -1) * 5
            for line in msg_lines:
                (tw, th), _ = cv2.getTextSize(line, self.font_face, self.font_scale_large, 1)
                if tw > max_text_width: max_text_width = tw
                total_text_height += th + 10

            box_w = max_text_width + 4 * self.info_panel_padding
            box_h = total_text_height + 2 * self.info_panel_padding
            box_w = max(300, box_w); box_h = max(100, box_h)

            x1, y1 = (w_canvas - box_w) // 2, (h_canvas - box_h) // 2
            x2, y2 = x1 + box_w, y1 + box_h

            overlay = draw_canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.confirmation_bg_color, -1)
            cv2.addWeighted(overlay, self.confirmation_panel_alpha, draw_canvas, 1 - self.confirmation_panel_alpha, 0, draw_canvas)

            current_text_y = y1 + self.info_panel_padding + cv2.getTextSize("T", self.font_face, self.font_scale_large, 1)[0][1]
            for line in msg_lines:
                (tw, th), _ = cv2.getTextSize(line, self.font_face, self.font_scale_large, 1)
                text_x = x1 + (box_w - tw) // 2
                cv2.putText(draw_canvas, line, (text_x, current_text_y),
                            self.font_face, self.font_scale_large, self.confirmation_text_color, 1, cv2.LINE_AA)
                current_text_y += th + 10


        if draw_canvas is not None:
            try:
                cv2.imshow(self.window_name, draw_canvas)
            except Exception as e:
                log.error(f"Error displaying image in OpenCV window '{self.window_name}': {e}", exc_info=True)
        else:
             log.warning("Draw canvas was None at the end of display function.")
             h_fb, w_fb = self.window_height, self.window_width
             fallback_canvas = np.zeros((h_fb, w_fb, 3), dtype=np.uint8)
             cv2.putText(fallback_canvas, "Error displaying content", (50, h_fb//2), self.font_face, 1, (0,0,255),2)
             cv2.imshow(self.window_name, fallback_canvas)


    def get_key_press(self, delay_ms: int = 20) -> int:
        try:
            return cv2.waitKey(delay_ms) & 0xFF
        except Exception as e:
             log.error(f"Error during cv2.waitKey: {e}", exc_info=True)
             return -1

    def destroy_window(self):
        log.info(f"Destroying OpenCV window: {self.window_name}")
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except Exception as e:
            log.error(f"Error destroying window '{self.window_name}': {e}", exc_info=True)

