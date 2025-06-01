"""
Handles the OpenCV user interface for QuickMasker.
Manages window creation, drawing, and event capturing.
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

        # Panel settings from config
        self.instruction_panel_width = self.config.get("instruction_panel_width", 300) # Default if not in config
        self.canvas_bg_color = tuple(self.config.get("canvas_bg_color", [30, 30, 30])) # BGR for main canvas
        self.info_panel_bg_color = tuple(self.config.get("info_panel_bg_color", [40, 40, 45])) # BGR for instruction panel
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

        # Calculated properties for image display area
        self.image_display_rect = (0, 0, 0, 0)  # x, y, width, height of the scaled image on canvas
        self.image_scale_factor = 1.0
        self.image_offset_in_display_rect = (0,0) # Offset of scaled image within its designated region

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
            cv2.setMouseCallback(self.window_name, self._on_mouse_wrapper)
            log.info("Mouse callback set successfully.")
        except Exception as e:
            log.error(f"Failed to set mouse callback: {e}", exc_info=True)

    def _on_mouse_wrapper(self, event, x, y, flags, param):
        """Wrapper for mouse callback to transform coordinates."""
        if self._mouse_callback_func:
            img_rect_x, img_rect_y, img_rect_w, img_rect_h = self.image_display_rect
            
            # Check if click is within the actual scaled image area
            if img_rect_x <= x < img_rect_x + img_rect_w and \
               img_rect_y <= y < img_rect_y + img_rect_h:
                
                if self.image_scale_factor == 0: # Avoid division by zero
                    log.warning("Mouse event on image region, but scale factor is zero.")
                    return

                # Translate window coordinates (relative to scaled images top-left on canvas) to original image coordinates.
                original_x = int((x - img_rect_x) / self.image_scale_factor)
                original_y = int((y - img_rect_y) / self.image_scale_factor)
                
                try:
                    self._mouse_callback_func(event, original_x, original_y, flags, param)
                except Exception as e:
                    log.error(f"Error in registered mouse callback: {e}", exc_info=True)
            # else: log.debug("Click outside scaled image region")


    def _wrap_text_cv(self, text: str, font_face: int, font_scale: float, thickness: int, max_width: int) -> List[str]:
        """Simple text wrapper for OpenCV."""
        lines = []
        words = text.split(' ')
        current_line = ""
        for word in words:
            test_line = current_line + (" " + word if current_line else word)
            (text_width, _), _ = cv2.getTextSize(test_line, font_face, font_scale, thickness)
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line: lines.append(current_line)
                current_line = word
                # Handle case where a single word is too long (simple truncation)
                (word_width, _), _ = cv2.getTextSize(current_line, font_face, font_scale, thickness)
                if word_width > max_width:
                    # Truncate word with ellipsis
                    truncated_word = ""
                    for char_idx, char_val in enumerate(current_line):
                        test_char_line = truncated_word + char_val
                        (tw_char, _), _ = cv2.getTextSize(test_char_line + "...", font_face, font_scale, thickness)
                        if tw_char <= max_width:
                            truncated_word = test_char_line
                        else:
                            break
                    current_line = truncated_word + "..." if truncated_word else "..."

        if current_line: lines.append(current_line)
        return lines if lines else [text]


    def display(self,
                original_image_bgr: Optional[np.ndarray],
                points_original_coords: List[Tuple[int, int]],
                point_labels: List[int],
                mask_original: Optional[np.ndarray],
                status_info: Dict[str, Any],
                show_confirmation: bool = False,
                confirmation_message: str = ""):
        """Draws the UI with a side panel for instructions."""

        # Create main canvas
        main_canvas = np.full((self.window_height, self.window_width, 3), self.canvas_bg_color, dtype=np.uint8)

        # Define regions
        img_region_x = 0
        img_region_y = 0
        img_region_w = self.window_width - self.instruction_panel_width - self.info_panel_margin
        img_region_h = self.window_height

        instr_panel_x = img_region_w + self.info_panel_margin
        instr_panel_y = self.info_panel_margin
        instr_panel_w = self.instruction_panel_width - (2 * self.info_panel_margin) # Panel width itself
        instr_panel_h = self.window_height - (2 * self.info_panel_margin) # Panel height itself

        # Draw instruction panel
        cv2.rectangle(main_canvas, 
                      (instr_panel_x, instr_panel_y), 
                      (instr_panel_x + instr_panel_w, instr_panel_y + instr_panel_h), 
                      self.info_panel_bg_color, -1)

        # Image Display Area
        scaled_w, scaled_h = 0,0
        if original_image_bgr is not None:
            orig_h, orig_w = original_image_bgr.shape[:2]
            if orig_w > 0 and orig_h > 0 and img_region_w > 0 and img_region_h > 0:
                scale_w_factor = img_region_w / orig_w
                scale_h_factor = img_region_h / orig_h
                self.image_scale_factor = min(scale_w_factor, scale_h_factor)
                if self.image_scale_factor <=0 : self.image_scale_factor = 1.0 # Safety

                scaled_w = int(orig_w * self.image_scale_factor)
                scaled_h = int(orig_h * self.image_scale_factor)

                display_img_scaled = cv2.resize(original_image_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

                # Center scaled image within img_region
                offset_x = img_region_x + (img_region_w - scaled_w) // 2
                offset_y = img_region_y + (img_region_h - scaled_h) // 2
                self.image_offset_in_display_rect = (offset_x, offset_y) # Store offset of image from (0,0) of window
                self.image_display_rect = (offset_x, offset_y, scaled_w, scaled_h)

                # Blit scaled image
                main_canvas[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = display_img_scaled

                # Draw mask on the scaled image area
                if mask_original is not None and not show_confirmation:
                    if scaled_w > 0 and scaled_h > 0:
                        scaled_mask = cv2.resize(mask_original.astype(np.uint8), (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST) > 0
                        img_area_on_canvas = main_canvas[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w]
                        try:
                            if img_area_on_canvas.shape[:2] == scaled_mask.shape:
                                roi = img_area_on_canvas[scaled_mask]
                                if roi.size > 0:
                                    mask_color_overlay = np.full_like(roi, self.mask_color)
                                    blended_roi = cv2.addWeighted(roi, 1 - self.mask_alpha, mask_color_overlay, self.mask_alpha, 0)
                                    img_area_on_canvas[scaled_mask] = blended_roi
                            else: log.warning("Scaled mask shape mismatch for drawing on image area.")
                        except Exception as e: log.warning(f"Error applying scaled mask: {e}")
                
                # Draw points (scaled to displayed image)
                if not show_confirmation:
                    for i, (orig_pt_x, orig_pt_y) in enumerate(points_original_coords):
                        scaled_pt_x = offset_x + int(orig_pt_x * self.image_scale_factor)
                        scaled_pt_y = offset_y + int(orig_pt_y * self.image_scale_factor)
                        if i < len(point_labels):
                            color = self.positive_color if point_labels[i] == 1 else self.negative_color
                            cv2.circle(main_canvas, (scaled_pt_x, scaled_pt_y), self.point_radius, color, -1)
                            cv2.circle(main_canvas, (scaled_pt_x, scaled_pt_y), self.point_radius, (0,0,0), 1)
            else: # Image invalid or region too small
                self.image_display_rect = (img_region_x, img_region_y, img_region_w, img_region_h)
                self.image_scale_factor = 1.0; self.image_offset_in_display_rect = (0,0)
                no_img_text = "Error: Image Invalid"
                (tw, th), _ = cv2.getTextSize(no_img_text, self.font_face, self.font_scale_large, 2)
                cv2.putText(main_canvas, no_img_text,
                            (img_region_x + (img_region_w - tw) // 2, img_region_y + (img_region_h + th) // 2),
                            self.font_face, self.font_scale_large, self.text_main_color, 2, cv2.LINE_AA)
        else: # No image loaded, and not showing confirmation
            self.image_display_rect = (img_region_x, img_region_y, img_region_w, img_region_h)
            self.image_scale_factor = 1.0; self.image_offset_in_display_rect = (0,0)
            no_img_text = status_info.get("message", "No Image Loaded")
            (tw, th), _ = cv2.getTextSize(no_img_text, self.font_face, self.font_scale_large, 2)
            cv2.putText(main_canvas, no_img_text,
                        (img_region_x + (img_region_w - tw) // 2, img_region_y + (img_region_h + th) // 2),
                        self.font_face, self.font_scale_large, self.text_main_color, 2, cv2.LINE_AA)

        # Draw Text in Instruction Panel
        text_panel_x_start = instr_panel_x + self.info_panel_padding
        current_y_instr = instr_panel_y + self.info_panel_padding
        line_thickness_instr = 1
        line_spacing_instr = 10 # Space between different text blocks
        wrapped_line_spacing = 5 # Space between wrapped lines of the same text block

        # Status Line
        status_line_text = status_info.get("status_line", "")
        if status_line_text:
            wrapped_status = self._wrap_text_cv(status_line_text, self.font_face, self.font_scale_large, line_thickness_instr, instr_panel_w - 2 * self.info_panel_padding)
            for line in wrapped_status:
                if current_y_instr + 20 > instr_panel_y + instr_panel_h - self.info_panel_padding : break # Check height
                (tw, th), baseline = cv2.getTextSize(line, self.font_face, self.font_scale_large, line_thickness_instr)
                cv2.putText(main_canvas, line, (text_panel_x_start, current_y_instr + th),
                            self.font_face, self.font_scale_large, self.text_main_color, line_thickness_instr, cv2.LINE_AA)
                current_y_instr += th + baseline + wrapped_line_spacing
            current_y_instr += (line_spacing_instr - wrapped_line_spacing) # Add larger gap after block

        # Points Line
        if current_y_instr < instr_panel_y + instr_panel_h - self.info_panel_padding - 20: # Check height before next block
            points_text_val = f"Points: {len(points_original_coords)} (LMB: Pos [+] | RMB: Neg [-])"
            wrapped_points = self._wrap_text_cv(points_text_val, self.font_face, self.font_scale_medium, line_thickness_instr, instr_panel_w - 2 * self.info_panel_padding)
            for line in wrapped_points:
                if current_y_instr + 20 > instr_panel_y + instr_panel_h - self.info_panel_padding : break
                (tw, th), baseline = cv2.getTextSize(line, self.font_face, self.font_scale_medium, line_thickness_instr)
                cv2.putText(main_canvas, line, (text_panel_x_start, current_y_instr + th),
                            self.font_face, self.font_scale_medium, self.text_secondary_color, line_thickness_instr, cv2.LINE_AA)
                current_y_instr += th + baseline + wrapped_line_spacing
            current_y_instr += (line_spacing_instr - wrapped_line_spacing)

        # Keys Line
        if current_y_instr < instr_panel_y + instr_panel_h - self.info_panel_padding - 20:
            keys_text_val = status_info.get("keys_hint", "N:Next|P:Prev|S:Save|C:Clear|D:Del|Q:Quit")
            wrapped_keys = self._wrap_text_cv(keys_text_val, self.font_face, self.font_scale_medium, line_thickness_instr, instr_panel_w - 2 * self.info_panel_padding)
            for line in wrapped_keys:
                if current_y_instr + 20 > instr_panel_y + instr_panel_h - self.info_panel_padding : break
                (tw, th), baseline = cv2.getTextSize(line, self.font_face, self.font_scale_medium, line_thickness_instr)
                cv2.putText(main_canvas, line, (text_panel_x_start, current_y_instr + th),
                            self.font_face, self.font_scale_medium, self.text_secondary_color, line_thickness_instr, cv2.LINE_AA)
                current_y_instr += th + baseline + wrapped_line_spacing
            # No extra spacing after the last block

        # Draw Confirmation Dialog (if active, overlays on image region)
        if show_confirmation:
            conf_region_x, conf_region_y, conf_region_w, conf_region_h = self.image_display_rect
            if conf_region_w <= 0 or conf_region_h <= 0: # Fallback if image region not set
                conf_region_w = img_region_w if img_region_w > 0 else self.window_width // 2
                conf_region_h = img_region_h if img_region_h > 0 else self.window_height // 2
                conf_region_x = img_region_x if img_region_w > 0 else self.info_panel_margin
                conf_region_y = img_region_y if img_region_h > 0 else self.info_panel_margin

            msg_lines = confirmation_message.split('\n')
            max_text_w_conf = 0; conf_line_heights = []; total_text_h_conf = (len(msg_lines) -1) * 5
            conf_cv_scale = self.font_scale_large; conf_cv_thickness = 2 # Bolder for confirmation

            for line in msg_lines:
                (tw_cv, th_cv), _ = cv2.getTextSize(line, self.font_face, conf_cv_scale, conf_cv_thickness)
                conf_line_heights.append(th_cv)
                if tw_cv > max_text_w_conf: max_text_w_conf = tw_cv
                total_text_h_conf += th_cv + 10

            box_padding = self.info_panel_padding * 2
            box_w = min(conf_region_w - 20, max_text_w_conf + 2 * box_padding)
            box_h = min(conf_region_h - 20, total_text_h_conf + box_padding) # Adjusted padding
            box_w = max(300, int(box_w)); box_h = max(100, int(box_h))

            box_x1 = conf_region_x + (conf_region_w - box_w) // 2
            box_y1 = conf_region_y + (conf_region_h - box_h) // 2

            conf_overlay_target_region = main_canvas[box_y1 : box_y1 + box_h, box_x1 : box_x1 + box_w]
            if conf_overlay_target_region.size > 0:
                bg_rect_conf = np.full(conf_overlay_target_region.shape, self.confirmation_bg_color, dtype=np.uint8)
                main_canvas[box_y1 : box_y1 + box_h, box_x1 : box_x1 + box_w] = cv2.addWeighted(
                    conf_overlay_target_region, 1 - self.confirmation_panel_alpha,
                    bg_rect_conf, self.confirmation_panel_alpha, 0
                )

            current_y_conf_text = box_y1 + box_padding
            for i, line in enumerate(msg_lines):
                line_h_cv = conf_line_heights[i]
                (tw, _), baseline = cv2.getTextSize(line, self.font_face, conf_cv_scale, conf_cv_thickness)
                text_x_conf = box_x1 + (box_w - tw) // 2
                # Adjust y for each line to be correctly placed (baseline rendering)
                cv2.putText(main_canvas, line, (text_x_conf, current_y_conf_text + line_h_cv), # Draw at baseline
                            self.font_face, conf_cv_scale, self.confirmation_text_color, conf_cv_thickness, cv2.LINE_AA)
                current_y_conf_text += line_h_cv + baseline + 5 # Add spacing for next line

        # Display final canvas
        try:
            cv2.imshow(self.window_name, main_canvas)
        except Exception as e:
            log.error(f"Error displaying final canvas: {e}", exc_info=True)


    def get_key_press(self, delay_ms: int = 20) -> int:
        try: return cv2.waitKey(delay_ms) & 0xFF
        except Exception as e: log.error(f"Error during cv2.waitKey: {e}", exc_info=True); return -1

    def destroy_window(self):
        log.info(f"Destroying OpenCV window: {self.window_name}")
        try: cv2.destroyWindow(self.window_name); cv2.waitKey(1)
        except Exception as e: log.error(f"Error destroying window: {e}", exc_info=True)

