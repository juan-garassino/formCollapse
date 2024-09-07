import svgwrite
import os
import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def save_svg(data: np.ndarray, filename: str, output_dir: str, canvas_size: Tuple[int, int] = (1000, 1000)) -> None:
    """
    Save the given data as an SVG file.
    Normalizes the data to fit the specified canvas size and creates a polyline representation.
    Saves the SVG file in the specified output directory.
    """
    svg_dir = os.path.join(output_dir, 'svg')
    os.makedirs(svg_dir, exist_ok=True)
    full_path = os.path.join(svg_dir, f"{filename}.svg")

    try:
        dwg = svgwrite.Drawing(full_path, size=canvas_size)
        
        # Normalize data to fit the SVG canvas
        normalized_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        scaled_data = normalized_data * (canvas_size[0] - 1)  # Subtract 1 to avoid edge cases
        
        # Create polyline
        polyline = dwg.polyline(points=scaled_data.tolist(), 
                                fill='none', 
                                stroke='black', 
                                stroke_width=1)
        dwg.add(polyline)
        
        dwg.save()
        logger.info(f"Saved SVG as {full_path}")
    except Exception as e:
        logger.error(f"Failed to save SVG: {str(e)}")
        raise

def generate_gcode(data: np.ndarray, filename: str, output_dir: str, canvas_size: Tuple[int, int] = (100, 100), z_up: float = 5, z_down: float = 0) -> None:
    """
    Generate G-code from the given data points.
    Normalizes the data to fit the specified canvas size and creates G-code commands.
    Saves the G-code file in the specified output directory.
    """
    gcode_dir = os.path.join(output_dir, 'gcode')
    os.makedirs(gcode_dir, exist_ok=True)
    full_path = os.path.join(gcode_dir, f"{filename}.gcode")

    try:
        # Normalize data to fit the canvas size
        normalized_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        scaled_data = normalized_data * (canvas_size[0] - 1)  # Subtract 1 to avoid edge cases
        
        with open(full_path, 'w') as f:
            # G-code preamble
            f.write("G21 ; Set units to millimeters\n")
            f.write("G90 ; Use absolute coordinates\n")
            f.write(f"G0 Z{z_up} ; Raise pen\n")
            
            # Write path
            for i, (x, y, _) in enumerate(scaled_data):
                if i == 0:
                    f.write(f"G0 X{x:.2f} Y{y:.2f} ; Move to start position\n")
                    f.write(f"G0 Z{z_down} ; Lower pen\n")
                else:
                    f.write(f"G1 X{x:.2f} Y{y:.2f} ; Draw to point\n")
            
            # Finish up
            f.write(f"G0 Z{z_up} ; Raise pen\n")
            f.write("G0 X0 Y0 ; Return to origin\n")
        
        logger.info(f"Saved G-code as {full_path}")
    except Exception as e:
        logger.error(f"Failed to generate G-code: {str(e)}")
        raise