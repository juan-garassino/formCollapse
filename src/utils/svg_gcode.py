import svgwrite
import os

def save_svg(points, filename, output_dir):
    svg_dir = os.path.join(output_dir, 'svg')
    os.makedirs(svg_dir, exist_ok=True)
    full_path = os.path.join(svg_dir, f"{filename}.svg")
    
    dwg = svgwrite.Drawing(full_path, profile='tiny', size=('100%', '100%'))
    dwg.viewbox(0, 0, 1000, 1000)
    
    # Normalize points to fit the SVG viewbox
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0)) * 1000
    
    # Draw the polyline in SVG
    dwg.add(dwg.polyline(points, fill='none', stroke='black', stroke_width=1))
    dwg.save()
    
    print(f"Saved plot as {full_path}")

def generate_gcode(data, filename, output_dir, canvas_size=(100, 100), z_up=5, z_down=0):
    gcode_dir = os.path.join(output_dir, 'gcode')
    os.makedirs(gcode_dir, exist_ok=True)
    full_path = os.path.join(gcode_dir, f"{filename}.gcode")

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
    
    print(f"Saved G-code as {full_path}")