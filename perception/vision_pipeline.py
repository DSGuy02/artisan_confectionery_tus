"""
vision_pipeline.py - Computer Vision Processing for Artisan Confectionery Cell
Technological University of the Shannon (TUS Athlone)
Integrated Robotics 4 - Continuous Assessment 3
"""

import cv2
import numpy as np


def convert_webots_to_cv(raw_image_bytes, width, height):
    """
    Converts raw Webots camera buffer into an OpenCV BGR image.
    
    Args:
        raw_image_bytes: Raw bytes returned by camera.getImage()
        width: Image width in pixels (e.g. 640)
        height: Image height in pixels (e.g. 480)
        
    Returns:
        numpy.ndarray: BGR image suitable for OpenCV processing, or None.
    """
    if raw_image_bytes is None:
        return None
    bgra_array = np.frombuffer(raw_image_bytes, np.uint8).reshape((height, width, 4))
    return cv2.cvtColor(bgra_array, cv2.COLOR_BGRA2BGR)


def classify_chocolate_color(hsv_roi, mask_roi):
    """
    Classifies chocolate variety based on HSV color characteristics.
    
    Args:
        hsv_roi: HSV image of the chocolate region
        mask_roi: Binary mask covering only the chocolate contour
        
    Returns:
        str: 'Dark', 'Milk', 'Ruby', or 'White'
    """
    mean_val = cv2.mean(hsv_roi, mask=mask_roi)
    hue, sat, val = mean_val[0], mean_val[1], mean_val[2]
    
    # White chocolate: very low saturation, high brightness
    if sat < 40 and val > 160:
        return "White"
    # Ruby chocolate: pink / magenta hue range (145 - 175)
    elif 145 <= hue <= 175 and sat >= 50:
        return "Ruby"
    # Dark chocolate: low brightness value, deep brown
    elif val < 85:
        return "Dark"
    # Milk chocolate: medium brown / orange-yellow hue, moderate brightness
    else:
        return "Milk"


def extract_orientation_and_centroid(contour):
    """
    Calculates centroid (u, v) and principal orientation angle theta.
    
    Args:
        contour: External contour from cv2.findContours
        
    Returns:
        tuple: ((cx, cy), angle_degrees, (width, height))
    """
    # Spatial moments for centroid
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0
        
    # Minimum area bounding rectangle for orientation
    rect = cv2.minAreaRect(contour)
    (rx, ry), (width, height), raw_angle = rect
    
    # Normalize angle so it aligns parallel to the major grasp axis
    angle = raw_angle
    if width < height:
        angle = raw_angle + 90.0
        width, height = height, width
        
    # Standardize angle to [-90, +90]
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
        
    return (cx, cy), angle, (width, height)


def evaluate_solidity_defect(contour, threshold=0.88):
    """
    Assesses surface integrity using the contour solidity metric:
    Solidity = Contour Area / Convex Hull Area.
    
    Chipped corners, deep cracks, or broken edges create indentations that
    reduce the contour area relative to its convex hull.
    
    Args:
        contour: External contour of the workpiece
        threshold: Minimum acceptable solidity (default 0.88)
        
    Returns:
        tuple: (solidity_ratio, is_defective)
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return 0.0, True
        
    solidity = float(area) / hull_area
    is_defective = (solidity < threshold)
    return solidity, is_defective


def pixel_to_world_coordinates(cx, cy, img_width=640, img_height=480,
                               table_x_center=0.0, table_y_center=0.4,
                               fov_width_m=0.35, fov_height_m=0.26):
    """
    Maps image pixel coordinates (u, v) to robot world coordinates (X_w, Y_w).
    
    Args:
        cx, cy: Pixel centroid coordinates
        img_width, img_height: Camera resolution
        table_x_center, table_y_center: World coordinates at center of camera FOV
        fov_width_m, fov_height_m: Metric dimensions of camera field of view
        
    Returns:
        tuple: (X_world, Y_world) in meters
    """
    norm_x = (cx - (img_width / 2.0)) / (img_width / 2.0)
    norm_y = ((img_height / 2.0) - cy) / (img_height / 2.0)
    
    world_x = table_x_center + (norm_x * (fov_width_m / 2.0))
    world_y = table_y_center + (norm_y * (fov_height_m / 2.0))
    return round(world_x, 4), round(world_y, 4)


def inspect_chocolate(bgr_image, min_contour_area=250):
    """
    Full perception pipeline for incoming confectionery workpiece.
    
    Args:
        bgr_image: Input BGR frame from overhead camera
        min_contour_area: Minimum pixel area to filter noise
        
    Returns:
        dict: Inspection results and robot motion targets
    """
    if bgr_image is None:
        return {"valid": False, "reason": "No image frame received"}
        
    h, w = bgr_image.shape[:2]
    
    # 1. Preprocessing: Gaussian Blur and Color Conversion
    blurred = cv2.GaussianBlur(bgr_image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 2. Thresholding: Detect any chocolate piece against lighter infeed table
    # Saturated or darker objects against light stainless steel / white belt
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    
    # Clean noise with morphological opening and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
    
    # 3. Contour Extraction
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"valid": False, "reason": "No workpiece contour detected"}
        
    # Largest contour is the chocolate piece
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    if area < min_contour_area:
        return {"valid": False, "reason": f"Contour area {area} below threshold"}
        
    # 4. Pose and Centroid
    (cx, cy), angle, (box_w, box_h) = extract_orientation_and_centroid(largest_contour)
    world_x, world_y = pixel_to_world_coordinates(cx, cy, img_width=w, img_height=h)
    
    # 5. Defect Screening (Solidity)
    solidity, is_defective = evaluate_solidity_defect(largest_contour, threshold=0.88)
    
    # 6. Color Classification
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
    choc_type = classify_chocolate_color(hsv, contour_mask)
    
    return {
        "valid": True,
        "centroid_px": (cx, cy),
        "world_coords_m": (world_x, world_y),
        "orientation_deg": round(angle, 1),
        "chocolate_type": choc_type,
        "solidity": round(solidity, 3),
        "is_defective": is_defective,
        "defect_reason": "Solidity Below Threshold (<0.88)" if is_defective else "None",
        "contour": largest_contour,
        "box_dimensions_px": (round(box_w, 1), round(box_h, 1))
    }


def draw_inspection_overlay(bgr_image, result):
    """
    Draws visual annotations onto frame for monitoring and report figures.
    """
    if not result.get("valid", False):
        return bgr_image
        
    annotated = bgr_image.copy()
    contour = result["contour"]
    cx, cy = result["centroid_px"]
    angle = result["orientation_deg"]
    choc_type = result["chocolate_type"]
    solidity = result["solidity"]
    is_defective = result["is_defective"]
    
    # Color coding: Green for PASS, Red for DEFECT
    color = (0, 0, 255) if is_defective else (0, 255, 0)
    
    # Draw contour and convex hull
    cv2.drawContours(annotated, [contour], -1, color, 2)
    hull = cv2.convexHull(contour)
    cv2.drawContours(annotated, [hull], -1, (255, 200, 0), 1)
    
    # Draw minimum area bounding box
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    cv2.drawContours(annotated, [box], 0, (255, 255, 255), 1)
    
    # Draw orientation axis arrow
    rad = np.radians(angle)
    length = 40
    end_x = int(cx + length * np.cos(rad))
    end_y = int(cy + length * np.sin(rad))
    cv2.arrowedLine(annotated, (cx, cy), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)
    cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
    
    # Display text summary
    status_text = "REJECT: DEFECT" if is_defective else "PASS: QUALITY OK"
    cv2.putText(annotated, f"Type: {choc_type} | Status: {status_text}", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(annotated, f"Angle: {angle:+.1f} deg | Solidity: {solidity:.3f}", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)
    cv2.putText(annotated, f"World Target: X={result['world_coords_m'][0]:.3f}m, Y={result['world_coords_m'][1]:.3f}m", 
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                
    return annotated
