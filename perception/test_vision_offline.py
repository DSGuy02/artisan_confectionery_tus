"""
test_vision_offline.py - Standalone Verification of OpenCV Inspection Pipeline
Allows Partner 2 to test and validate computer vision algorithms offline without Webots.
"""

import os
import sys
import cv2
import numpy as np

# Ensure local imports work regardless of current working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from vision_pipeline import inspect_chocolate, draw_inspection_overlay


def create_synthetic_test_images(output_dir):
    """Generates synthetic test images mimicking chocolates on the infeed table."""
    os.makedirs(output_dir, exist_ok=True)
    images = {}
    
    # 1. Normal Dark Chocolate (Square, unrotated)
    img_dark = np.full((480, 640, 3), 240, dtype=np.uint8)  # Light table surface
    # Dark chocolate: deep brown (B: 20, G: 30, R: 45)
    cv2.rectangle(img_dark, (270, 190), (370, 290), (20, 30, 45), -1)
    images["dark_chocolate_normal.png"] = img_dark
    
    # 2. Rotated Milk Chocolate (Rectangular, rotated by 45 degrees)
    img_milk = np.full((480, 640, 3), 240, dtype=np.uint8)
    # Milk chocolate: medium brown (B: 40, G: 70, R: 120)
    center = (320, 240)
    size = (120, 70)
    angle = 45.0
    rect = (center, size, angle)
    box = np.int64(cv2.boxPoints(rect))
    cv2.drawContours(img_milk, [box], 0, (40, 70, 120), -1)
    images["milk_chocolate_rotated_45deg.png"] = img_milk
    
    # 3. Ruby Chocolate (Pink / Magenta)
    img_ruby = np.full((480, 640, 3), 240, dtype=np.uint8)
    # Ruby chocolate: pink (B: 140, G: 70, R: 190)
    pts = np.array([[320, 180], [380, 280], [260, 280]], np.int32)
    cv2.fillPoly(img_ruby, [pts], (140, 70, 190))
    images["ruby_chocolate_normal.png"] = img_ruby
    
    # 4. Chipped Defective Chocolate (Concave fracture/chip gouge on edge)
    img_chipped = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.rectangle(img_chipped, (270, 190), (370, 290), (20, 30, 45), -1)
    # Carve a concave fracture notch into the side (reduces Area relative to Convex Hull)
    chip_pts = np.array([[370, 200], [315, 240], [370, 280]], np.int32)
    cv2.fillPoly(img_chipped, [chip_pts], (240, 240, 240))  # Match background table
    images["dark_chocolate_chipped_defect.png"] = img_chipped
    
    for filename, img in images.items():
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        
    return images


def run_offline_tests():
    print("==================================================================")
    print(" Running Offline Perception Tests for Artisan Confectionery Cell ")
    print("==================================================================")
    
    samples_dir = os.path.join(CURRENT_DIR, "sample_images")
    results_dir = os.path.join(CURRENT_DIR, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    test_images = create_synthetic_test_images(samples_dir)
    
    print(f"\n{'Test Sample':<35} | {'Type':<8} | {'Angle (deg)':<11} | {'Solidity':<8} | {'Status'}")
    print("-" * 80)
    
    for name, img in test_images.items():
        result = inspect_chocolate(img)
        
        if result["valid"]:
            choc_type = result["chocolate_type"]
            angle = result["orientation_deg"]
            solidity = result["solidity"]
            status = "DEFECT" if result["is_defective"] else "PASS"
            
            print(f"{name:<35} | {choc_type:<8} | {angle:+6.1f} deg  | {solidity:<8.3f} | {status}")
            
            # Save annotated visualization
            annotated = draw_inspection_overlay(img, result)
            out_path = os.path.join(results_dir, f"annotated_{name}")
            cv2.imwrite(out_path, annotated)
        else:
            print(f"{name:<35} | FAILED TO DETECT: {result.get('reason')}")
            
    print("-" * 80)
    print(f"[SUCCESS] All test images evaluated. Annotated results saved to: {results_dir}/\n")


if __name__ == "__main__":
    run_offline_tests()
