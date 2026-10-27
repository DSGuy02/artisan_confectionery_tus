"""
Perception package for the Artisan Confectionery Robotic Cell.
Provides OpenCV computer vision pipelines for workpiece segmentation,
color classification, orientation estimation, and surface defect screening.
"""
from .vision_pipeline import (
    convert_webots_to_cv,
    classify_chocolate_color,
    extract_orientation_and_centroid,
    evaluate_solidity_defect,
    inspect_chocolate,
    draw_inspection_overlay
)

__all__ = [
    'convert_webots_to_cv',
    'classify_chocolate_color',
    'extract_orientation_and_centroid',
    'evaluate_solidity_defect',
    'inspect_chocolate',
    'draw_inspection_overlay'
]
