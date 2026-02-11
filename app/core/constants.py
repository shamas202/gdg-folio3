"""
Configuration constants for image search pipeline.

All quality thresholds, category definitions, and validation rules
are centralized here for easy tuning and maintenance.
"""

# === 3-TIER CATEGORY SYSTEM (46 total) ===

# TIER 1: SMALL - Tiny items (5-20cm), close-up shots, detail critical
SMALL_CATEGORIES = [
    # Tableware
    "cup",
    "plate",
    # Small Kitchen Appliances & Cookware
    "cooking-pot",
    "coffee-maker",
    "cooking-appliance",
    "food-processor",
    # Very Small Decorative Items
    "candle",
    "vase",
    "flower",
    "statue-and-antique",
]

# TIER 2: MEDIUM - Mid-size items (20-120cm), standard product shots
MEDIUM_CATEGORIES = [
    # Single-Person Seating
    "chair",
    "office-chair",
    # Small/Medium Tables
    "side-table",
    "console",
    "center-table",
    "service-table",
    "tv-table",
    # Storage & Display
    "storage-box",
    "shelve",
    # Small Bedroom Items
    "pillow",
    # ALL Lighting Fixtures (tall but narrow)
    "lighting",
    "lampshade",
    "wall-lighting",
    "outdoor-lighting",
    "chandelier",
    "pendant-lighting",
    "floor-stand",
    # Decorative Items
    "decorative-hanger",
    "wall-clock",
    "art-canvas",
    "laundry-basket",
    "serving-utensil-and-tray",
    "flower-pot-and-plant",
]

# TIER 3: LARGE - Large furniture (150-300cm), lifestyle shots, context helps
LARGE_CATEGORIES = [
    # Multi-Person Seating
    "3-seater-sofa",
    "2-seater-sofa",
    "l-shape-sofa",
    "sofa",
    "chaise-lounge",
    # Bedroom Furniture & Textiles
    "bed",
    "bedspread",
    "mattresses",
    "comforter",
    # Work & Bedroom Tables
    "dressing-table",
    "office-table",
    "dining-table",
    # Large Floor Items
    "carpet",
    "wardrobe",
]

# === IMAGE QUALITY THRESHOLDS ===

# File & Dimension Limits
MAX_FILE_SIZE_MB = 15
MIN_DIMENSION_UNIVERSAL = 500  # Both width AND height for ALL categories
MAX_DIMENSION = 4096  # Auto-resize if exceeded

# Blur Detection (Laplacian Variance) - 3-TIER SYSTEM
BLUR_THRESHOLDS = {
    "small": {
        "reject": 30,  # Small items - moderate (was 40)
        "warn": 70,
    },
    "medium": {
        "reject": 25,  # Medium items - lenient
        "warn": 65,
    },
    "large": {
        "reject": 20,  # Large items - very lenient (textiles)
        "warn": 60,
    },
    "default": {
        "reject": 25,  # Universal threshold for retrieval (when category unknown)
        "warn": 65,
    },
}


# Contrast (Standard Deviation) - UNIVERSAL
CONTRAST_THRESHOLDS = {
    "reject_min": 1.0,  # Blank/uniform image - hard rejection
    "warn_min": 10.0,  # Low contrast - warning
}

# === STAGE 2: ROA (Ratio of Area) THRESHOLDS - 3-TIER SYSTEM ===
# NOTE: ROA validation only for INGESTION, NOT for retrieval
# Retrieval skips ROA to be user-friendly (room photo framing varies)

ROA_THRESHOLDS = {
    "small": {
        "min": 0.06,  # 6% minimum - Small objects (cups, plates, candles)
        # Very lenient: tiny items can be distant in frame
        # 1000×1000 image needs ≥245×245px object (6%)
        # Applied: INGESTION only
    },
    "medium": {
        "min": 0.08,  # 8% minimum - Medium objects (chairs, lamps, small tables)
        # Lenient: allows reasonable distance for mid-size items
        # 1000×1000 image needs ≥283×283px object (8%)
        # Applied: INGESTION only
    },
    "large": {
        "min": 0.12,  # 12% minimum - Large objects (sofas, beds, dining tables)
        # Balanced: allows lifestyle shots with room context
        # 1000×1000 image needs ≥346×346px object (12%)
        # Applied: INGESTION only
    },
}

# === CROP GENERATION SETTINGS - 3-TIER SYSTEM ===

# Medium crop padding (fixed ratio)
MEDIUM_PADDING_RATIO = 0.15  # 15% padding around object

# Augmentation settings
ROTATION_ANGLES = [-5, 5]  # Subtle tilts in degrees (realistic camera angles)
ENABLE_HORIZONTAL_FLIP = True  # Mirror images (left/right viewpoint)
ENABLE_VERTICAL_FLIP = False  # Products upside-down is unnatural

# Crop strategy per tier
CROP_STRATEGY = {
    "small": ["tight"],                  # Small: tight only (4 crops)
    "medium": ["tight"],                 # Medium: tight only (4 crops)
    "large": ["tight", "medium", "full"],  # Large: all crops (9 crops)
}
