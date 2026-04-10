"""
Image quality constants for the visual search pipeline.
"""

# === FILE & DIMENSION LIMITS ===
MAX_FILE_SIZE_MB = 10
MIN_DIMENSION_UNIVERSAL = 300
MAX_DIMENSION = 4096

# === CONTRAST THRESHOLD ===
# Hard rejection for blank / uniform images (std < 1.0)
CONTRAST_THRESHOLDS = {
    "reject_min": 1.0,
    "warn_min": 10.0,
}
