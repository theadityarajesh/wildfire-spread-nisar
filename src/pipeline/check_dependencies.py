#!/usr/bin/env python3
"""Quick check for required dependencies for data_pipeline.py"""

import sys

required = {
    "ee": "earthengine-api",
    "geemap": "geemap",
    "pygridmet": "pygridmet",
    "asf_search": "asf-search",
    "dotenv": "python-dotenv",
    "xarray": "xarray",
    "rioxarray": "rioxarray",
}

print("Checking dependencies for data_pipeline.py...\n")

missing = []
for module, package in required.items():
    try:
        __import__(module)
        print(f"✓ {module:15s} ({package})")
    except ImportError:
        print(f"✗ {module:15s} ({package}) - MISSING")
        missing.append(package)

if missing:
    print(f"\n⚠️  Missing {len(missing)} package(s). Install with:")
    print(f"   pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("\n✓ All dependencies installed!")
    sys.exit(0)
