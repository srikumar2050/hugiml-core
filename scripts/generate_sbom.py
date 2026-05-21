#!/usr/bin/env python3
# Copyright 2026 Srikumar Krishnamoorthy
# Apache-2.0 License
"""Generate a CycloneDX-lite SBOM for the installed hugiml-core package.

Usage::

    python scripts/generate_sbom.py [--output sbom.json]
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hugiml.serialization import generate_sbom  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hugiml-core SBOM")
    parser.add_argument(
        "--output",
        default="sbom-hugiml.json",
        help="Output path for the SBOM JSON (default: sbom-hugiml.json)",
    )
    args = parser.parse_args()
    sbom = generate_sbom(output_path=args.output)
    print(f"SBOM written to {args.output}")
    print(f"  format:     {sbom.get('bomFormat')}")
    print(f"  components: {len(sbom.get('components', []))}")
    component = sbom.get("metadata", {}).get("component", {})
    print(f"  package:    {component.get('name')} {component.get('version')}")
    print(json.dumps(sbom, indent=2))


if __name__ == "__main__":
    main()
