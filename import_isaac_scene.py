"""One-time import of the Isaac Sim warehouse used by Isaac Lab (needs ``pip install usd-core``).

python import_isaac_scene.py
python play.py --hall isaac
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin_depot.isaac_scene import main  # noqa: E402

if __name__ == "__main__":
    main()
