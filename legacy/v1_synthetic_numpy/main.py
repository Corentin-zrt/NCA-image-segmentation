import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nca_segmentation.control_panel.app import ControlPanel


if __name__ == "__main__":
    ControlPanel().run()
