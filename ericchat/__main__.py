import os

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from ericchat.app import run

if __name__ == "__main__":
    run()
