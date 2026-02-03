from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from .notices import (get_gpt_oss_20b_notice, get_gpt_oss_120b_notice,
                      get_smol_3b_notice)


@dataclass
class ModelDetails:
    name: str
    short_name: str
    type: str
    required_memory: int
    hf_id: Optional[str]
    save_path: Optional[Path]
    is_downloaded: bool
    details_path: Optional[Path]
    notice: str

def _make_model(
    model_dir: Path,
    label: str,
    short_name: str,
    hf_id: str,
    subdir: str,
    check_redownload: bool,
    memory: int,
    notice: str
) -> ModelDetails:
    path = Path(model_dir) / f"default/{subdir}"
    path.mkdir(parents=True, exist_ok=True)

    details_path = path / "erictransformer_details.json"
    is_downloaded = details_path.exists()

    prefix = "💾" if (is_downloaded or check_redownload) else "🔗"
    display_name = f"{prefix} {label}: {short_name}"

    return ModelDetails(
        name=display_name,
        short_name=short_name,
        required_memory=memory,
        type="hf",
        hf_id=hf_id,
        save_path=path,
        is_downloaded=is_downloaded,
        details_path=details_path,
        notice=notice
    )


def available_model_factory(model_dir: Path, check_redownload: bool = False) -> Tuple[Dict[str, ModelDetails], str]:
    # Define once; easy to extend with new sizes.
    configs = [

        # We set required_memory to 0 for each model.
        # This is okay because we show recommended memory for each model at the top of the notice.
        # It's up to the user if they want to try to load it with less than the required memory.
        # Also seeing the available models and the approx memory for each might inspire them to try another computer with more memory.
        # required_memory was kept as we might want to use it in the future, especially after we add more models.
        ("3B", "EricFillion/smollm3-3b-mlx", "EricFillion/smollm3-3b-mlx", "ericfillion_smollm3_3b_mlx", 0, get_smol_3b_notice()),
        ("20B", "EricFillion/gpt-oss-20b-mlx", "EricFillion/gpt-oss-20b-mlx", "ericfillion_gpt_oss_20b_mlx", 0 , get_gpt_oss_20b_notice()),
        ("120B", "EricFillion/gpt-oss-120b-mlx", "EricFillion/gpt-oss-120b-mlx", "ericfillion_gpt_oss_120b_mlx", 0, get_gpt_oss_120b_notice()),
    ]

    models = [
        _make_model(model_dir, label, short_name, hf_id, subdir, check_redownload, memory, notice)
        for (label, short_name, hf_id, subdir, memory, notice) in configs
    ]

    # Keep return shape: keys are the user-facing names, value is ModelDetails.
    model_map = {m.name: m for m in models}
    default_name = models[0].name  # The smallest is the default

    return model_map, default_name
