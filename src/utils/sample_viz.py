from __future__ import annotations

from typing import Any


def build_ase_view(sample_vis_count: int) -> Any | None:
    if int(sample_vis_count) <= 0:
        return None
    try:
        from src.utils.ase_notebook import AseView

        return AseView(
            rotations="45x,45y,45z",
            canvas_size=(400, 400),
            show_bonds=False,
            atom_show_label=True,
        )
    except Exception as exc:
        print(f"[sample-vis] Could not initialize AseView; image logging disabled ({exc}).")
        return None


__all__ = ["build_ase_view"]
