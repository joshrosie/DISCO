import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from pymatgen.io.cif import CifWriter
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element

NMAX = 20  # default max atoms for MP20; override per dataset via MP20Tokens(nmax=...)
VZ = 94  # elements are 1..94, 0 is NULL padding but there are not 94 elements in MP20!!!


def lattice_to_Y(lengths, angles_deg):
    """
    Encode lattice parameters to Y representation.
    """
    a, b, c = lengths
    alpha, beta, gamma = angles_deg
    return np.array(
        [
            np.log(a),
            np.log(b),
            np.log(c),
            np.cos(np.deg2rad(alpha)),
            np.cos(np.deg2rad(beta)),
            np.cos(np.deg2rad(gamma)),
        ],
        dtype=np.float32,
    )


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def decode_Y1(y1):
    y1 = _to_numpy(y1)
    lengths = np.exp(y1[:3])
    angles = np.degrees(np.arccos(np.clip(y1[3:], -1.0, 1.0)))
    return lengths.astype(np.float32), angles.astype(np.float32)


def tokens_to_eval_dict(item: dict, sample_idx: int = 0) -> dict:
    mask = ~_to_numpy(item["pad_mask"]).astype(bool)
    atom_types = _to_numpy(item["A0"])[mask]
    frac_coords = _to_numpy(item["F1"])[mask]
    lengths, angles = decode_Y1(item["Y1"])
    return {
        "frac_coords": frac_coords,
        "atom_types": atom_types,
        "lengths": lengths,
        "angles": angles,
        "sample_idx": sample_idx,
    }


def tokens_to_structure(item: dict):

    mask = ~_to_numpy(item["pad_mask"]).astype(bool)
    atom_types = _to_numpy(item["A0"])[mask]
    if atom_types.size == 0:
        raise ValueError("Cannot build a structure from a sample with zero atoms.")
    if (atom_types <= 0).any(): # invalid atomic numbers - PAD is 0
        raise ValueError("Found non-positive atomic number in unpadded positions.")
    species = [Element.from_Z(int(z)) for z in atom_types]
    frac_coords = _to_numpy(item["F1"])[mask]
    lengths, angles = decode_Y1(item["Y1"])
    lattice = Lattice.from_parameters(*(lengths.tolist() + angles.tolist()))
    return Structure(
        lattice=lattice,
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False,
    )


def tokens_batch_to_structures(batch: dict):

    structs = []
    batch_size = _to_numpy(batch["A0"]).shape[0]
    for i in range(batch_size):
        item = {
            "A0": batch["A0"][i],
            "F1": batch["F1"][i],
            "Y1": batch["Y1"][i],
            "pad_mask": batch["pad_mask"][i],
        }
        mask = ~_to_numpy(item["pad_mask"]).astype(bool)
        atom_types = _to_numpy(item["A0"])[mask]
        if atom_types.size == 0:
            raise ValueError("Cannot build a structure from a sample with zero atoms.")
        if (atom_types <= 0).any(): # invalid atomic numbers - PAD is 0
            raise ValueError("Found non-positive atomic number in unpadded positions.")
        species = [Element.from_Z(int(z)) for z in atom_types]
        frac_coords = _to_numpy(item["F1"])[mask]
        lengths, angles = decode_Y1(item["Y1"])
        lattice = Lattice.from_parameters(*(lengths.tolist() + angles.tolist()))
        structs.append(
            Structure(
                lattice=lattice,
                species=species,
                coords=frac_coords,
                coords_are_cartesian=False,
            )
        )
    return structs


def tokens_batch_to_cif_strings(batch: dict) -> list[str]:
    from pymatgen.io.cif import CifWriter

    return [str(CifWriter(struct)) for struct in tokens_batch_to_structures(batch)]


def tokens_to_cif_string(item: dict) -> str:

    struct = tokens_to_structure(item)
    return str(CifWriter(struct))


def translate_frac_coords(frac_coords, pad_mask, rng=None):
    coords = _to_numpy(frac_coords).copy()
    mask = ~_to_numpy(pad_mask).astype(bool)
    if mask.any():
        if rng is None:
            # Use NumPy's global RNG so caller-level seeding can control augmentation.
            delta = np.random.random(3).astype(np.float32)
        else:
            try:
                delta = rng.random(3, dtype=np.float32)
            except TypeError:
                delta = np.asarray(rng.random(3), dtype=np.float32)
        coords[mask] = (coords[mask] + delta) % 1.0
    return coords


def collate_mp20_tokens(batch):
    """Simple collate function for MP20Tokens dataset."""
    out = {
        "mp_id": [b["mp_id"] for b in batch],
        "A0": torch.stack([b["A0"] for b in batch], dim=0),  # (B,NMAX)
        "F1": torch.stack([b["F1"] for b in batch], dim=0),  # (B,NMAX,3)
        "Y1": torch.stack([b["Y1"] for b in batch], dim=0),  # (B,6)
        "pad_mask": torch.stack([b["pad_mask"] for b in batch], dim=0),  # (B,NMAX)
        "num_atoms": torch.tensor([b["num_atoms"] for b in batch], dtype=torch.long),
    }
    return out


CHEMISTRY_NUM_ELEMENTS = 103  # Z=1..103 covers every MP20 element with headroom.


class MP20TokensWithPropsCollator:
    """Picklable collator that adds property tensors and missingness masks.

    Scalar / categorical props (formation_energy_per_atom, band_gap, spacegroup,
    ...) are read from per-sample dict keys and stacked into (B,) tensors.

    ``chemistry`` is a special case: the target is a multi-hot vector over
    elements, derived directly from the sample's A0 (atomic numbers). No new
    cache field is needed — we just read the atoms the model is already given
    and convert their unique Z values to a (num_elements,) bool.
    """

    def __init__(self, prop_keys: list[str] | None = None):
        self.keys = list(prop_keys or [])

    def __call__(self, batch):
        out = collate_mp20_tokens(batch)
        if not self.keys:
            return out
        props: dict[str, torch.Tensor] = {}
        missing: dict[str, torch.Tensor] = {}
        for k in self.keys:
            if k == "chemistry":
                # Multi-hot over elements, derived from each sample's A0.
                multi_hots = []
                miss_chem = []
                for b in batch:
                    a0 = b["A0"]
                    # Non-zero entries of A0 are the actual atomic numbers.
                    zs = a0[a0 > 0].tolist() if hasattr(a0, "tolist") else [
                        int(x) for x in a0 if int(x) > 0
                    ]
                    mh = torch.zeros(CHEMISTRY_NUM_ELEMENTS, dtype=torch.float32)
                    if zs:
                        for z in zs:
                            if 1 <= z <= CHEMISTRY_NUM_ELEMENTS:
                                mh[z - 1] = 1.0
                        multi_hots.append(mh)
                        miss_chem.append(False)
                    else:
                        multi_hots.append(mh)
                        miss_chem.append(True)
                props[k] = torch.stack(multi_hots, dim=0)  # (B, num_elements)
                missing[k] = torch.tensor(miss_chem, dtype=torch.bool)
                continue

            vals: list[float] = []
            miss: list[bool] = []
            all_whole = True
            for b in batch:
                # Tolerate legacy `prop__<k>` cache keys alongside the raw column name.
                v = b.get(k, b.get(f"prop__{k}"))
                if v is None:
                    vals.append(0.0)
                    miss.append(True)
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    vals.append(0.0)
                    miss.append(True)
                    continue
                if math.isnan(fv):
                    vals.append(0.0)
                    miss.append(True)
                    continue
                vals.append(fv)
                miss.append(False)
                if not float(fv).is_integer():
                    all_whole = False
            # Treat purely-integer columns (e.g. spacegroup) as Long; otherwise Float.
            if all_whole and any(not m for m in miss):
                props[k] = torch.tensor([int(v) for v in vals], dtype=torch.long)
            else:
                props[k] = torch.tensor(vals, dtype=torch.float32)
            missing[k] = torch.tensor(miss, dtype=torch.bool)
        out["props"] = props
        out["prop_missing"] = missing
        return out


def make_collate_with_props(prop_keys: list[str] | None = None):
    """Factory for a collate fn that also stacks property values + missingness masks.

    For each key in `prop_keys`, the returned batch contains:
      - `props[key]`: (B,) float32 for scalar or long for categorical (auto-inferred per value type)
      - `prop_missing[key]`: (B,) bool, True when the value is absent or NaN for that sample

    Samples with missing properties get a placeholder value (0 for ints, 0.0 for floats);
    training code must use `prop_missing` to force unconditional for those samples.
    """
    return MP20TokensWithPropsCollator(prop_keys)


def _items_have_requested_props(items: list[dict], prop_list: list[str]) -> bool:
    if not prop_list:
        return True
    if not items:
        return True
    for key in prop_list:
        # chemistry is derived from A0 on the fly; always "available" regardless
        # of whether a cache column exists.
        if key == "chemistry":
            continue
        if not any((key in item) or (f"prop__{key}" in item) for item in items):
            return False
    return True


class MP20Tokens(Dataset):
    """
    Plain PyTorch dataset returning dicts with:
      A0: (NMAX,) long
      F1: (NMAX,3) float
      Y1: (6,) float
      pad_mask: (NMAX,) bool   True where padded
    """

    def __init__(
        self,
        root: str,
        force_reprocess: bool = False,
        augment_translate: bool = False,
        prop_list: list[str] | None = None,
        split: str = "all",
        nmax: int = NMAX,
        use_space_group: bool = False,
    ):
        self.root = root
        self.augment_translate = augment_translate
        self.prop_list = prop_list or []
        self.split = split
        self.nmax = int(nmax)
        # Space group is a derived field (computed by preprocessing via spglib), not a
        # CSV column, so we auto-enable symmetry analysis whenever it's requested.
        self.use_space_group = bool(use_space_group) or ("spacegroup" in self.prop_list)
        if self.nmax <= 0:
            raise ValueError(f"nmax must be > 0, got {self.nmax}.")
        os.makedirs(root, exist_ok=True)

        split_name = "all" if split is None else split
        self.split_name = split_name
        sg_suffix = "_sg" if self.use_space_group else ""
        nmax_suffix = "" if self.nmax == NMAX else f"_nmax{self.nmax}"
        self.cache_suffix = f"{nmax_suffix}{sg_suffix}"
        self.raw_csv = os.path.join(root, "raw", f"{split_name}.csv")
        self.proc_pt = os.path.join(
            root, "processed", f"mp20_tokens_{split_name}{self.cache_suffix}.pt"
        )
        os.makedirs(os.path.dirname(self.raw_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.proc_pt), exist_ok=True)

        cached_items = None
        should_process = force_reprocess or (not os.path.exists(self.proc_pt))
        if not should_process:
            cached_items = torch.load(self.proc_pt, weights_only=False)
            if not _items_have_requested_props(cached_items, self.prop_list):
                missing = [
                    k
                    for k in self.prop_list
                    if not any(
                        (k in item) or (f"prop__{k}" in item)
                        for item in cached_items
                    )
                ]
                print(
                    "[data] processed token cache is missing requested properties "
                    f"{missing}; rebuilding {self.proc_pt}"
                )
                should_process = True
                cached_items = None

        if should_process:
            self._download_if_needed()
            self._process_and_save()
            cached_items = None

        # MP20 token cache is a trusted local artifact; disable weights_only to allow full pickle.
        self.items = (
            cached_items
            if cached_items is not None
            else torch.load(self.proc_pt, weights_only=False)
        )

    def _download_if_needed(self):
        if os.path.exists(self.raw_csv):
            return
        from huggingface_hub import hf_hub_download

        if self.split not in ("all", None):
            raise FileNotFoundError(
                f"Missing split CSV: {self.raw_csv}. Provide raw/{self.split}.csv."
            )

        hf_hub_download(
            repo_id="chaitjo/MP20_ADiT",
            filename="raw/all.csv",
            repo_type="dataset",
            local_dir=self.root,
        )

    def _process_and_save(self):
        # Important: if you previously cached raw/all.pt with crystalnn,
        # delete it to ensure it reprocesses correctly.
        raw_cache = os.path.join(
            self.root, "raw", f"{self.split_name}_tokens{self.cache_suffix}.pt"
        )
        cached_data = None
        if os.path.exists(raw_cache):
            cached_data = torch.load(raw_cache, weights_only=False)
            if self.prop_list:
                sample = cached_data[0] if cached_data else {}
                missing_props = [k for k in self.prop_list if k not in sample]
                if missing_props:
                    cached_data = None
        if cached_data is None:
            from src.data.preprocessing_utils import preprocess

            cached_data = preprocess(
                self.raw_csv,
                niggli=True,
                primitive=False,
                graph_method="none",  # <-- key: no graphs
                prop_list=self.prop_list,
                use_space_group=self.use_space_group,
                # tol=0.1,
                num_workers=32,
            )
            torch.save(cached_data, raw_cache)

        items = []
        initial_total = len(cached_data)
        for d in cached_data:
            ga = d["graph_arrays"]
            atom_types = np.array(ga["atom_types"], dtype=np.int64)  # atomic numbers
            frac_coords = np.array(ga["frac_coords"], dtype=np.float32)
            lengths = np.array(ga["lengths"], dtype=np.float32)
            angles = np.array(ga["angles"], dtype=np.float32)
            n = int(ga["num_atoms"])

            # Strict filtering to match nmax.
            if n > self.nmax:
                continue
            if atom_types.max(initial=0) > VZ:
                continue

            # Build padded tensors
            A0 = np.zeros((self.nmax,), dtype=np.int64)
            A0[:n] = atom_types

            F1 = np.zeros((self.nmax, 3), dtype=np.float32)
            F1[:n] = frac_coords % 1.0

            Y1 = lattice_to_Y(lengths, angles)

            pad_mask = np.ones((self.nmax,), dtype=np.bool_)
            pad_mask[:n] = False

            items.append(
                {
                    "mp_id": d.get("mp_id", d.get("material_id", None)),
                    "A0": torch.from_numpy(A0),
                    "F1": torch.from_numpy(F1),
                    "Y1": torch.from_numpy(Y1),
                    "pad_mask": torch.from_numpy(pad_mask),
                    "num_atoms": n,
                    # keep cif for debugging/visualization:
                    # "cif": d.get("cif", None),
                }
            )
            if self.prop_list:
                for k in self.prop_list:
                    if k in d:
                        items[-1][k] = d[k]

        torch.save(items, self.proc_pt)
        self.items = items
        final_retention = len(items)

        # Write human-readable stats about filtering.
        filtered_out = initial_total - final_retention
        filtered_pct = (filtered_out / initial_total * 100.0) if initial_total else 0.0
        info_path = os.path.join(
            self.root,
            "processed",
            f"mp20_tokens_{self.split_name}{self.cache_suffix}_info.txt",
        )
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(
                "MP20Tokens preprocessing summary\n"
                f"split: {self.split_name}\n"
                f"total_raw: {initial_total}\n"
                f"kept_after_filters: {final_retention}\n"
                f"filtered_out: {filtered_out}\n"
                f"filtered_pct: {filtered_pct:.2f}%\n"
                "filters:\n"
                f" - num_atoms <= NMAX ({self.nmax})\n"
                f" - max_atomic_number <= VZ ({VZ})\n"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if not self.augment_translate:
            return item

        out = dict(item)
        if self.augment_translate:
            out["F1"] = torch.from_numpy(
                translate_frac_coords(out["F1"], out["pad_mask"])
            )
        return out
