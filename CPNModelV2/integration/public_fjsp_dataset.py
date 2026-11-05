from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterator, Optional, Tuple

from CPNModelV2.core.fjsp_cpn_v4 import FJSPCPNV4
from CPNModelV2.core.fjsp_cpn_v5 import FJSPCPNV5
from CPNModelV2.integration.equivalence_core import CanonicalInstance


@dataclass(frozen=True)
class PublicFJSPRecord:
    """One parsed public FJSP instance and its stable metadata."""

    path: Path
    name: str
    num_jobs: int
    num_machines: int
    num_operations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines,
            "num_operations": self.num_operations,
        }


class PublicFJSPDataset:
    """Deterministic manifest over standard ``.fjs`` instance files."""

    def __init__(
        self,
        directory: Path | str,
        *,
        limit: Optional[int] = None,
        expected_jobs: Optional[int] = None,
        expected_machines: Optional[int] = None,
    ) -> None:
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise FileNotFoundError(f"FJSP dataset directory not found: {self.directory}")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")

        paths = sorted(self.directory.glob("*.fjs"), key=_natural_key)
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            raise ValueError(f"No .fjs files found in {self.directory}")

        records = []
        for path in paths:
            instance = CanonicalInstance.from_fjs(path)
            if expected_jobs is not None and instance.num_jobs != expected_jobs:
                raise ValueError(
                    f"{path.name} has {instance.num_jobs} jobs, expected {expected_jobs}"
                )
            if (
                expected_machines is not None
                and instance.num_machines != expected_machines
            ):
                raise ValueError(
                    f"{path.name} has {instance.num_machines} machines, "
                    f"expected {expected_machines}"
                )
            records.append(PublicFJSPRecord(
                path=path.resolve(),
                name=path.name,
                num_jobs=instance.num_jobs,
                num_machines=instance.num_machines,
                num_operations=instance.total_operations,
            ))
        self.records: Tuple[PublicFJSPRecord, ...] = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[PublicFJSPRecord]:
        return iter(self.records)

    @property
    def paths(self) -> Tuple[Path, ...]:
        return tuple(record.path for record in self.records)

    def manifest(self) -> list[Dict[str, Any]]:
        return [record.to_dict() for record in self.records]


class FJSFileNetFactory:
    """Create a fresh V4 CPN from one public instance for every reset."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        self.canonical = CanonicalInstance.from_fjs(self.path)
        self.instance = self.canonical.to_cpn_instance()

    def __call__(self, seed: Optional[int] = None) -> FJSPCPNV4:
        del seed
        return FJSPCPNV4(self.instance)


class FJSFileV5NetFactory(FJSFileNetFactory):
    """Create a fresh V5 CPN from one public instance for every reset."""

    def __call__(self, seed: Optional[int] = None) -> FJSPCPNV5:
        del seed
        return FJSPCPNV5(self.instance)


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]
