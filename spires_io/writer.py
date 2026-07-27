"""Reserved SPIReS output writer API."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spires_contract import SpiresData


@dataclass(frozen=True)
class SpiresDataWriter:
    """Deferred writer for SPIReS inversion outputs."""

    data: SpiresData
    output_path: Path | None = None
    output_policy: Mapping[str, Any] = field(default_factory=dict)
    scene_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_path is not None:
            object.__setattr__(self, "output_path", Path(self.output_path))
        object.__setattr__(self, "output_policy", dict(self.output_policy))
        object.__setattr__(self, "scene_metadata", dict(self.scene_metadata))

    @classmethod
    def from_data(
        cls,
        data: SpiresData,
        *,
        output_path: str | Path | None = None,
        output_policy: Mapping[str, Any] | None = None,
        scene_metadata: Mapping[str, Any] | None = None,
    ) -> "SpiresDataWriter":
        """Create a writer bound to prepared data and future output policy."""
        return cls(
            data=data,
            output_path=output_path,
            output_policy={} if output_policy is None else output_policy,
            scene_metadata={} if scene_metadata is None else scene_metadata,
        )

    def write(self, inversion_results: Any, **kwargs: Any) -> None:
        """Write inversion results using the stored scene context."""
        raise NotImplementedError(
            "SpiresDataWriter.write is reserved for a future output implementation"
        )
