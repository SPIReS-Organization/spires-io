"""Lightweight structured logging helpers for SPIRES workflows."""

from datetime import datetime
import json
import logging
from pathlib import Path
import threading
from typing import Any


_LOG_FIELD_PRIORITY = (
    "title",
    "log_path",
    "manifest_path",
    "event",
    "stage",
    "event_type",
    "status",
    "platform",
    "tile",
    "aggregate_log_path",
    "date",
    "output_dataset_path",
    "r0_year",
    "retry_count",
    "sensor",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "slurm_cluster_name",
    "slurm_job_id",
    "slurm_job_name",
    "slurm_submit_dir",
    "task_index",
    "water_year",
    "scene_name",
    "input_path",
    "source_type",
    "product",
    "scenes_requested",
    "scenes_prepared",
    "time_count",
    "requested_time_coverage_start",
    "requested_time_coverage_end",
    "time_coverage_start",
    "time_coverage_end",
    "selected_bands",
    "bands_500m",
    "bands_1km",
    "band_selection_source",
    "lut_name",
    "lut_file",
    "output_shape",
    "elapsed_seconds",
)

_VISUAL_LOG_PREFIX_BY_EVENT_TYPE = {
    "start": "====== START ======",
    "summary": "====== SUMMARY ======",
    "submission": "====== SUBMISSION ======",
}
_FIELD_RE = __import__("re").compile(r'([A-Za-z0-9_]+)=(".*?"|\{.*?\}|\[.*?\]|[^ ]+)')


def _serialize_log_value(value: Any) -> str:
    """Serialize a log field into a stable plain-text representation."""
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return json.dumps(",".join(value))
    return json.dumps(value, sort_keys=True)


def format_log_event(event: str, **fields: Any) -> str:
    """Format a structured log event as a single plain-text line."""
    if fields.get("event_type") == "context":
        excluded_keys = {"title", "event", "event_type", "scope", "stage", "status"}
        ordered_keys = [key for key in _LOG_FIELD_PRIORITY if key in fields and key not in excluded_keys]
        ordered_keys.extend(
            sorted(key for key in fields if key not in ordered_keys and key not in excluded_keys)
        )
        title = str(fields.get("title", "CONTEXT"))
        body = [title]
        for key in ordered_keys:
            body.append(f"{key}={_serialize_log_value(fields[key])}")
        return "\n".join(body)

    ordered_keys = [key for key in _LOG_FIELD_PRIORITY if key in fields]
    ordered_keys.extend(sorted(key for key in fields if key not in _LOG_FIELD_PRIORITY))
    parts = []
    visual_prefix = _VISUAL_LOG_PREFIX_BY_EVENT_TYPE.get(fields.get("event_type")) if bool(fields.get("scope")) else None
    if visual_prefix:
        parts.append(visual_prefix)
    parts.append(f"event={_serialize_log_value(event)}")
    for key in ordered_keys:
        parts.append(f"{key}={_serialize_log_value(fields[key])}")
    return " ".join(parts)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured event on the provided logger."""
    logger.log(
        level,
        format_log_event(event, **fields),
        extra={
            "_spires_event": event,
            "_spires_event_type": fields.get("event_type"),
            "_spires_parent_event": fields.get("parent_event"),
            "_spires_scope": bool(fields.get("scope")),
        },
    )


class _SPIRESLogFormatter(logging.Formatter):
    """Formatter that adds a bare separator line before highlighted records."""

    _lock = threading.Lock()
    _depth_by_logger: dict[str, int] = {}
    _scope_stack_by_logger: dict[str, list[str]] = {}

    @staticmethod
    def _parse_structured_message(message: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, raw_value in _FIELD_RE.findall(message):
            try:
                result[key] = json.loads(raw_value)
            except json.JSONDecodeError:
                result[key] = raw_value
        return result

    def format(self, record: logging.LogRecord) -> str:
        logger_key = record.name
        event = getattr(record, "_spires_event", None)
        event_type = getattr(record, "_spires_event_type", None)
        parent_event = getattr(record, "_spires_parent_event", None)
        scope = bool(getattr(record, "_spires_scope", False))
        if event is None or event_type is None:
            parsed = self._parse_structured_message(record.getMessage())
            event = parsed.get("event")
            event_type = parsed.get("event_type")
            parent_event = parsed.get("parent_event")
            scope = bool(parsed.get("scope", False))

        with self._lock:
            depth = self._depth_by_logger.get(logger_key, 0)
            stack = self._scope_stack_by_logger.get(logger_key, [])

            if parent_event:
                try:
                    parent_depth = stack.index(str(parent_event)) + 1
                    depth = max(depth, parent_depth)
                except ValueError:
                    pass

            indent = "    " * max(depth - 1, 0)
            record.msg = record.getMessage()
            record.args = ()
            formatted = super().format(record)
            formatted = "\n".join(f"{indent}{line}" for line in formatted.splitlines())

            if scope and event_type == "start" and event is not None:
                stack.append(str(event))
                self._scope_stack_by_logger[logger_key] = stack
                self._depth_by_logger[logger_key] = depth + 1
            elif scope and event_type in {"summary", "submission"} and event is not None:
                if stack:
                    try:
                        idx = len(stack) - 1 - stack[::-1].index(str(event))
                        stack = stack[:idx]
                    except ValueError:
                        stack = stack[:-1]
                self._scope_stack_by_logger[logger_key] = stack
                self._depth_by_logger[logger_key] = len(stack)
            else:
                self._scope_stack_by_logger[logger_key] = stack
                self._depth_by_logger[logger_key] = depth

        visual_prefix = next(
            (prefix for prefix in _VISUAL_LOG_PREFIX_BY_EVENT_TYPE.values() if record.getMessage().lstrip().startswith(prefix)),
            None,
        )
        if visual_prefix is None:
            if event_type == "context":
                message_lines = record.getMessage().splitlines()
                if not message_lines:
                    return formatted
                title = message_lines[0].strip()
                body = message_lines[1:]
                separator = "=" * max(len(title), 21)
                indented_body = "\n".join(body)
                if indented_body:
                    return "\n".join(
                        [
                            self.formatTime(record, self.datefmt),
                            separator,
                            title,
                            separator,
                            indented_body,
                            separator,
                            separator,
                        ]
                    )
                return "\n".join(
                    [
                        self.formatTime(record, self.datefmt),
                        separator,
                        title,
                        separator,
                        separator,
                        separator,
                    ]
                )
            return formatted

        if event_type == "submission":
            return formatted

        timestamp = self.formatTime(record, self.datefmt)
        prefix_width = len(f"{timestamp} {record.levelname} {record.name}")
        separator = "=" * prefix_width
        if event_type == "start":
            return f"{separator}\n{formatted}"
        if event_type == "summary":
            return f"{formatted}\n{separator}"
        return formatted


class _SPIRESPlainIndentedFormatter(_SPIRESLogFormatter):
    """Same contextual indentation behavior without visual separator lines."""

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record)


def configure_spires_file_logger(
    log_path: str | Path,
    *,
    logger_name: str = "spires",
    level: int = logging.INFO,
    log_to_stdout: bool = True,
    mode: str = "w",
    aggregate_log_path: str | Path | None = None,
    enable_context_indentation: bool = True,
    aggregate_show_separators: bool = True,
) -> logging.Logger:
    """
    Configure a plain-text SPIRES logger suitable for `.log` or `.txt` files.

    The logger writes timestamped lines to ``log_path`` and can optionally also
    stream the same messages to stdout so Slurm captures them.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    resolved_log_path = Path(log_path).expanduser().resolve()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    if enable_context_indentation:
        file_formatter: logging.Formatter = _SPIRESPlainIndentedFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        aggregate_formatter: logging.Formatter
        if aggregate_show_separators:
            aggregate_formatter = _SPIRESLogFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        else:
            aggregate_formatter = _SPIRESPlainIndentedFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    else:
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        aggregate_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    logger.handlers.clear()

    file_handler = logging.FileHandler(resolved_log_path, mode=mode)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if aggregate_log_path is not None:
        resolved_aggregate_path = Path(aggregate_log_path).expanduser().resolve()
        resolved_aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_handler = logging.FileHandler(resolved_aggregate_path, mode="a")
        aggregate_handler.setLevel(level)
        aggregate_handler.setFormatter(aggregate_formatter)
        logger.addHandler(aggregate_handler)

    if log_to_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(file_formatter)
        logger.addHandler(stream_handler)

    return logger


def make_spires_log_path(
    log_dir: str | Path,
    *,
    prefix: str = "spires",
    tile: str | None = None,
    sensor: str | None = None,
    label: str | None = None,
    extension: str = ".log",
    timestamp: str | None = None,
) -> Path:
    """Create a timestamped per-run log path for notebook or batch workflows."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [prefix]
    if sensor:
        parts.append(sensor)
    if tile:
        parts.append(tile)
    if label:
        parts.append(label)
    parts.append(timestamp)

    log_dir = Path(log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / ("_".join(parts) + extension)


def remove_empty_log_file(path: str | Path) -> bool:
    """Delete `path` if it exists and has zero size."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        return False
    if resolved.stat().st_size != 0:
        return False
    resolved.unlink()
    return True
