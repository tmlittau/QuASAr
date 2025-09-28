"""Helpers for selecting circuits for the theoretical estimation CLI."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, Mapping, Sequence

from quasar.circuit import Circuit

from . import circuits as circuit_lib
from . import large_scale_circuits as large_circuit_lib
from . import paper_figures
from . import showcase_benchmarks


@dataclass(frozen=True)
class BuilderInfo:
    """Metadata describing a circuit builder available for estimation."""

    canonical: str
    builder: Callable[..., Circuit]
    first_param_name: str
    required_params: tuple[str, ...]
    optional_params: tuple[str, ...]
    accepts_kwargs: bool


def _format_signature(info: BuilderInfo) -> str:
    params: list[str] = []
    if info.required_params:
        params.append("required=" + ", ".join(info.required_params))
    if info.optional_params:
        params.append("optional=" + ", ".join(info.optional_params))
    if not params:
        params.append("no extra parameters")
    return "; ".join(params)


def _collect_builders() -> tuple[dict[str, BuilderInfo], dict[str, str | None]]:
    modules: Mapping[str, object] = {
        "circuits": circuit_lib,
        "large_scale_circuits": large_circuit_lib,
    }
    builders: dict[str, BuilderInfo] = {}
    aliases: dict[str, str | None] = {}

    for label, module in modules.items():
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_"):
                continue
            if obj.__module__ != module.__name__:
                continue

            signature = inspect.signature(obj)
            params = list(signature.parameters.values())
            if not params:
                continue

            first_param = None
            for candidate in params:
                if candidate.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    first_param = candidate
                    break
            if first_param is None:
                continue

            identifier = first_param.name.lower()
            if not any(
                token in identifier
                for token in ("qubit", "bit", "width", "size", "distance", "chain")
            ):
                continue

            accepts_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in params
            )

            required: list[str] = []
            optional: list[str] = []
            for param in params:
                if param is first_param:
                    continue
                if param.kind in {
                    inspect.Parameter.VAR_POSITIONAL,
                }:
                    continue
                if param.default is inspect._empty and param.kind != inspect.Parameter.VAR_KEYWORD:
                    required.append(param.name)
                elif param.kind != inspect.Parameter.VAR_KEYWORD:
                    optional.append(param.name)

            if first_param.kind == inspect.Parameter.KEYWORD_ONLY:
                first_name = first_param.name

                def _make_wrapper(func: Callable[..., Circuit], param_name: str):
                    @wraps(func)
                    def wrapper(n_qubits: int, **kwargs):
                        if param_name in kwargs and kwargs[param_name] != n_qubits:
                            raise ValueError(
                                "First parameter %r conflicts with supplied qubit width"
                                % param_name
                            )
                        kwargs[param_name] = n_qubits
                        return func(**kwargs)

                    return wrapper

                builder_callable = _make_wrapper(obj, first_name)
            elif first_param.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }:
                builder_callable = obj
            else:
                continue

            canonical = f"{label}.{name}"
            builders[canonical] = BuilderInfo(
                canonical=canonical,
                builder=builder_callable,
                first_param_name=first_param.name,
                required_params=tuple(required),
                optional_params=tuple(optional),
                accepts_kwargs=accepts_kwargs,
            )

            alias = name
            if alias in aliases:
                aliases[alias] = None
            else:
                aliases[alias] = canonical

    return builders, aliases


_BUILDERS, _ALIASES = _collect_builders()


GROUPS: Mapping[str, Sequence[paper_figures.CircuitSpec]] = {
    "paper": paper_figures.CIRCUITS,
    "showcase": tuple(
        paper_figures.CircuitSpec(
            spec.name,
            spec.constructor,
            spec.default_qubits,
            None,
        )
        for spec in showcase_benchmarks.SHOWCASE_CIRCUITS.values()
    ),
}


def available_circuit_names() -> Mapping[str, BuilderInfo]:
    """Return the metadata of all builders keyed by canonical name."""

    return dict(_BUILDERS)


def available_aliases() -> Mapping[str, str | None]:
    """Return mapping of short aliases to canonical builder names."""

    return dict(_ALIASES)


def format_available_groups() -> str:
    """Return a formatted description of built-in estimation groups."""

    lines = ["Available estimation groups:"]
    for name, specs in sorted(GROUPS.items()):
        circuit_names = ", ".join(spec.name for spec in specs)
        lines.append(f"  - {name}: {circuit_names}")
    return "\n".join(lines)


def format_available_circuits() -> str:
    """Return a formatted description of available builder functions."""

    lines = ["Available estimation circuits:"]
    for canonical, info in sorted(_BUILDERS.items()):
        alias_notes: list[str] = []
        for alias, target in sorted(_ALIASES.items()):
            if target == canonical and alias != canonical.split(".")[-1]:
                alias_notes.append(alias)
        alias_text = f" (alias: {', '.join(alias_notes)})" if alias_notes else ""
        signature_text = _format_signature(info)
        lines.append(f"  - {canonical}{alias_text}: {signature_text}")
    return "\n".join(lines)


def _parse_kwargs(text: str) -> Dict[str, object]:
    if not text:
        return {}
    try:
        expr = ast.parse(f"f({text})", mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid parameter specification '{text}': {exc}") from exc
    if not isinstance(expr, ast.Expression) or not isinstance(expr.body, ast.Call):
        raise ValueError(f"Invalid parameter specification '{text}'")
    kwargs: Dict[str, object] = {}
    for keyword in expr.body.keywords:
        if keyword.arg is None:
            raise ValueError("Parameter values must be provided as keyword arguments")
        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    return kwargs


def _normalise_builder_name(name: str) -> str:
    raw = name.strip()
    if "." in raw:
        return raw
    alias = _ALIASES.get(raw)
    if alias is None:
        available = ", ".join(sorted(_BUILDERS))
        raise ValueError(
            f"Unknown or ambiguous circuit '{name}'. Available circuits: {available}"
        )
    return alias


def _parse_spec(text: str) -> tuple[str, dict[str, object], tuple[int, ...]]:
    raw = text.strip()
    widths: tuple[int, ...] = ()
    if ":" in raw:
        prefix, width_part = raw.split(":", 1)
        width_tokens = [token.strip() for token in width_part.split(",") if token.strip()]
        if not width_tokens:
            raise ValueError(f"No qubit widths provided in '{text}'")
        try:
            widths = tuple(int(token) for token in width_tokens)
        except ValueError as exc:
            raise ValueError(f"Invalid qubit width in '{text}': {exc}") from exc
    else:
        prefix = raw
    if "[" in prefix:
        if not prefix.endswith("]"):
            raise ValueError(f"Malformed parameter specification in '{text}'")
        name_part, kwargs_part = prefix[:-1].split("[", 1)
        kwargs = _parse_kwargs(kwargs_part)
    else:
        name_part = prefix
        kwargs = {}
    canonical = _normalise_builder_name(name_part)
    return canonical, kwargs, widths


def _format_name(base: str, kwargs: Mapping[str, object]) -> str:
    display = base.split(".")[-1]
    if not kwargs:
        return display
    parts = ", ".join(f"{key}={value!r}" for key, value in sorted(kwargs.items()))
    return f"{display}[{parts}]"


def _validate_kwargs(name: str, info: BuilderInfo, kwargs: Mapping[str, object]) -> None:
    if info.first_param_name in kwargs:
        raise ValueError(
            f"Circuit '{name}' must specify qubit widths instead of '{info.first_param_name}'"
        )
    missing = [param for param in info.required_params if param not in kwargs]
    if missing:
        required = ", ".join(missing)
        raise ValueError(f"Circuit '{name}' requires parameters: {required}")
    if not info.accepts_kwargs:
        allowed = set(info.required_params) | set(info.optional_params)
        unknown = [key for key in kwargs if key not in allowed]
        if unknown:
            unexpected = ", ".join(sorted(unknown))
            raise ValueError(
                f"Circuit '{name}' does not accept parameter(s): {unexpected}"
            )


def resolve_requested_specs(
    circuits: Sequence[str] | None,
    groups: Sequence[str] | None,
    *,
    default_group: str = "showcase",
) -> tuple[paper_figures.CircuitSpec, ...]:
    """Return the circuit specifications requested via CLI arguments."""

    selected: list[paper_figures.CircuitSpec] = []
    seen: set[tuple[str, tuple[int, ...], Callable[..., Circuit]]] = set()

    if groups:
        for name in groups:
            if name not in GROUPS:
                available = ", ".join(sorted(GROUPS))
                raise ValueError(
                    f"Unknown estimation group '{name}'. Available groups: {available}"
                )
            for spec in GROUPS[name]:
                key = (spec.name, tuple(spec.qubits), spec.builder)
                if key not in seen:
                    selected.append(spec)
                    seen.add(key)

    if circuits:
        for raw in circuits:
            canonical, kwargs, widths = _parse_spec(raw)
            info = _BUILDERS[canonical]
            if not widths:
                raise ValueError(
                    f"Circuit '{raw}' must include at least one qubit width using ':'"
                )
            _validate_kwargs(canonical, info, kwargs)
            name = _format_name(canonical, kwargs)
            spec = paper_figures.CircuitSpec(
                name,
                info.builder,
                widths,
                kwargs or None,
            )
            key = (spec.name, tuple(spec.qubits), spec.builder)
            if key not in seen:
                selected.append(spec)
                seen.add(key)

    if not selected:
        if default_group == "paper":
            return tuple(paper_figures.CIRCUITS)
        if default_group not in GROUPS:
            raise ValueError(
                f"Unknown default estimation group '{default_group}'. Available groups: "
                + ", ".join(sorted(GROUPS)),
            )
        return tuple(GROUPS[default_group])

    return tuple(selected)
