#!/usr/bin/env python3
"""Generate a YAML file with code-defined defaults for gin configurables."""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import ptyrax
import ptyrax.external_configurables  # noqa: F401
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


def _import_all_ptyrax_modules() -> None:
    """Import all non-test ptyrax modules to register gin configurables."""
    for module_info in pkgutil.walk_packages(ptyrax.__path__, ptyrax.__name__ + "."):
        module_name = module_info.name
        if module_name.endswith("_test") or ".tests" in module_name:
            continue
        importlib.import_module(module_name)


def _get_source_path(target: object, repo_root: Path, module_fallback: str) -> str:
    """Resolve source file path for a configurable target."""
    source_file = inspect.getsourcefile(target)
    if source_file is not None:
        try:
            return str(Path(source_file).resolve().relative_to(repo_root))
        except ValueError:
            return str(Path(source_file).resolve())
    return f"{module_fallback.replace('.', '/')}.py"


def _collect_defaults(repo_root: Path) -> CommentedMap:
    """Collect defaults sorted by source file with section comments."""
    import gin.config as gin_config

    sortable_entries: list[tuple[str, str, dict[str, object]]] = []
    registry = gin_config._REGISTRY

    for selector, configurable in registry.items():
        if not configurable.module.startswith("ptyrax"):
            continue

        target = configurable.wrapped
        signature_fn = target
        if inspect.isclass(target):
            signature_fn = gin_config._find_class_construction_fn(target)

        defaults = gin_config._get_default_configurable_parameter_values(
            signature_fn,
            configurable.allowlist,
            configurable.denylist,
        )
        # The current YAML-to-gin binder flattens nested mappings into key paths,
        # so dict-valued parameter defaults cannot be represented losslessly.
        defaults = {key: value for key, value in defaults.items() if not isinstance(value, dict)}
        if not defaults:
            continue

        minimal_selector = registry.minimal_selector(selector)
        source_path = _get_source_path(target, repo_root, configurable.module)
        sortable_entries.append((source_path, minimal_selector, copy.deepcopy(defaults)))

    sortable_entries.sort(key=lambda entry: (entry[0], entry[1]))

    defaults_by_selector: CommentedMap = CommentedMap()
    previous_source_path: str | None = None
    for source_path, selector, defaults in sortable_entries:
        if source_path != previous_source_path:
            defaults_by_selector.yaml_set_comment_before_after_key(selector, before=f"From {source_path}")
            previous_source_path = source_path
        defaults_by_selector[selector] = defaults
    return defaults_by_selector


def generate_default_yaml(output_path: Path) -> None:
    """Write the default configuration YAML to disk."""
    _import_all_ptyrax_modules()

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.representer.ignore_aliases = lambda _data: True

    repo_root = Path(__file__).resolve().parents[1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root_config = CommentedMap()
    root_config["__main__"] = _collect_defaults(repo_root)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.dump(root_config, file)

    _format_section_comments(output_path)


def _format_section_comments(output_path: Path) -> None:
    """Expand one-line section comments into thick three-line blocks."""
    text = output_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    formatted_lines: list[str] = []
    border = "=" * 78

    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        comment_match = re.match(r"#\s+From\s+.+", stripped)
        if comment_match:
            header = stripped[2:].strip()
            formatted_lines.append("")
            formatted_lines.append(f"{indent}# {border}")
            formatted_lines.append(f"{indent}# {header}")
            formatted_lines.append(f"{indent}# {border}")
            continue
        formatted_lines.append(line)

    output_path.write_text("\n".join(formatted_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default="configs/defaults.yaml",
        help="Path to write the generated defaults YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate_default_yaml(Path(arguments.output))
