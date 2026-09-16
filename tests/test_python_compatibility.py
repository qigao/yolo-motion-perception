"""Static regressions that remain collectable on the minimum Python version."""

import ast
from pathlib import Path


def test_python_310_sources_do_not_import_typing_self() -> None:
    root = Path(__file__).resolve().parents[1]
    violations = []
    for directory in (root / "src", root / "tests"):
        for path in sorted(directory.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "typing"
                    and any(alias.name == "Self" for alias in node.names)
                ):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}")

    assert not violations, f"typing.Self is unavailable on Python 3.10: {violations}"
