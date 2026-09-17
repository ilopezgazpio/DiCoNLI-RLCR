"""Keep the package layout modular as new features are added."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/main/python/rlcr"


def test_application_has_no_root_python_scripts():
    assert not list(ROOT.glob("*.py"))


def test_one_application_main_function():
    entry_points = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text())
        entry_points.extend(
            path for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
    assert entry_points == [PACKAGE / "cli.py"]


def test_classes_have_separate_bounded_modules():
    for path in PACKAGE.rglob("*.py"):
        source = path.read_text()
        classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
        assert len(classes) <= 1, f"Multiple classes in {path}"
        assert len(source.splitlines()) <= 300, f"Split the responsibilities in {path}"
