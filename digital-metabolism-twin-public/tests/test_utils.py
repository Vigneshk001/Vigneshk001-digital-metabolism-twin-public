from pathlib import Path

from src.utils.helpers import hash_files


def test_hash_files_changes_with_content(tmp_path: Path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("hello")
    f2.write_text("world")
    hash1 = hash_files([f1, f2])
    f2.write_text("world!")
    hash2 = hash_files([f1, f2])
    assert hash1 != hash2
