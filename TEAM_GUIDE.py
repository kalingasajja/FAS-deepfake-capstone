"""
CLI helper to display TEAM_GUIDE.md in the terminal.
Usage: python TEAM_GUIDE.py
"""
import pathlib

guide_path = pathlib.Path(__file__).parent / "TEAM_GUIDE.md"
if guide_path.exists():
    print(guide_path.read_text(encoding="utf-8"))
else:
    print("TEAM_GUIDE.md not found.")

