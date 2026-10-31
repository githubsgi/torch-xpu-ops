"""Append-only lessons ledger (port of core/lessons.py): what earlier trials MEASURED, never what anyone predicted.

Own file per run (<workdir>/lessons.md). Optional shared ledger dir: every run reads every file there and writes
only its own, so concurrent runs never touch the same file. A lesson is a record of a past measurement and never a
substitute for one.
"""

import time
from pathlib import Path


class Lessons:
    def __init__(self, workdir, shared_dir=None, tag="run"):
        self.own = Path(workdir) / "lessons.md"
        self.shared = Path(shared_dir).expanduser() if shared_dir else None
        self.tag = tag
        if self.shared:
            self.shared.mkdir(parents=True, exist_ok=True)
        if not self.own.exists():
            self.own.write_text(f"# Measured lessons ({tag})\n\nEach line is a measurement of one evaluated candidate "
                                f"(quick-suite score = geomean speedup vs production; > 1 is faster). Append-only.\n\n")

    def append(self, line):
        entry = f"- {time.strftime('%F %T')} {line}\n"
        with open(self.own, "a") as f:
            f.write(entry)
        if self.shared:
            with open(self.shared / f"{self.tag}.md", "a") as f:
                f.write(entry)

    def read(self, max_lines=40):
        """Own lessons plus the other files of the shared ledger, newest last, capped."""
        lines = [l for l in self.own.read_text().splitlines() if l.startswith("- ")]
        if self.shared:
            for p in sorted(self.shared.glob("*.md")):
                if p.name != f"{self.tag}.md":
                    lines += [f"[{p.stem}] {l}" for l in p.read_text().splitlines() if l.startswith("- ")]
        return "\n".join(lines[-max_lines:])
