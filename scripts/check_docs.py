#!/usr/bin/env python3
"""Deterministic doc/invariant checker for AlchemistPlaybook.

Lightweight analog of living-docs' `check` (docs/decisions/0004). Enforces
the mechanical parts of the constitution's invariants:

  3. indexed-or-it-doesn't-exist  (docs reachable from docs/index.md;
     references present in the SKILL.md routing table; links resolve)
  5. no-structural-change-without-its-doc  (frontmatter typing, decision
     record shape)
  hard constraints: SKILL.md description <= 1024 chars; .skill archive in
     sync with the source tree; index.html has a DATA block.

Exit 0 = all pass. Stdlib only.
"""

import re
import sys
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SKILL_DIR = ROOT / "alchemist-playbook"
ARCHIVE = ROOT / "alchemist-playbook.skill"

ALLOWED_TYPES = {"index", "constitution", "architecture", "conventions",
                 "glossary", "workflow", "decision"}
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")

failures, passes = [], []


def check(ok, label, detail=""):
    if ok:
        passes.append(label)
    else:
        failures.append(f"{label}: {detail}" if detail else label)


def frontmatter(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    return (m.group(1) if m else None), text


# --- 1. docs/ frontmatter typing -------------------------------------------
doc_files = sorted(p for p in DOCS.rglob("*.md"))
for p in doc_files:
    fm, _ = frontmatter(p)
    rel = p.relative_to(ROOT)
    if fm is None:
        check(False, f"frontmatter {rel}", "missing YAML frontmatter")
        continue
    tm = re.search(r"^type:\s*(\S+)", fm, re.M)
    check(tm is not None and tm.group(1) in ALLOWED_TYPES,
          f"frontmatter {rel}", f"type must be one of {sorted(ALLOWED_TYPES)}")
    for field in ("title", "status", "updated"):
        check(re.search(rf"^{field}:", fm, re.M) is not None,
              f"frontmatter {rel}", f"missing '{field}:'")
    if p.parent.name == "decisions":
        check(re.match(r"^\d{4}-[a-z0-9-]+\.md$", p.name) is not None,
              f"decision name {rel}", "must be NNNN-kebab-case.md")
        check(re.search(r"^decided:", fm, re.M) is not None,
              f"decision {rel}", "missing 'decided:'")

# --- 2. reachability from docs/index.md + link integrity -------------------
reached, queue = set(), [DOCS / "index.md"]
while queue:
    p = queue.pop()
    if p in reached or not p.exists():
        continue
    reached.add(p)
    _, text = frontmatter(p)
    for target in LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        t = (p.parent / target).resolve()
        check(t.exists(), f"link {p.relative_to(ROOT)} -> {target}", "target missing")
        if t.suffix == ".md" and DOCS in t.parents:
            queue.append(t)

for p in doc_files:
    check(p in reached, f"indexed {p.relative_to(ROOT)}",
          "not reachable from docs/index.md")

# --- 3. routing table <-> references/ sync ----------------------------------
skill_md = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
routed = set(re.findall(r"references/([a-z0-9-]+\.md)", skill_md))
on_disk = {p.name for p in (SKILL_DIR / "references").glob("*.md")}
for name in sorted(on_disk - routed):
    check(False, f"routing references/{name}", "exists but not mentioned in SKILL.md")
for name in sorted(routed - on_disk):
    check(False, f"routing references/{name}", "mentioned in SKILL.md but missing")
if on_disk == routed:
    check(True, "routing table in sync with references/")

# --- 4. SKILL.md description cap --------------------------------------------
fm, _ = frontmatter(SKILL_DIR / "SKILL.md")
desc_lines, in_desc = [], False
for line in fm.splitlines():
    if re.match(r"^description:", line):
        in_desc = True
        continue
    if in_desc:
        if line.startswith((" ", "\t")):
            desc_lines.append(line.strip())
        else:
            break
desc = " ".join(desc_lines)
check(len(desc) <= 1024, "SKILL.md description length",
      f"{len(desc)} chars > 1024 cap")

# --- 5. archive freshness (content-level, mtime-independent) ----------------
src_files = {
    str(p.relative_to(ROOT)): zlib.crc32(p.read_bytes()) & 0xFFFFFFFF
    for p in SKILL_DIR.rglob("*")
    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
}
with zipfile.ZipFile(ARCHIVE) as zf:
    zip_files = {i.filename: i.CRC for i in zf.infolist() if not i.is_dir()}
stale = sorted(
    (set(src_files) ^ set(zip_files))
    | {f for f in set(src_files) & set(zip_files) if src_files[f] != zip_files[f]}
)
check(not stale, "archive freshness",
      "out of sync (repackage per docs/workflows/release.md): "
      + ", ".join(stale[:5]) + ("…" if len(stale) > 5 else ""))

# --- 6. human report sanity --------------------------------------------------
html_path = DOCS / "index.html"
html = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
check("const DATA" in html, "docs/index.html", "missing or lacks `const DATA` block")
check("http://" not in html and 'src="https' not in html and "@import" not in html,
      "docs/index.html self-contained", "must not load external resources")

# --- report ------------------------------------------------------------------
print(f"check_docs: {len(passes)} passed, {len(failures)} failed")
for f in failures:
    print(f"  FAIL  {f}")
sys.exit(1 if failures else 0)
