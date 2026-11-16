"""Build the report PDF from the markdown draft, on the Apart template's layout (apart.tex).

    python3 experiments/epic2/phase2_confidence_gap/report/build.py [draft.md]

Default draft: report_full_draft.md (kept out of git). Writes build/report.tex and build/report.pdf
(also kept out of git). Needs pandoc and tectonic.

Markdown conventions the script understands:
- first line "# Title"; second line "Author · Affiliation · email" (" · "-separated; several authors
  separated by ";");
- "## Abstract" then one paragraph;
- a figure is a line "**[Insert ... `path/to/figure.png`]**" followed by a caption paragraph
  starting "**Figure X.**" (or both on one line); numbering is automatic, "C" for appendix figures;
- "## Appendix A ..." starts the appendices.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
FIG = re.compile(r"^\*\*\[Insert[^`]*`([^`]+)`\]\*\*\s*(.*)$")
CAPTION = re.compile(r"^\*\*Figure [A-Z]?\d+\.\*\*\s*")


def latex(md: str) -> str:
    return subprocess.run(["pandoc", "-f", "markdown+autolink_bare_uris", "-t", "latex", "--wrap=none"], input=md,
                          capture_output=True, text=True, check=True).stdout.strip()


def figure(path: str, caption: str, wide: bool) -> str:
    cap = latex(caption)
    width = r"\textwidth" if wide else r"0.92\textwidth"
    return ("```{=latex}\n\\begin{figure}[H]\n\\centering\n"
            f"\\includegraphics[width={width}]{{{path}}}\n\\caption{{{cap}}}\n\\end{{figure}}\n```")


def body(lines: list[str]) -> str:
    out, i, in_appendix = [], 0, False
    while i < len(lines):
        line = lines[i]
        if line.startswith("## Appendix") and not in_appendix:
            in_appendix = True
            out.append("```{=latex}\n\\clearpage\n\\setcounter{figure}{0}\n"
                       "\\renewcommand{\\thefigure}{C\\arabic{figure}}\n```\n")
        m = FIG.match(line)
        if m:
            path, rest = m.group(1), m.group(2)
            if not rest:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                rest, i = lines[j], j
            out.append(figure(path, CAPTION.sub("", rest), wide=not in_appendix))
        else:
            out.append(line)
        i += 1
    return "\n".join(out)


def main() -> None:
    src = HERE / (sys.argv[1] if len(sys.argv) > 1 else "report_full_draft.md")
    lines = src.read_text().splitlines()
    title = lines[0].lstrip("# ").strip()
    authors = []
    for a in lines[2].split(";"):
        parts = [p.strip() for p in a.split("·")]
        authors.append(r"\\".join(latex(p) for p in parts if p))
    rest = lines[3:]
    k = rest.index("## Abstract")
    j = k + 1
    while not rest[j].strip():
        j += 1
    abstract = latex(rest[j])
    md = body(rest[j + 1:])
    tex_body = subprocess.run(["pandoc", "-f", "markdown+autolink_bare_uris", "-t", "latex", "--wrap=none",
                               "--shift-heading-level-by=-1"],
                              input=md, capture_output=True, text=True, check=True).stdout
    template = (HERE / "apart.tex").read_text()
    author_block = r"\quad".join(r"\begin{tabular}[t]{c}" + a + r"\end{tabular}" for a in authors)
    tex = (template.replace("$title$", title).replace("$authors$", author_block)
           .replace("$abstract$", abstract).replace("$body$", tex_body))
    BUILD.mkdir(exist_ok=True)
    (BUILD / "report.tex").write_text(tex)
    subprocess.run(["tectonic", "--keep-logs", "-o", str(BUILD), str(BUILD / "report.tex")], check=True,
                   cwd=HERE)
    print("wrote", BUILD / "report.pdf")


if __name__ == "__main__":
    main()
