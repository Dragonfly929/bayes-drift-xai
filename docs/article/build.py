"""Build the article: articol.md + references.bib -> articol.docx -> articol.pdf

    python docs/article/build.py

Needs pandoc (>= 3) and, for the PDF, LibreOffice (soffice). Figures are read from ../../figures, so run
`python -m bdx.analyze` first. Native Word equations are produced from the LaTeX math in the Markdown.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "Times New Roman"


def make_reference(path):
    """Pandoc's default reference.docx restyled for an academic paper."""
    with open(path, "wb") as f:
        f.write(subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                               check=True, capture_output=True).stdout)
    d = Document(path)
    st = d.styles

    def font(name, size, bold=None, italic=None, color=RGBColor(0, 0, 0), align=None, before=None, after=None):
        match = [s for s in st if s.name and s.name.lower() == name.lower()]
        if not match:
            return
        s = match[0]
        s.font.name = FONT
        s.font.size = Pt(size)
        s.font.color.rgb = color
        rpr = s.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.append(rf)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), FONT)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]
        if bold is not None:
            s.font.bold = bold
        if italic is not None:
            s.font.italic = italic
        if not hasattr(s, "paragraph_format"):      # character style (e.g. Hyperlink)
            return
        pf = s.paragraph_format
        if align is not None:
            pf.alignment = align
        if before is not None:
            pf.space_before = Pt(before)
        if after is not None:
            pf.space_after = Pt(after)

    J = WD_ALIGN_PARAGRAPH.JUSTIFY
    font("Normal", 11)
    font("Body Text", 11, align=J, before=0, after=6)
    font("First Paragraph", 11, align=J, before=0, after=6)
    font("Compact", 11, align=J)
    font("Title", 16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    font("Subtitle", 12, align=WD_ALIGN_PARAGRAPH.CENTER)
    font("Author", 11, align=WD_ALIGN_PARAGRAPH.CENTER)
    font("Date", 10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=12)
    font("Heading 1", 13, bold=True, before=14, after=6)
    font("Heading 2", 11.5, bold=True, italic=False, before=10, after=4)
    font("Heading 3", 11, bold=False, italic=True, before=8, after=3)
    font("Caption", 9.5, italic=False, align=J, after=10)
    font("Image Caption", 9.5, italic=False, align=J, after=10)
    font("Table Caption", 9.5, italic=False, align=WD_ALIGN_PARAGRAPH.LEFT, before=8, after=4)
    font("Bibliography", 9.5, align=WD_ALIGN_PARAGRAPH.LEFT)
    font("Hyperlink", 11, color=RGBColor(0x1F, 0x4E, 0x9A))
    for sec in d.sections:
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        sec.left_margin = sec.right_margin = Cm(2.3)
        sec.top_margin = sec.bottom_margin = Cm(2.2)
    d.save(path)


def _border(tag, val="single", sz=8):
    e = OxmlElement(f"w:{tag}")
    e.set(qn("w:val"), val)
    e.set(qn("w:sz"), str(sz))
    e.set(qn("w:space"), "0")
    e.set(qn("w:color"), "000000")
    return e


def postprocess(path):
    """Academic (booktabs-like) tables, small table font, page numbers, bibliography hanging indent."""
    d = Document(path)
    for t in d.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = t._tbl.tblPr
        old = tblPr.find(qn("w:tblBorders"))
        if old is not None:
            tblPr.remove(old)
        b = OxmlElement("w:tblBorders")
        b.append(_border("top", sz=10))
        b.append(_border("bottom", sz=10))
        b.append(_border("insideH", sz=2))
        for tag in ("left", "right", "insideV"):
            b.append(_border(tag, "nil"))
        tblPr.append(b)
        for i, row in enumerate(t.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.space_before = Pt(1)
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs:
                        r.font.size = Pt(8.5)
                        r.font.name = FONT
                        if i == 0:
                            r.font.bold = True
    # page number in the footer
    for sec in d.sections:
        p = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if kind:
                fc = OxmlElement("w:fldChar")
                fc.set(qn("w:fldCharType"), kind)
                run._r.append(fc)
            else:
                it = OxmlElement("w:instrText")
                it.set(qn("xml:space"), "preserve")
                it.text = text
                run._r.append(it)
        run.font.size = Pt(9)
    for p in d.paragraphs:
        if p.style.name == "Bibliography":
            p.paragraph_format.left_indent = Cm(0.8)
            p.paragraph_format.first_line_indent = Cm(-0.8)
            p.paragraph_format.space_after = Pt(3)
    d.save(path)


def main():
    os.chdir(HERE)
    ref = os.path.join(HERE, "reference.docx")
    make_reference(ref)
    out = os.path.join(HERE, "articol.docx")
    subprocess.run(["pandoc", "articol.md", "-o", out, "--citeproc", "--bibliography", "references.bib",
                    "--reference-doc", ref, "--resource-path", HERE, "-M", "lang=ro-RO",
                    "-M", "link-citations=true"], check=True)
    postprocess(out)
    os.remove(ref)
    if shutil.which("xelatex") and "--no-pdf" not in sys.argv:
        # PDF straight from Markdown with XeLaTeX (LibreOffice drops Word equations when exporting to PDF)
        hdr = os.path.join(HERE, "_header.tex")
        open(hdr, "w").write("\\usepackage{caption}\n\\captionsetup{labelformat=empty,font=small}\n"
                             "\\usepackage{float}\n\\floatplacement{figure}{H}\n"
                             "\\setlength{\\emergencystretch}{3em}\n")
        subprocess.run(["pandoc", "articol.md", "-o", os.path.join(HERE, "articol.pdf"), "--citeproc",
                        "--bibliography", "references.bib", "--resource-path", HERE, "--pdf-engine=xelatex",
                        "-M", "lang=ro-RO", "-V", "fontfamily=fontspec", "-V", "mainfont=TeX Gyre Termes", "-V", "mathfont=Latin Modern Math",
                        "-V", "geometry:a4paper,margin=2.3cm", "-V", "fontsize=11pt", "-V", "colorlinks=true",
                        "-V", "linkcolor=blue", "-V", "urlcolor=blue", "-V", "citecolor=blue",
                        "-H", hdr], check=True)
        os.remove(hdr)
    print("built", out)


if __name__ == "__main__":
    main()
