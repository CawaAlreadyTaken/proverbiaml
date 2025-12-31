#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import List
import pandas as pd
import pdfplumber

TITLE_FONT_SUBSTRINGS = ("AdvTimes-b", "AdvTimes-bi", "AdvTimes-i")

def join_and_clean(tokens: List[str]) -> str:
    tokens = [t for t in tokens if t and t.strip()]
    if not tokens:
        return ""

    text = " ".join(tokens)

    # Ricompone sillabazioni di fine riga: "asso- luto" -> "assoluto"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Spazi prima della punteggiatura
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Spazi dentro parentesi/quadre
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]])", r"\1", text)

    # Normalizza spazi
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def order_words_by_columns_and_lines(words, page_width: float, y_line_tol: float = 3.5):
    mid = page_width / 2.0
    cols = [[], []]
    for w in words:
        cols[0 if w["x0"] < mid else 1].append(w)

    ordered = []
    for ci in [0, 1]:
        col_words = sorted(cols[ci], key=lambda w: (w["top"], w["x0"]))

        lines = []
        current_line = []
        current_top = None

        for w in col_words:
            if current_top is None:
                current_line = [w]
                current_top = w["top"]
                continue

            if abs(w["top"] - current_top) <= y_line_tol:
                current_line.append(w)
                current_top = min(current_top, w["top"])
            else:
                current_line.sort(key=lambda ww: ww["x0"])
                lines.append((current_top, current_line))
                current_line = [w]
                current_top = w["top"]

        if current_line:
            current_line.sort(key=lambda ww: ww["x0"])
            lines.append((current_top, current_line))

        lines.sort(key=lambda t: t[0])
        for _, line_words in lines:
            ordered.extend(line_words)

    return ordered


def parse_proverbi_pdf(
    pdf_path: str,
    skip_pages: int = 64,
    x_tolerance: float = 1.0,
    y_tolerance: float = 3.0,
    y_line_tol: float = 3.5,
) -> pd.DataFrame:
    rows = []
    current_category = None

    in_entry = False
    in_title = True
    title_tokens: List[str] = []
    desc_tokens: List[str] = []

    def flush_entry():
        nonlocal in_entry, in_title, title_tokens, desc_tokens
        if not in_entry:
            return
        titolo = join_and_clean(title_tokens)
        descrizione = join_and_clean(desc_tokens)
        if titolo:
            rows.append(
                {
                    "categoria": current_category,
                    "titolo": titolo,
                    "descrizione": descrizione,
                }
            )
        in_entry = False
        in_title = True
        title_tokens = []
        desc_tokens = []

    with pdfplumber.open(pdf_path) as pdf:
        for pno in range(skip_pages, len(pdf.pages)):
            page = pdf.pages[pno]
            mid = page.width / 2.0

            words = page.extract_words(
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
                extra_attrs=["size", "fontname"],
            )
            ordered = order_words_by_columns_and_lines(words, page.width, y_line_tol=y_line_tol)

            for w in ordered:
                txt = w["text"]
                font = w.get("fontname", "")
                size = float(w.get("size", 0.0))

                # Simbolo-freccetta/ornamenti (nel PDF spesso appare come "f" in font Advfs)
                if font.endswith("Advfs") and txt in {"f", "g", "h"}:
                    continue

                # Categoria (es: ABATE, ABITUDINE, ...)
                if ("AdvUniv-b" in font) and (txt.strip().upper() == txt.strip()) and len(txt.strip()) >= 3:
                    flush_entry()
                    current_category = txt.strip()
                    continue

                # Numero del proverbio
                if re.fullmatch(r"\d{1,4}", txt) and size <= 7.2 and ("AdvUniv" in font):
                    # Filtro per evitare numeri di pagina/altre numerazioni fuori margine:
                    if w["x0"] < mid:
                        if not (90 <= w["x0"] <= 210):
                            continue
                    else:
                        if not (mid + 5 <= w["x0"] <= mid + 210):
                            continue

                    flush_entry()
                    in_entry = True
                    in_title = True
                    continue

                if not in_entry:
                    continue

                is_title_font = any(s in font for s in TITLE_FONT_SUBSTRINGS)

                if in_title and is_title_font:
                    title_tokens.append(txt)
                elif in_title:
                    # Permetti punteggiatura immediatamente dopo il titolo
                    if txt in {",", ".", ";", ":", "?", "!", ")", "]"} and title_tokens:
                        title_tokens.append(txt)
                    else:
                        in_title = False
                        desc_tokens.append(txt)
                else:
                    desc_tokens.append(txt)

    flush_entry()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    # Esempio: python parse_proverbi.py proverbi.pdf --skip-pages 64 --out proverbi.csv

    ap = argparse.ArgumentParser(description="Estrae proverbi (categoria, titolo, descrizione) da proverbi.pdf")
    ap.add_argument("pdf", help="Percorso del PDF (es: proverbi.pdf)")
    ap.add_argument("--skip-pages", type=int, default=64, help="Pagine iniziali da saltare (default: 64)")
    ap.add_argument("--out", default="", help="Se valorizzato, salva il CSV nel percorso indicato")
    args = ap.parse_args()

    df = parse_proverbi_pdf(args.pdf, skip_pages=args.skip_pages)
    print(df.head(20))
    print(f"\nRighe estratte: {len(df):,}")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"CSV salvato in: {args.out}")

