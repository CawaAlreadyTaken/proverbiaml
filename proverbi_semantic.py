#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sqlite3
from typing import List, Tuple

import numpy as np
import pandas as pd

# FAISS
import faiss

# Transformer embeddings
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def init_db(conn: sqlite3.Connection, recreate: bool) -> None:
    cur = conn.cursor()
    if recreate:
        cur.execute("DROP TABLE IF EXISTS proverbi")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS proverbi (
            id          INTEGER PRIMARY KEY,
            categoria   TEXT NOT NULL,
            titolo      TEXT NOT NULL,
            descrizione TEXT NOT NULL
        )
        """
    )
    conn.commit()


def load_csv(csv_path: str, encoding: str) -> pd.DataFrame:
    # engine="python" gestisce bene virgole/quote in descrizione
    df = pd.read_csv(csv_path, encoding=encoding, engine="python")

    required = {"categoria", "titolo", "descrizione"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV: mancano le colonne: {sorted(missing)}")

    # Normalizza NaN -> ""
    df["categoria"] = df["categoria"].fillna("").astype(str)
    df["titolo"] = df["titolo"].fillna("").astype(str)
    df["descrizione"] = df["descrizione"].fillna("").astype(str)

    return df


def make_text_for_embedding(titolo: str, descrizione: str, include_title: bool) -> str:
    d = (descrizione or "").strip()
    t = (titolo or "").strip()

    if include_title:
        # migliora spesso la retrieval, specialmente quando descrizione è corta o mancante
        if d:
            return f"{t}. {d}" if t else d
        return t  # fallback

    # richiesto: embeddare la descrizione; fallback sul titolo se descrizione vuota
    return d if d else t


def build(
    csv_path: str,
    db_path: str,
    index_path: str,
    meta_path: str,
    model_name: str,
    encoding: str,
    batch_size: int,
    include_title: bool,
    recreate: bool,
) -> None:
    ensure_parent_dir(db_path)
    ensure_parent_dir(index_path)
    ensure_parent_dir(meta_path)

    df = load_csv(csv_path, encoding=encoding)

    # Assegna ID stabili (0..N-1) così l'ID nel DB coincide con l'ID nell'indice vettoriale
    df = df.reset_index(drop=True)
    df["id"] = np.arange(len(df), dtype=np.int64)

    # Salva su SQLite
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn, recreate=recreate)
        cur = conn.cursor()

        if recreate:
            cur.execute("DELETE FROM proverbi")

        rows = list(
            zip(
                df["id"].tolist(),
                df["categoria"].tolist(),
                df["titolo"].tolist(),
                df["descrizione"].tolist(),
            )
        )
        cur.executemany(
            "INSERT OR REPLACE INTO proverbi (id, categoria, titolo, descrizione) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    # Embedding
    model = SentenceTransformer(model_name)

    texts: List[str] = [
        make_text_for_embedding(t, d, include_title=include_title)
        for t, d in zip(df["titolo"].tolist(), df["descrizione"].tolist())
    ]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # così cosine similarity == inner product
    ).astype("float32")

    dim = embeddings.shape[1]
    ids = df["id"].to_numpy(dtype=np.int64)

    # Indice FAISS (cosine via inner product su vettori normalizzati)
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(embeddings, ids)

    faiss.write_index(index, index_path)

    meta = {
        "model_name": model_name,
        "dim": dim,
        "count": int(len(df)),
        "include_title": bool(include_title),
        "db_path": os.path.abspath(db_path),
        "index_path": os.path.abspath(index_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nOK ✅")
    print(f"- DB SQLite:   {db_path}")
    print(f"- Indice:     {index_path}")
    print(f"- Meta:       {meta_path}")
    print(f"- Proverbi:   {len(df)}")
    print(f"- Modello:    {model_name}")
    print(f"- Dimensione: {dim}")


def fetch_by_ids(conn: sqlite3.Connection, ids: List[int]) -> List[Tuple[int, str, str, str]]:
    if not ids:
        return []
    placeholders = ",".join(["?"] * len(ids))
    cur = conn.cursor()
    cur.execute(
        f"SELECT id, categoria, titolo, descrizione FROM proverbi WHERE id IN ({placeholders})",
        ids,
    )
    rows = cur.fetchall()
    # Ricostruisci in dizionario per mantenere l'ordine del ranking
    by_id = {r[0]: r for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def search(
    db_path: str,
    index_path: str,
    meta_path: str,
    query: str,
    top_k: int,
) -> None:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_name = meta["model_name"]
    model = SentenceTransformer(model_name)

    index = faiss.read_index(index_path)

    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(q_emb, top_k)
    scores = scores[0].tolist()
    ids = ids[0].astype(int).tolist()

    conn = sqlite3.connect(db_path)
    try:
        rows = fetch_by_ids(conn, ids)
    finally:
        conn.close()

    # Stampa risultati
    print("\nRisultati:")
    for rank, (pid, score) in enumerate(zip(ids, scores), start=1):
        row = next((r for r in rows if r[0] == pid), None)
        if not row:
            continue
        _id, categoria, titolo, descrizione = row
        print(f"\n#{rank}  score={score:.4f}  id={_id}")
        print(f"Categoria: {categoria}")
        print(f"Titolo:    {titolo}")
        if descrizione.strip():
            print(f"Descr.:    {descrizione}")
        else:
            print("Descr.:    (vuota)")

    if not rows:
        print("(nessun risultato trovato - controlla DB/indice)")


def main():
    p = argparse.ArgumentParser(description="Indicizza e cerca proverbi con embedding + FAISS + SQLite (tutto su file).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Crea/ricrea DB SQLite + indice FAISS a partire dal CSV.")
    pb.add_argument("--csv", default="proverbi.csv", help="Path del CSV (default: proverbi.csv)")
    pb.add_argument("--db", default="proverbi.db", help="Path del DB SQLite (default: proverbi.db)")
    pb.add_argument("--index", default="proverbi.faiss", help="Path indice FAISS (default: proverbi.faiss)")
    pb.add_argument("--meta", default="proverbi.meta.json", help="Path meta JSON (default: proverbi.meta.json)")
    pb.add_argument("--model", default=DEFAULT_MODEL, help=f"Modello embeddings (default: {DEFAULT_MODEL})")
    pb.add_argument("--encoding", default="utf-8", help="Encoding CSV (default: utf-8)")
    pb.add_argument("--batch-size", type=int, default=64, help="Batch size embedding (default: 64)")
    pb.add_argument("--include-title", action="store_true", help="Embeddings su 'titolo + descrizione' (consigliato).")
    pb.add_argument("--recreate", action="store_true", help="Drop/recreate tabella e ricrea indice da zero.")

    ps = sub.add_parser("search", help="Cerca nel DB/indice con una descrizione in italiano.")
    ps.add_argument("--db", default="proverbi.db", help="Path DB SQLite (default: proverbi.db)")
    ps.add_argument("--index", default="proverbi.faiss", help="Path indice FAISS (default: proverbi.faiss)")
    ps.add_argument("--meta", default="proverbi.meta.json", help="Path meta JSON (default: proverbi.meta.json)")
    ps.add_argument("--query", required=True, help="Testo della situazione da cercare")
    ps.add_argument("--top-k", type=int, default=5, help="Numero risultati (default: 5)")

    args = p.parse_args()

    if args.cmd == "build":
        build(
            csv_path=args.csv,
            db_path=args.db,
            index_path=args.index,
            meta_path=args.meta,
            model_name=args.model,
            encoding=args.encoding,
            batch_size=args.batch_size,
            include_title=args.include_title,
            recreate=args.recreate,
        )
    elif args.cmd == "search":
        search(
            db_path=args.db,
            index_path=args.index,
            meta_path=args.meta,
            query=args.query,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    # Esempio crea db + indice: python proverbi_semantic.py build --csv proverbi.csv --recreate --include-title
    # Esempio cerca proverbio: python proverbi_semantic.py search --query "Il capo dà il cattivo esempio e tutti lo seguono" --top-k 3
    main()

