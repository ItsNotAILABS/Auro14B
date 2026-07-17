"""GitHub monorepos as knowledge database + max embedding index.

Uses all harvested Medina / ItsNotAILABS / FreddyCreates / potential-succotash
docs as:
  1. SQLite document store (searchable info DB)
  2. Dense embedding matrix (ANN-style top-k retrieval)
  3. Training experience feed for continuous Auro mind training
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from auro_native_llm.corpus.embeddings import MaxEmbedder, cosine
from auro_native_llm.corpus.harvest import CorpusDocument, CorpusIndex, harvest_all, harvest_paths, default_roots

_DEFAULT_DB = Path.home() / ".auro_corpus" / "github_knowledge.db"
_DEFAULT_EMB = Path.home() / ".auro_corpus" / "github_embeddings.npz"


@dataclass
class RetrievalHit:
    doc_id: int
    path: str
    repo: str
    kind: str
    score: float
    preview: str
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "path": self.path,
            "repo": self.repo,
            "kind": self.kind,
            "score": self.score,
            "preview": self.preview,
        }


class GitHubKnowledgeDB:
    """Persistent info DB + max embeddings over your GitHubs."""

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        emb_path: Optional[str | Path] = None,
        *,
        embedder: Optional[MaxEmbedder] = None,
        max_dim: int = 0,
    ) -> None:
        self.db_path = Path(db_path or _DEFAULT_DB)
        self.emb_path = Path(emb_path or _DEFAULT_EMB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or MaxEmbedder(
            n_bands=128,
            n_fft_scales=4,
            n_phi=128,
            n_ngram=256,
            target_dim=max_dim,  # 0 = full concat ~ max
        )
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        self._matrix: Optional[np.ndarray] = None  # [N, dim]
        self._ids: List[int] = []
        self._load_embeddings()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                repo TEXT NOT NULL,
                kind TEXT,
                source TEXT,
                chars INTEGER,
                text TEXT,
                content_hash TEXT,
                updated_at REAL,
                UNIQUE(repo, path)
            );
            CREATE INDEX IF NOT EXISTS idx_docs_repo ON documents(repo);
            CREATE INDEX IF NOT EXISTS idx_docs_kind ON documents(kind);
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._conn.commit()

    def ingest_index(self, index: CorpusIndex, *, reembed: bool = True) -> Dict[str, Any]:
        """Upsert corpus documents into the knowledge DB."""
        n_new = 0
        n_upd = 0
        for d in index.documents:
            h = str(hash((d.repo, d.path, d.chars, d.text[:200])))
            cur = self._conn.execute(
                "SELECT id, content_hash FROM documents WHERE repo=? AND path=?",
                (d.repo, d.path),
            )
            row = cur.fetchone()
            now = time.time()
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO documents(path, repo, kind, source, chars, text, content_hash, updated_at)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (d.path, d.repo, d.kind, d.source, d.chars, d.text, h, now),
                )
                n_new += 1
            elif row["content_hash"] != h:
                self._conn.execute(
                    """
                    UPDATE documents SET kind=?, source=?, chars=?, text=?, content_hash=?, updated_at=?
                    WHERE id=?
                    """,
                    (d.kind, d.source, d.chars, d.text, h, now, row["id"]),
                )
                n_upd += 1
        self._conn.commit()
        self._set_meta("last_ingest", str(time.time()))
        self._set_meta(
            "roots",
            json.dumps([str(r) for r in (index.roots or [])]),
        )
        emb_stats = {}
        if reembed:
            emb_stats = self.rebuild_embeddings()
        return {
            "ok": True,
            "new": n_new,
            "updated": n_upd,
            "total": self.count(),
            "embeddings": emb_stats,
            "repos": self.repo_counts(),
        }

    def harvest_and_ingest(
        self,
        *,
        include_github: bool = True,
        include_succotash: bool = True,
        max_files: int = 4000,
        max_chars: int = 10_000_000,
        reembed: bool = True,
    ) -> Dict[str, Any]:
        """Harvest all GitHubs / local monorepos + optional succotash, then embed."""
        t0 = time.time()
        docs: List[CorpusDocument] = []
        roots = list(default_roots())

        # potential-succotash first-class
        if include_succotash:
            try:
                from auro_native_llm.succotash.corpus import harvest_succotash_corpus

                sidx = harvest_succotash_corpus(
                    max_files=min(1500, max_files),
                    max_total_chars=min(4_000_000, max_chars // 2),
                    clone=True,
                )
                docs.extend(sidx.documents)
                roots.extend(sidx.roots)
            except Exception as exc:
                pass

        # multi-repo harvest (cached clones preferred)
        try:
            idx = harvest_all(
                include_github_clones=include_github,
                max_files=max_files,
                max_total_chars=max_chars,
                clone_max_repos=40,
            )
            docs.extend(idx.documents)
            roots.extend(idx.roots)
        except Exception:
            # local only fallback
            idx = harvest_paths(roots[:8], max_files=max_files, max_total_chars=max_chars)
            docs.extend(idx.documents)

        # dedupe by repo:path keep longest
        by_key: Dict[str, CorpusDocument] = {}
        for d in docs:
            k = f"{d.repo}::{d.path}"
            if k not in by_key or d.chars > by_key[k].chars:
                by_key[k] = d
        uniq = list(by_key.values())
        index = CorpusIndex(documents=uniq, roots=list(dict.fromkeys(roots)))
        stats = self.ingest_index(index, reembed=reembed)
        stats["harvest_docs"] = len(uniq)
        stats["elapsed_s"] = time.time() - t0
        stats["embedder"] = self.embedder.info()
        return stats

    def rebuild_embeddings(self, *, batch_report: int = 100) -> Dict[str, Any]:
        """Embed every document at max capacity."""
        rows = self._conn.execute(
            "SELECT id, text FROM documents ORDER BY id"
        ).fetchall()
        if not rows:
            self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float64)
            self._ids = []
            return {"ok": True, "n": 0, "dim": self.embedder.dim}

        ids: List[int] = []
        vecs: List[np.ndarray] = []
        t0 = time.time()
        for i, row in enumerate(rows):
            text = row["text"] or ""
            # cap per-doc embed cost
            if len(text) > 12_000:
                text = text[:12_000]
            vecs.append(self.embedder.embed_text(text))
            ids.append(int(row["id"]))
            if batch_report and (i + 1) % batch_report == 0:
                print(f"[github-db] embedded {i+1}/{len(rows)} dim={self.embedder.dim}", flush=True)
        mat = np.stack(vecs, axis=0).astype(np.float64)
        self._matrix = mat
        self._ids = ids
        np.savez_compressed(
            self.emb_path,
            matrix=mat,
            ids=np.array(ids, dtype=np.int64),
            dim=np.array([self.embedder.dim]),
            embedder=json.dumps(self.embedder.info()),
        )
        self._set_meta("emb_path", str(self.emb_path))
        self._set_meta("emb_dim", str(self.embedder.dim))
        self._set_meta("emb_n", str(len(ids)))
        return {
            "ok": True,
            "n": len(ids),
            "dim": self.embedder.dim,
            "path": str(self.emb_path),
            "elapsed_s": time.time() - t0,
            "bytes": int(mat.nbytes),
        }

    def _load_embeddings(self) -> None:
        if not self.emb_path.exists():
            return
        try:
            data = np.load(self.emb_path, allow_pickle=True)
            self._matrix = data["matrix"]
            self._ids = [int(x) for x in data["ids"].tolist()]
        except Exception:
            self._matrix = None
            self._ids = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 12,
        repo: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> List[RetrievalHit]:
        """Max-embedding similarity search over GitHub knowledge."""
        if self._matrix is None or self._matrix.shape[0] == 0:
            # fallback keyword
            return self.keyword_search(query, top_k=top_k, repo=repo)
        q = self.embedder.embed_text(query)
        # cosine via matmul (rows L2-normalized)
        scores = self._matrix @ q
        order = np.argsort(-scores)
        hits: List[RetrievalHit] = []
        for idx in order:
            doc_id = self._ids[int(idx)]
            row = self._conn.execute(
                "SELECT id, path, repo, kind, text FROM documents WHERE id=?",
                (doc_id,),
            ).fetchone()
            if row is None:
                continue
            if repo and row["repo"] != repo:
                continue
            if kind and row["kind"] != kind:
                continue
            text = row["text"] or ""
            hits.append(
                RetrievalHit(
                    doc_id=int(row["id"]),
                    path=row["path"],
                    repo=row["repo"],
                    kind=row["kind"] or "",
                    score=float(scores[int(idx)]),
                    preview=text[:240],
                    text=text[:4000],
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def keyword_search(
        self, query: str, *, top_k: int = 12, repo: Optional[str] = None
    ) -> List[RetrievalHit]:
        tokens = [t for t in query.lower().split() if len(t) > 2]
        if not tokens:
            return []
        rows = self._conn.execute("SELECT id, path, repo, kind, text FROM documents").fetchall()
        scored: List[Tuple[float, sqlite3.Row]] = []
        for r in rows:
            if repo and r["repo"] != repo:
                continue
            blob = (r["text"] or "").lower()
            s = sum(blob.count(t) for t in tokens)
            if s > 0:
                scored.append((float(s), r))
        scored.sort(key=lambda x: -x[0])
        out = []
        for s, r in scored[:top_k]:
            text = r["text"] or ""
            out.append(
                RetrievalHit(
                    doc_id=int(r["id"]),
                    path=r["path"],
                    repo=r["repo"],
                    kind=r["kind"] or "",
                    score=s,
                    preview=text[:240],
                    text=text[:4000],
                )
            )
        return out

    def training_blocks(
        self,
        query: Optional[str] = None,
        *,
        max_blocks: int = 200,
        max_chars: int = 800_000,
        top_k_retrieve: int = 40,
    ) -> List[str]:
        """Texts for continuous training: retrieval-augmented or full scan."""
        blocks: List[str] = []
        total = 0
        if query:
            for h in self.search(query, top_k=top_k_retrieve):
                block = (
                    f"[GITHUB_DB repo={h.repo} path={h.path} score={h.score:.3f}]\n"
                    f"{h.text}\n[/GITHUB_DB]"
                )
                if total + len(block) > max_chars:
                    break
                blocks.append(block)
                total += len(block)
        # fill remaining from diverse repos
        if len(blocks) < max_blocks:
            by_repo = self.repo_counts()
            repos = list(by_repo.keys())
            for i, repo in enumerate(repos):
                rows = self._conn.execute(
                    "SELECT path, repo, kind, text FROM documents WHERE repo=? ORDER BY chars DESC LIMIT 8",
                    (repo,),
                ).fetchall()
                for r in rows:
                    block = (
                        f"[GITHUB_DB repo={r['repo']} path={r['path']} kind={r['kind']}]\n"
                        f"{(r['text'] or '')[:3000]}\n[/GITHUB_DB]"
                    )
                    if total + len(block) > max_chars:
                        return blocks
                    blocks.append(block)
                    total += len(block)
                    if len(blocks) >= max_blocks:
                        return blocks
        return blocks

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"])

    def repo_counts(self) -> Dict[str, int]:
        rows = self._conn.execute(
            "SELECT repo, COUNT(*) AS c FROM documents GROUP BY repo ORDER BY c DESC"
        ).fetchall()
        return {r["repo"]: int(r["c"]) for r in rows}

    def stats(self) -> Dict[str, Any]:
        return {
            "schema": "auro.github_knowledge_db.v1",
            "db_path": str(self.db_path),
            "emb_path": str(self.emb_path),
            "documents": self.count(),
            "repos": self.repo_counts(),
            "repo_count": len(self.repo_counts()),
            "embeddings_loaded": self._matrix is not None,
            "embedding_n": 0 if self._matrix is None else int(self._matrix.shape[0]),
            "embedding_dim": 0 if self._matrix is None else int(self._matrix.shape[1]),
            "embedder": self.embedder.info(),
            "total_chars": int(
                self._conn.execute("SELECT COALESCE(SUM(chars),0) AS c FROM documents").fetchone()["c"]
            ),
        }

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
