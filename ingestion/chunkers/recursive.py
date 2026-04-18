"""Recursive character/token-aware chunking with overlap."""

from __future__ import annotations

import re
import uuid

import tiktoken

from core.config import Settings
from core.models import ChunkRecord, DocumentInput


class RecursiveChunker:
    """Splits on paragraphs/sentences then packs into token windows."""

    def __init__(self, settings: Settings) -> None:
        self._chunk_size = settings.chunk_size
        self._overlap = settings.chunk_overlap
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = tiktoken.get_encoding("gpt2")

    def _split_units(self, text: str) -> list[str]:
        parts = re.split(r"\n\n+", text.strip())
        units: list[str] = []
        for p in parts:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            for s in sentences:
                s = s.strip()
                if s:
                    units.append(s)
        return units if units else [text]

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    async def chunk(self, doc: DocumentInput) -> list[ChunkRecord]:
        units = self._split_units(doc.text)
        chunks: list[str] = []
        buf: list[str] = []
        buf_tokens = 0

        for u in units:
            t = self._count_tokens(u)
            if t > self._chunk_size:
                # hard-split long unit
                enc = self._enc.encode(u)
                start = 0
                while start < len(enc):
                    end = min(start + self._chunk_size, len(enc))
                    piece = self._enc.decode(enc[start:end])
                    chunks.append(piece)
                    start = end - self._overlap if end < len(enc) else end
                buf, buf_tokens = [], 0
                continue

            if buf_tokens + t > self._chunk_size and buf:
                chunks.append(" ".join(buf))
                # overlap: keep tail units that fit in overlap window
                overlap_buf: list[str] = []
                ot = 0
                for b in reversed(buf):
                    bt = self._count_tokens(b)
                    if ot + bt <= self._overlap:
                        overlap_buf.insert(0, b)
                        ot += bt
                    else:
                        break
                buf = overlap_buf
                buf_tokens = sum(self._count_tokens(x) for x in buf)
            buf.append(u)
            buf_tokens += t

        if buf:
            chunks.append(" ".join(buf))

        out: list[ChunkRecord] = []
        for i, c in enumerate(chunks):
            cid = f"{doc.id}::{uuid.uuid4().hex[:12]}::{i}"
            meta = {**doc.metadata, "chunk_index": i, "strategy": "recursive"}
            out.append(ChunkRecord(chunk_id=cid, doc_id=doc.id, text=c, metadata=meta))
        return out
