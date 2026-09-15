"""命令行入口：先建索引，再提问。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag.ingest import build_index
from rag.pipeline import ask


def main() -> None:
    parser = argparse.ArgumentParser(description="星河银行 RAG Demo")
    parser.add_argument("question", nargs="?", help="要问知识库的问题")
    parser.add_argument("--rebuild", action="store_true", help="重建向量索引")
    parser.add_argument("--k", type=int, default=None, help="检索条数")
    args = parser.parse_args()

    if args.rebuild or args.question is None:
        stats = build_index(reset=True)
        print(
            f"索引完成：{stats['documents']} 篇文档 → {stats['chunks']} 个 chunk"
            f"（embedding={stats['embedding_backend']}）"
        )
        if args.question is None:
            return

    result = ask(args.question, k=args.k)
    print(f"\n生成器：{result.generator}")
    print(f"答案：{result.answer}\n")
    print("引用：")
    for i, doc in enumerate(result.sources, start=1):
        src = Path(str(doc.metadata.get("source", ""))).name
        snippet = doc.page_content.strip().replace("\n", " ")[:80]
        print(f"  [{i}] {src}: {snippet}")


if __name__ == "__main__":
    main()
