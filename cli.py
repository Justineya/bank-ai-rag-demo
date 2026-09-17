"""命令行入口：先建索引，再提问。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag.eval import run_eval
from rag.ingest import build_index
from rag.pipeline import ask
from rag.tenants import apply_tenant


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 教室（星河银行 / 闽信保险）")
    parser.add_argument("question", nargs="?", help="要问知识库的问题")
    parser.add_argument("--rebuild", action="store_true", help="重建向量索引")
    parser.add_argument("--k", type=int, default=None, help="检索条数")
    parser.add_argument("--eval", action="store_true", help="跑固定评测集并打印命中/拒答/引用率")
    parser.add_argument("--tenant", choices=["bank", "minxin"], default=None, help="叙事：bank 星河 / minxin 闽信")
    args = parser.parse_args()
    if args.tenant:
        apply_tenant(args.tenant)

    if args.eval:
        if args.rebuild:
            stats = build_index(reset=True)
            print(f"索引完成：{stats['chunks']} chunk（{stats['embedding_backend']}）")
        report = run_eval(k=args.k)
        print(
            f"评测 {report['n']} 条 · 命中率 {report['hit_rate']:.0%} · "
            f"拒答正确率 {report['refuse_accuracy']:.0%} · "
            f"引用点名率 {report['cite_named_rate']:.0%} · 及格 {report['pass_rate']:.0%}"
        )
        for row in report["rows"]:
            mark = "OK" if row["answer_ok"] else "FAIL"
            print(f"  [{mark}] {row['id']}: {row['question']}")
        return

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
