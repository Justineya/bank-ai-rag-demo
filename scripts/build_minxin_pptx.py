#!/usr/bin/env python3
"""Build a sample deck from the Minxin Insurance PPTX template.

Method
------
The template has one empty slide layout and no placeholders. The design
lives on the sample slides themselves (freeform shapes, pictures, embedded
fonts). Opening it with python-pptx and adding a layout slide would produce
blank pages, and redrawing the master in PptxGenJS would not keep the artwork.

This script therefore:

1. Copies the original .pptx package (a zip of OOXML parts).
2. Keeps the visual sample slides and drops the design-brief pages.
3. Clones the section-divider slide once per chapter.
4. Replaces only the text inside existing ``<a:t>`` nodes, so font, size,
   color, alignment, and position stay as designed.
5. Rewrites the slide list, relationships, and content types to match.

Text is kept near the length of the original copy because these boxes do not
autofit. Figures in the deck are demonstration copy for the layout, not
reported company results.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "minxin-insurance-template.pptx"
OUTPUT = ROOT / "samples" / "minxin-insurance-2026-demo.pptx"

SLIDE_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
)


def replace_text_nodes(xml: str, pairs: list[tuple[str, str]]) -> str:
    """Replace each exact ``<a:t>`` value once, leaving run properties intact."""
    for old, new in pairs:
        needle = f"<a:t>{old}</a:t>"
        found = xml.count(needle)
        if found != 1:
            raise SystemExit(f"expected 1 occurrence of {old!r}, found {found}")
        xml = xml.replace(needle, f"<a:t>{escape(new)}</a:t>", 1)
    return xml


def divider(xml: str, title: str, number: str) -> str:
    pairs = [("封面页", title), ("20XX", "2026")]
    if number != "01":
        pairs.append(("01", number))
    return replace_text_nodes(xml, pairs)


def build_slides(parts: dict[str, str]) -> list[tuple[str, str]]:
    """Return (slide xml, slide rels xml) in presentation order."""
    cover = replace_text_nodes(
        parts["ppt/slides/slide1.xml"],
        [
            ("主讲人：AiPPT", "汇报人：演示"),
            ("时间：20XX.X", "时间：2026.3"),
            ("闽信保险有限公司PPT模板设计大纲", "闽信保险有限公司\n2026年度业务汇报"),
            ("202X", "2026"),
        ],
    )
    agenda = replace_text_nodes(
        parts["ppt/slides/slide6.xml"],
        [
            ("章节导航", "汇报目录"),
            (
                "介绍公司成立背景、发展历程、核心业务\n展示公司文化、价值观、团队建设",
                "香港财产险服务商，服务本地企业\n聚焦保障、风控与客户服务",
            ),
            (
                "阐述业务覆盖范围、市场定位\n分析业务模式、运营流程",
                "企业财产、责任险与工程险\n覆盖制造、物流和地产场景",
            ),
            (
                "介绍产品种类、特色\n分析产品竞争力、市场反馈",
                "险种矩阵搭配场景方案\n突出承保能力与服务响应",
            ),
            (
                "展示业务数据、财务数据\n分析数据趋势、行业对比",
                "保费、赔付与区域覆盖\n以下数字为演示口径",
            ),
            (
                "阐述短期目标、长期愿景\n分析战略规划、实施路径",
                "深耕大湾区，推进科技应用\n持续优化承保与理赔体验",
            ),
        ],
    )
    overview = replace_text_nodes(
        parts["ppt/slides/slide14.xml"],
        [
            (
                "香港专业财产险服务商，深耕本地市场\n提供多元化保险产品，满足客户需求",
                "立足香港的专业财产险服务商\n为企业客户提供风险保障",
            ),
            (
                "成立于20XX年，注册资本X亿元\n获得多项资质荣誉，树立良好口碑",
                "扎根本地市场，持续完善服务体系\n以稳健承保积累长期口碑",
            ),
            (
                "覆盖企业财产险、责任险、工程险等\n为不同行业提供定制化保险方案",
                "以企业财产、责任、工程险为主\n按行业场景设计保障方案",
            ),
        ],
    )
    layout = replace_text_nodes(
        parts["ppt/slides/slide15.xml"],
        [
            (
                "风险管理能力强，专业团队精准评估\n全球化再保网络，分散风险，提升承保能力",
                "专业团队评估风险，匹配承保方案\n再保安排分担风险，承保更稳",
            ),
            (
                "承保规模增长率达X%，市场拓展迅速\n客户留存率高达X%，客户满意度高",
                "保费规模稳步增长，续保意愿稳定\n客户留存保持高位，口碑扎实",
            ),
            (
                "合作知名企业，如XX集团、XX公司\n为其提供全方位保险保障，助力企业发展",
                "服务制造、物流与地产等企业\n覆盖承保、批改与理赔协同",
            ),
        ],
    )
    products = replace_text_nodes(
        parts["ppt/slides/slide16.xml"],
        [
            (
                "按险种分类，涵盖多种核心产品\n突出产品亮点，满足不同客户场景需求",
                "财产、责任、工程险\n核心亮点便于比较",
            ),
            ("75%", "68%"),
            ("场景化方案", "场景方案"),
            (
                "工厂保险方案，保障生产安全\n物流保险方案，降低运输风险",
                "工厂方案覆盖生产与营业中断\n物流方案覆盖仓储和运输",
            ),
            ("52%", "54%"),
            ("地产项目方案", "工程与地产"),
            (
                "提供全面保险保障，助力项目顺利推进",
                "覆盖建设期与运营期主要风险\n工程、物业一并纳入方案",
            ),
            ("81%", "73%"),
        ],
    )
    results = replace_text_nodes(
        parts["ppt/slides/slide17.xml"],
        [
            (
                "赔付率控制在合理范围，低于行业平均水平\n有效提升公司盈利能力",
                "赔付率处于可控区间\n低于演示用的对照水平",
            ),
            (
                "近三年保费收入持续增长，年均增长率达X%\n2024年保费收入预计突破X亿元",
                "近三年保费保持增长\n财产险仍是收入基本盘",
            ),
            (
                "业务覆盖全国多个省份，重点布局大湾区\n在大湾区市场份额逐步提升",
                "以香港为基地服务企业客户\n业务协同延伸至大湾区",
            ),
        ],
    )
    future = replace_text_nodes(
        parts["ppt/slides/slide18.xml"],
        [
            (
                "深化大湾区布局，拓展市场份额\n加强与本地企业合作，提升品牌知名度",
                "深化大湾区企业客户覆盖\n加强本地渠道与品牌合作",
            ),
            (
                "科技赋能保险，推动业务创新升级\n打造智能化保险服务，提升客户体验",
                "把科技用在核保与理赔环节\n让投保、批改、理赔更顺畅",
            ),
            (
                "持续投入科技研发，引入先进科技手段\n优化业务流程，提升运营效率",
                "稳步投入系统与数据能力\n优化流程，缩短服务响应",
            ),
        ],
    )
    contact = replace_text_nodes(
        parts["ppt/slides/slide20.xml"],
        [
            ("地址/电话/官网/二维码", "中环 / 852-0000 / 官网"),
            ("提供详细联系方式，方便客户咨询", "欢迎来电，索取方案与合作联络"),
        ],
    )
    slogan = replace_text_nodes(
        parts["ppt/slides/slide21.xml"],
        [
            ("Slogan", "品牌理念"),
            ("传达公司理念，增强客户信任", "稳健承保，陪伴企业应对风险"),
        ],
    )
    thanks = replace_text_nodes(
        parts["ppt/slides/slide22.xml"],
        [
            ("主讲人：AiPPT", "汇报人：演示"),
            ("时间：20XX.X", "时间：2026.3"),
            ("谢谢大家", "感谢聆听"),
            ("202X", "2026"),
        ],
    )

    section = parts["ppt/slides/slide3.xml"]
    section_rels = parts["ppt/slides/_rels/slide3.xml.rels"]
    chapters = [
        ("企业概况", "01", overview, parts["ppt/slides/_rels/slide14.xml.rels"]),
        ("业务布局", "02", layout, parts["ppt/slides/_rels/slide15.xml.rels"]),
        ("产品优势", "03", products, parts["ppt/slides/_rels/slide16.xml.rels"]),
        ("数据成果", "04", results, parts["ppt/slides/_rels/slide17.xml.rels"]),
        ("未来规划", "05", future, parts["ppt/slides/_rels/slide18.xml.rels"]),
    ]

    ordered: list[tuple[str, str]] = [
        (cover, parts["ppt/slides/_rels/slide1.xml.rels"]),
        (agenda, parts["ppt/slides/_rels/slide6.xml.rels"]),
    ]
    for title, number, body, rels in chapters:
        ordered.append((divider(section, title, number), section_rels))
        ordered.append((body, rels))
    ordered.append((contact, parts["ppt/slides/_rels/slide20.xml.rels"]))
    ordered.append((slogan, parts["ppt/slides/_rels/slide21.xml.rels"]))
    ordered.append((thanks, parts["ppt/slides/_rels/slide22.xml.rels"]))
    return ordered


def load_parts(path: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    raw: dict[str, bytes] = {}
    text: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            data = archive.read(info.filename)
            raw[info.filename] = data
            if info.filename.endswith(".xml") or info.filename.endswith(".rels"):
                text[info.filename] = data.decode("utf-8")
    return raw, text


def write_package(raw: dict[str, bytes], text: dict[str, str], slides: list[tuple[str, str]], dest: Path) -> None:
    slide_names = {f"ppt/slides/slide{i}.xml" for i in range(1, 23)}
    rel_names = {f"ppt/slides/_rels/slide{i}.xml.rels" for i in range(1, 23)}

    count = len(slides)
    ids = "".join(
        f'<p:sldId id="{256 + i}" r:id="rId{3 + i}" />' for i in range(count)
    )
    presentation = re.sub(
        r"<p:sldIdLst>.*?</p:sldIdLst>",
        f"<p:sldIdLst>{ids}</p:sldIdLst>",
        text["ppt/presentation.xml"],
        count=1,
        flags=re.S,
    )

    rel_lines = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>',
    ]
    for i in range(count):
        rel_lines.append(
            f'<Relationship Id="rId{3 + i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i + 1}.xml"/>'
        )
    rel_lines.extend(
        [
            '<Relationship Id="rId25" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/font" Target="fonts/font5.fntdata"/>',
            '<Relationship Id="rId26" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/font" Target="fonts/font2.fntdata"/>',
            '<Relationship Id="rId27" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/font" Target="fonts/font4.fntdata"/>',
            '<Relationship Id="rId28" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/font" Target="fonts/font1.fntdata"/>',
            '<Relationship Id="rId29" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/font" Target="fonts/font3.fntdata"/>',
        ]
    )
    presentation_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        + "\n".join(rel_lines)
        + "\n</Relationships>\n"
    )

    content_types = text["[Content_Types].xml"]
    content_types = re.sub(
        r'\s*<Override PartName="/ppt/slides/slide\d+\.xml"[^>]*/>',
        "",
        content_types,
    )
    overrides = "\n".join(
        f'    <Override PartName="/ppt/slides/slide{i}.xml" ContentType="{SLIDE_CONTENT_TYPE}" />'
        for i in range(1, count + 1)
    )
    content_types = content_types.replace("</Types>", f"{overrides}\n</Types>")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in raw.items():
            if name in slide_names or name in rel_names:
                continue
            if name in {
                "ppt/presentation.xml",
                "ppt/_rels/presentation.xml.rels",
                "[Content_Types].xml",
            }:
                continue
            archive.writestr(name, data)
        archive.writestr("[Content_Types].xml", content_types.encode("utf-8"))
        archive.writestr("ppt/presentation.xml", presentation.encode("utf-8"))
        archive.writestr("ppt/_rels/presentation.xml.rels", presentation_rels.encode("utf-8"))
        for index, (slide_xml, rels_xml) in enumerate(slides, start=1):
            archive.writestr(f"ppt/slides/slide{index}.xml", slide_xml.encode("utf-8"))
            archive.writestr(f"ppt/slides/_rels/slide{index}.xml.rels", rels_xml.encode("utf-8"))
        # raw iteration above may have skipped directories; zipfile does not need them.


def main() -> None:
    if not TEMPLATE.exists():
        raise SystemExit(f"missing template: {TEMPLATE}")
    raw, text = load_parts(TEMPLATE)
    slides = build_slides(text)
    write_package(raw, text, slides, OUTPUT)
    print(f"wrote {OUTPUT} ({len(slides)} slides)")


if __name__ == "__main__":
    main()
