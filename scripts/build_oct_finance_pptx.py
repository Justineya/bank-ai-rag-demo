#!/usr/bin/env python3
"""Fill the September finance-report template with the October workbook.

The sample deck has no chart parts and no placeholders. Each page is a
designed slide: a title, a table, and a short comment. This script copies
that package and replaces the text inside existing runs, so fonts, colours,
column widths, and positions stay as designed.

Where the October workbook has one extra line that the September table needs
in order to add up, the line is inserted inside the same table frame and the
row heights are scaled to fit. Lines the template never showed (the longer
P&L bridge, two extra combined-ratio classes) stay off the page, and the
total row still uses the workbook total.
"""

from __future__ import annotations

import copy
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "2026-09-finance-template.pptx"
OUTPUT = ROOT / "samples" / "2026-10-finance-report.pptx"

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"

ET.register_namespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
ET.register_namespace("p", "http://schemas.openxmlformats.org/presentationml/2006/main")
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")


def money(value, digits: int = 1) -> str:
    number = round(float(value), digits)
    if digits == 0:
        return f"{int(round(number)):,}"
    return f"{number:,.{digits}f}"


def pct(value) -> str:
    if value is None or value == "":
        return "—"
    number = round(float(value), 1)
    if abs(number - 100) < 0.05:
        return "100%"
    if abs(number) >= 100 and abs(number - round(number)) < 0.05:
        return f"{int(round(number))}%"
    return f"{number:.1f}%"


def points(value) -> str:
    return f"{round(float(value), 1):.1f}"


def shape_paras(root: ET.Element):
    found = []
    for sp in root.iter(f"{P}sp"):
        body = sp.find(f"{P}txBody")
        if body is None:
            continue
        paras = []
        for para in body.findall(f"{A}p"):
            nodes = list(para.iter(f"{A}t"))
            text = "".join(node.text or "" for node in nodes)
            if text:
                paras.append((nodes, text))
        if paras:
            found.append(paras)
    return found


def replace_shape(root: ET.Element, old: str, new: str) -> None:
    for paras in shape_paras(root):
        current = "\n".join(text for _, text in paras)
        if current != old:
            continue
        if len(paras) == 1:
            nodes = paras[0][0]
            nodes[0].text = new
            for extra in nodes[1:]:
                extra.text = ""
            return
        lines = new.split("\n")
        if len(lines) != len(paras):
            raise SystemExit(f"paragraph count mismatch for {old[:40]!r}")
        for (nodes, _), line in zip(paras, lines):
            nodes[0].text = line
            for extra in nodes[1:]:
                extra.text = ""
        return
    raise SystemExit(f"shape text not found: {old[:60]!r}")


def tables(root: ET.Element):
    return list(root.iter(f"{A}tbl"))


def insert_before_last(tbl: ET.Element, prototype_index: int = 1) -> None:
    rows = tbl.findall(f"{A}tr")
    clone = copy.deepcopy(rows[prototype_index])
    children = list(tbl)
    tbl.insert(children.index(rows[-1]), clone)


def fit_row_heights(tbl: ET.Element, total: int) -> None:
    rows = tbl.findall(f"{A}tr")
    base, remainder = divmod(total, len(rows))
    for index, row in enumerate(rows):
        row.set("h", str(base + (1 if index < remainder else 0)))


def paint_bands(tbl: ET.Element) -> None:
    rows = tbl.findall(f"{A}tr")
    for index, row in enumerate(rows):
        if index == 0:
            continue
        color = "FFFFFF" if index % 2 == 1 else "F7F7F7"
        for cell in row.findall(f"{A}tc"):
            props = cell.find(f"{A}tcPr")
            if props is None:
                continue
            paint = props.find(f"{A}solidFill/{A}srgbClr")
            if paint is not None:
                paint.set("val", color)


def restyle_data_rows(tbl: ET.Element, color: str, bold: str) -> None:
    """The sample paints its last table row orange and bold. On the work-status
    page that row is another task, not a total, so every data row uses the
    same dark colour as the first task."""
    for row in tbl.findall(f"{A}tr")[1:]:
        for run in row.iter(f"{A}rPr"):
            run.set("b", bold)
            paint = run.find(f"{A}solidFill/{A}srgbClr")
            if paint is not None:
                paint.set("val", color)


def fill_table(tbl: ET.Element, matrix: list[list[str]]) -> None:
    rows = tbl.findall(f"{A}tr")
    if len(rows) != len(matrix):
        raise SystemExit(f"row count {len(rows)} != {len(matrix)}")
    for row, values in zip(rows, matrix):
        cells = row.findall(f"{A}tc")
        if len(cells) != len(values):
            raise SystemExit(f"col count {len(cells)} != {len(values)} on {values[0]!r}")
        for cell, value in zip(cells, values):
            nodes = list(cell.iter(f"{A}t"))
            if not nodes:
                raise SystemExit(f"no text node in cell {value!r}")
            nodes[0].text = value
            for extra in nodes[1:]:
                extra.text = ""


def prepare(tbl: ET.Element, extra_rows: int) -> None:
    total = sum(int(row.get("h")) for row in tbl.findall(f"{A}tr"))
    for _ in range(extra_rows):
        insert_before_last(tbl)
    fit_row_heights(tbl, total)
    paint_bands(tbl)


def build(slides: dict[str, str]) -> dict[str, bytes]:
    roots = {name: ET.fromstring(xml) for name, xml in slides.items()}

    s1 = roots["ppt/slides/slide1.xml"]
    replace_shape(s1, "2026年9月財務情況匯報", "2026年10月財務情況匯報")
    replace_shape(s1, "2026年10月9日", "2026年11月6日")

    s2 = roots["ppt/slides/slide2.xml"]
    replace_shape(s2, "萬港元，%。當月為2026年9月，累計為1–9月。", "萬港元，%。當月為2026年10月，累計為1–10月。")
    replace_shape(
        s2,
        "9月當月錄得稅後盈利39.6萬港元，環比減少76.2萬。保險收入1,520.4萬，環比增加86.2萬；保險服務業績184.6萬。",
        "10月當月錄得稅後盈利154.2萬港元，環比增加105.2萬。保險收入1,720.4萬，環比增加121.5萬；保險服務業績263.6萬。",
    )
    replace_shape(
        s2,
        "1–9月累計稅後利潤1,299.1萬。保險收入14,760.2萬，同比增加1,210.5萬；投資收益1,093.9萬。",
        "1–10月累計稅後利潤1,453.3萬。保險收入16,480.6萬，同比增加1,488.2萬；投資收益1,180.4萬。",
    )
    fill_table(
        tables(s2)[0],
        [
            ["項目", "10月當月", "環比增減", "環比%", "1–10月", "同比增減", "同比%", "修訂預算", "完成%"],
            ["保險收入", money(1720.4), money(121.5), pct(7.6), money(16480.6), money(1488.2), pct(9.9), money(21000), pct(78.5)],
            ["保險服務費用", money(-1288.6), money(-175.8), pct(15.8), money(-12890.4), money(-980.6), pct(8.2), money(-16000), pct(80.6)],
            ["再保險合約費用淨額", money(-168.2), money(5.2), pct(-3), money(-1488.5), money(-42.1), pct(2.9), money(-2100), pct(70.9)],
            ["保險服務業績", money(263.6), money(-49.1), pct(-15.7), money(2101.7), money(465.5), pct(28.5), money(2900), pct(72.5)],
            ["減：其他管理費用", money(-186.4), money(-13.7), pct(7.9), money(-1766.2), money(-188.5), pct(11.9), money(-2700), pct(65.4)],
            ["加：投資收益", money(86.5), money(185.7), pct(-187.2), money(1180.4), money(-420.8), pct(-26.3), money(2500), pct(47.2)],
            ["稅前利潤", money(176.6), money(115.9), pct(191), money(1622.1), money(159.2), pct(10.9), money(2154), pct(75.3)],
            ["稅後利潤", money(154.2), money(105.2), pct(214.7), money(1453.3), money(136.6), pct(10.4), money(1874), pct(77.6)],
        ],
    )

    s3 = roots["ppt/slides/slide3.xml"]
    replace_shape(
        s3,
        "香港佔比41.1%（2025：40.2%），澳門佔比58.9%（2025：59.8%）。\n直接業務佔比61.7%（2025：75.6%），分入業務佔比38.3%（2025：24.4%）。\n同比增加主要來自虛構的北灣銀行團體醫療及南灣勞務險，保費合計約2,050萬港元。",
        "香港佔比41.2%（2025：40.4%），澳門佔比58.8%（2025：59.6%）。\n直接業務佔比60.2%（2025：73.7%），分入業務佔比39.8%（2025：26.3%）。\n同比增加主要來自虛構的北灣銀行團體醫療及南灣勞務險，保費合計約2,280萬港元。",
    )
    fill_table(
        tables(s3)[0],
        [
            ["項目", "2026年1–10月", "2025年1–10月", "同比增減", "同比%", "修訂預算", "完成%"],
            ["毛保費合計", money(18640.5), money(16120.8), money(2519.7), pct(15.6), money(23000), pct(81)],
            ["香港業務", money(7688.4), money(6520.2), money(1168.2), pct(17.9), money(9800), pct(78.5)],
            ["澳門業務", money(10952.1), money(9600.6), money(1351.5), pct(14.1), money(13200), pct(83)],
            ["直接業務", money(11220.3), money(11880.4), money(-660.1), pct(-5.6), "—", "—"],
            ["分入業務", money(7420.2), money(4240.4), money(3179.8), pct(75), "—", "—"],
        ],
    )

    s4 = roots["ppt/slides/slide4.xml"]
    replace_shape(
        s4,
        "截至9月底完成修訂預算70.3%，落後時間進度約4.7個百分點。汽車險、財產全險、其他險下跌。意外及健康險、勞工險增幅主要來自分入。",
        "截至10月底完成修訂預算77.7%，落後時間進度約5.6個百分點。汽車險、財產全險、其他險下跌。意外及健康險、勞工險增幅主要來自分入。",
    )
    fill_table(
        tables(s4)[0],
        [
            ["險種", "2026年1–10月", "2025年1–10月", "同比增減", "同比%", "修訂預算", "完成%"],
            ["汽車險", money(1688.2), money(2310.5), money(-622.3), pct(-26.9), money(2100), pct(80.4)],
            ["意外及健康險", money(1488.6), money(420.4), money(1068.2), pct(254.1), money(1600), pct(93)],
            ["勞工險", money(1622.4), money(210.8), money(1411.6), pct(669.6), money(1750), pct(92.7)],
            ["火險", money(4920.5), money(4888.2), money(32.3), pct(0.7), money(6800), pct(72.4)],
            ["財產全險", money(4288.7), money(4510.6), money(-221.9), pct(-4.9), money(6400), pct(67)],
            ["其他險", money(402.1), money(510.3), money(-108.2), pct(-21.2), money(850), pct(47.3)],
            ["合同分入", money(4230), money(3270), money(960), pct(29.4), money(4500), pct(94)],
            ["合計", money(18640.5), money(16120.8), money(2519.7), pct(15.6), money(24000), pct(77.7)],
        ],
    )

    s5 = roots["ppt/slides/slide5.xml"]
    replace_shape(s5, "2026年9月底", "2026年10月底")
    replace_shape(s5, "2025年9月底", "2025年10月底")
    replace_shape(
        s5,
        "佔比最大為火險26.4%（2025：30.2%），其次財產全險23.2%，合同分入23.1%。",
        "佔比最大為火險26.4%（2025：30.3%），其次財產全險23.0%，合同分入22.7%。",
    )
    fill_table(
        tables(s5)[0],
        [
            ["險種", "佔比"],
            ["火險", pct(26.4)],
            ["財產全險", pct(23.0)],
            ["合同分入", pct(22.7)],
            ["汽車險", pct(9.1)],
            ["勞工險", pct(8.7)],
            ["意外及健康險", pct(8.0)],
            ["其他險", pct(2.1)],
        ],
    )
    fill_table(
        tables(s5)[1],
        [
            ["險種", "佔比"],
            ["火險", pct(30.3)],
            ["財產全險", pct(28.0)],
            ["合同分入", pct(20.3)],
            ["汽車險", pct(14.3)],
            ["勞工險", pct(1.3)],
            ["意外及健康險", pct(2.6)],
            ["其他險", pct(3.2)],
        ],
    )

    s6 = roots["ppt/slides/slide6.xml"]
    replace_shape(s6, "單位：萬港元。累計6,940.6萬，同比增加1,060.2萬。", "單位：萬港元。累計7,688.4萬，同比增加1,168.2萬。")
    replace_shape(
        s6,
        "合同分入佔55.8%（2025：47.8%），汽車險佔20.6%，意外及健康險佔12.8%。",
        "合同分入佔56.1%（2025：48.8%），汽車險佔20.5%，意外及健康險佔12.9%。",
    )
    fill_table(
        tables(s6)[0],
        [
            ["險種", "2026年1–10月", "2025年1–10月", "同比增減", "同比%", "佔比"],
            ["汽車險", money(1576.1), money(2236), money(-659.9), pct(-29.5), pct(20.5)],
            ["意外及健康險", money(991.4), money(104.3), money(887.1), pct(850.5), pct(12.9)],
            ["勞工險", money(98.2), money(88), money(10.2), pct(11.6), pct(1.3)],
            ["火險", money(210.5), money(240.2), money(-29.7), pct(-12.4), pct(2.7)],
            ["財產全險", money(420.8), money(480.1), money(-59.3), pct(-12.4), pct(5.5)],
            ["其他險", money(80.4), money(188.6), money(-108.2), pct(-57.4), pct(1.0)],
            ["合同分入", money(4311), money(3183), money(1128), pct(35.4), pct(56.1)],
            ["合計", money(7688.4), money(6520.2), money(1168.2), pct(17.9), "100%"],
        ],
    )

    s7 = roots["ppt/slides/slide7.xml"]
    fill_table(
        tables(s7)[0],
        [
            ["險種", "2026年1–10月", "2025年1–10月", "同比增減", "同比%", "佔比"],
            ["汽車險", money(1576.1), money(2236), money(-659.9), pct(-29.5), pct(46.7)],
            ["意外及健康險", money(420.6), money(90.2), money(330.4), pct(366.3), pct(12.5)],
            ["勞工險", money(98.2), money(88), money(10.2), pct(11.6), pct(2.9)],
            ["火險", money(210.5), money(240.2), money(-29.7), pct(-12.4), pct(6.2)],
            ["財產全險", money(420.8), money(480.1), money(-59.3), pct(-12.4), pct(12.5)],
            ["其他險", money(80.4), money(120.4), money(-40), pct(-33.2), pct(2.4)],
            ["合計", money(2806.6), money(3254.9), money(-448.3), pct(-13.8), "100%"],
        ],
    )

    s8 = roots["ppt/slides/slide8.xml"]
    tbl = tables(s8)[0]
    prepare(tbl, 1)
    fill_table(
        tbl,
        [
            ["險種", "2026年1–10月", "2025年1–10月", "同比增減", "同比%", "佔比"],
            ["意外及健康險（臨分）", money(570.8), money(14.1), money(556.7), pct(3948.2), pct(11.7)],
            ["其他臨分", money(0), money(68.2), money(-68.2), pct(-100), pct(0)],
            ["合同分入", money(4311), money(3183), money(1128), pct(35.4), pct(88.3)],
            ["合計", money(4881.8), money(3265.3), money(1616.5), pct(49.5), "100%"],
        ],
    )

    s9 = roots["ppt/slides/slide9.xml"]
    replace_shape(s9, "單位：萬港元。累計9,939.6萬，同比增加1,199.5萬。", "單位：萬港元。累計10,952.1萬，同比增加1,351.5萬。")
    replace_shape(
        s9,
        "火險為最大險種，佔42.9%（2025：48.0%），其次財產全險35.6%，勞工險13.3%。",
        "火險為最大險種，佔43.0%（2025：48.4%），其次財產全險35.3%，勞工險13.9%。",
    )
    fill_table(
        tables(s9)[0],
        [
            ["險種", "2026年1–10月", "2025年1–10月", "同比增減", "同比%", "佔比"],
            ["汽車險", money(112.1), money(74.5), money(37.6), pct(50.5), pct(1.0)],
            ["意外及健康險", money(497.2), money(316.1), money(181.1), pct(57.3), pct(4.5)],
            ["勞工險", money(1524.2), money(122.8), money(1401.4), pct(1141.2), pct(13.9)],
            ["火險", money(4710), money(4648), money(62), pct(1.3), pct(43.0)],
            ["財產全險", money(3867.9), money(4030.5), money(-162.6), pct(-4.0), pct(35.3)],
            ["其他險", money(240.7), money(408.7), money(-168), pct(-41.1), pct(2.2)],
            ["合計", money(10952.1), money(9600.6), money(1351.5), pct(14.1), "100%"],
        ],
    )

    s10 = roots["ppt/slides/slide10.xml"]
    fill_table(
        tables(s10)[0],
        [
            ["險種", "2026年1–10月", "2025年1–10月", "同比增減", "同比%", "佔比"],
            ["汽車險", money(112.1), money(74.5), money(37.6), pct(50.5), pct(1.3)],
            ["意外及健康險", money(360.4), money(250.2), money(110.2), pct(44.0), pct(4.3)],
            ["勞工險", money(210.5), money(122.8), money(87.7), pct(71.4), pct(2.5)],
            ["火險", money(4710), money(4648), money(62), pct(1.3), pct(56.0)],
            ["財產全險", money(3867.9), money(4030.5), money(-162.6), pct(-4.0), pct(46.0)],
            ["其他險", money(152.8), money(240.5), money(-87.7), pct(-36.5), pct(1.8)],
            ["合計", money(8413.7), money(9366.5), money(-952.8), pct(-10.2), "100%"],
        ],
    )

    s11 = roots["ppt/slides/slide11.xml"]
    fill_table(
        tables(s11)[0],
        [
            ["險種", "2026年1–10月", "2025年1–10月", "同比增減", "同比%", "佔比"],
            ["意外及健康險", money(136.8), money(65.9), money(70.9), pct(107.6), pct(5.4)],
            ["勞工險", money(1313.7), money(0), money(1313.7), "—", pct(51.8)],
            ["其他險", money(87.9), money(168.2), money(-80.3), pct(-47.7), pct(3.5)],
            ["合計", money(2538.4), money(234.1), money(2304.3), pct(984.3), "100%"],
        ],
    )

    s12 = roots["ppt/slides/slide12.xml"]
    replace_shape(s12, "單位：萬港元。2026年1–9月。", "單位：萬港元。2026年1–10月。")
    replace_shape(
        s12,
        "累計已承付申索淨額4,388.6萬，同比增加1,170.8萬，主要是勞工險、意外及健康險及合同分入。\n香港3,625.3萬，同比增加788.2萬。澳門763.3萬，同比增加382.6萬。",
        "累計已承付申索淨額4,845.1萬，同比增加1,372.7萬，主要是勞工險、意外及健康險及合同分入。\n香港3,945.3萬，同比增加899.3萬。澳門899.8萬，同比增加473.4萬。",
    )
    tbl = tables(s12)[0]
    prepare(tbl, 1)
    fill_table(
        tbl,
        [
            ["險種", "香港", "香港同比", "澳門", "澳門同比", "合計", "合計同比", "合計同比%"],
            ["汽車險", money(1880.2), money(96.4), money(52.8), money(18.6), money(1933), money(115), pct(6.3)],
            ["意外及健康險", money(610.5), money(420.2), money(160.4), money(22.1), money(770.9), money(442.3), pct(134.6)],
            ["勞工險", money(-80.4), money(40.2), money(820.6), money(640.5), money(740.2), money(680.7), pct(1144.4)],
            ["火險", money(8.2), money(-12.4), money(-98.6), money(-120.4), money(-90.4), money(-132.8), pct(-312.7)],
            ["財產全險", money(22.6), money(6.4), money(-40.2), money(-88.6), money(-17.6), money(-82.2), pct(-127.3)],
            ["其他險", money(-6.2), money(-40.1), money(4.8), money(1.2), money(-1.4), money(-38.9), pct(-96.5)],
            ["合同分入", money(1510.4), money(388.6), money(0), money(0), money(1510.4), money(388.6), pct(34.6)],
            ["合計", money(3945.3), money(899.3), money(899.8), money(473.4), money(4845.1), money(1372.7), pct(39.5)],
        ],
    )

    s13 = roots["ppt/slides/slide13.xml"]
    replace_shape(
        s13,
        "截至9月底，綜合成本率85.3%，比去年同期86.8%下降1.5個百分點。\n淨佣金率47.2%，比預算48.5%低1.3個百分點。\n賠付率32.6%，比預算29.0%高3.6個百分點。費用率5.5%，比預算5.8%低0.3個百分點。",
        "截至10月底，綜合成本率86.4%，比去年同期87.2%下降0.8個百分點。\n淨佣金率46.8%，比預算48.5%低1.7個百分點。\n賠付率34.2%，比預算29.0%高5.2個百分點。費用率5.4%，比預算5.8%低0.4個百分點。",
    )
    fill_table(
        tables(s13)[0],
        [
            ["險種", "淨佣金率", "賠付率", "費用率", "綜合成本率", "去年綜合", "變動(百分點)"],
            ["意外及健康險", pct(9.2), pct(72.4), pct(8.1), pct(89.7), pct(89.1), points(0.6)],
            ["汽車險", pct(27.6), pct(104.2), pct(9.4), pct(141.2), pct(135.1), points(6.1)],
            ["財產險", pct(68.4), pct(1.8), pct(5.4), pct(75.6), pct(81.2), points(-5.6)],
            ["勞工險", pct(8.1), pct(96.4), pct(12.6), pct(117.1), pct(49.1), points(68)],
            ["合同分入", pct(28.6), pct(46.2), pct(0.3), pct(75.1), pct(71.7), points(3.4)],
            ["合計", pct(46.8), pct(34.2), pct(5.4), pct(86.4), pct(87.2), points(-0.8)],
        ],
    )

    s14 = roots["ppt/slides/slide14.xml"]
    fill_table(
        tables(s14)[0],
        [
            ["險種", "毛保費", "保險收入", "保險服務費用", "再保費用淨額", "保險服務業績", "同比%"],
            ["意外及健康險", money(1488.6), money(1024.5), money(910.2), money(22.4), money(91.9), pct(110.3)],
            ["汽車險", money(1688.2), money(2010.4), money(2688.6), money(230.5), money(-908.7), pct(10.8)],
            ["財產險", money(9209.2), money(8620.8), money(5710.4), money(980.2), money(1930.2), pct(27)],
            ["勞工險", money(1622.4), money(710.6), money(742.8), money(102.4), money(-134.6), pct(-277.3)],
            ["合同分入", money(4230), money(3825.8), money(2775.6), money(0), money(1050.2), pct(21.6)],
            ["合計", money(18547.6), money(16480.6), money(12741.8), money(1379.7), money(2101.7), pct(28.5)],
        ],
    )

    s15 = roots["ppt/slides/slide15.xml"]
    replace_shape(
        s15,
        "累計投資收益1,093.9萬。ETF/FVPL未實現虧損210.4萬。股票當年估值減少720.0萬（虛構）。",
        "累計投資收益1,180.4萬。ETF/FVPL未實現虧損246.8萬。股票當年估值減少860.0萬（虛構）。",
    )
    fill_table(
        tables(s15)[0],
        [
            ["項目", "投資規模", "2026年1–10月", "2025年1–10月", "同比增減", "同比%", "回報率"],
            ["銀行存款", money(24100, 0), money(352.6), money(498.2), money(-145.6), pct(-29.2), "1.76%"],
            ["債券", money(16880, 0), money(512.4), money(310.5), money(201.9), pct(65), "3.64%"],
            ["股票-FVOCI", money(11540, 0), money(468.2), money(360.4), money(107.8), pct(29.9), "4.87%"],
            ["股票-ETF/FVPL", money(1520, 0), money(-246.8), money(12.4), money(-259.2), pct(-2090.3), "-19.48%"],
            ["貨幣基金", money(980, 0), money(18.6), money(4.2), money(14.4), pct(342.9), "2.28%"],
            ["投資物業", money(11950, 0), money(75.4), money(415.5), money(-340.1), pct(-81.9), "0.76%"],
            ["合計", money(66970, 0), money(1180.4), money(1601.2), money(-420.8), pct(-26.3), "2.12%"],
        ],
    )

    s16 = roots["ppt/slides/slide16.xml"]
    replace_shape(s16, "9月底澳門業務償付比率為 226%", "10月底澳門業務償付比率為 218%")
    replace_shape(s16, "環比上月減少 6 個百分點", "環比上月減少 8 個百分點")

    s17 = roots["ppt/slides/slide17.xml"]
    replace_shape(
        s17,
        "截至9月底應收保費6,647.2萬，環比減少110.4萬。單位：萬港元。",
        "截至10月底應收保費6,520.4萬，環比減少126.8萬。單位：萬港元。",
    )
    fill_table(
        tables(s17)[0],
        [
            ["渠道", "拖欠應收", "總應收", "拖欠比率", "預警值", "行動值"],
            ["直接（逾期>60天）", money(18.6), money(310.4), pct(6), "20%", "25%"],
            ["臨分分入（逾期>90天）", money(246.2), money(1188.5), pct(20.7), "40%", "50%"],
            ["合同分入（逾期>90天）", money(1420.8), money(5021.5), pct(28.3), "40%", "50%"],
            ["合計", money(1685.6), money(6520.4), pct(25.9), "—", "—"],
        ],
    )

    s18 = roots["ppt/slides/slide18.xml"]
    replace_shape(
        s18,
        "截至9月底應收保費2,796.8萬，環比減少64.2萬。單位：萬港元。",
        "截至10月底應收保費2,710.6萬，環比減少86.2萬。單位：萬港元。",
    )
    fill_table(
        tables(s18)[0],
        [
            ["渠道", "拖欠應收", "總應收", "拖欠比率", "預警值", "行動值"],
            ["直接（逾期>60天）", money(98.4), money(1390.2), pct(7.1), "20%", "25%"],
            ["臨分分入（逾期>90天）", money(18.6), money(1320.4), pct(1.4), "40%", "50%"],
            ["合計", money(117), money(2710.6), pct(4.3), "—", "—"],
        ],
    )

    s19 = roots["ppt/slides/slide19.xml"]
    replace_shape(
        s19,
        "單位：港元。截至2026年9月底。公司名為虛構代號，負數表示應付。",
        "單位：港元。截至2026年10月底。公司名為虛構代號，負數表示應付。",
    )
    tbl = tables(s19)[0]
    prepare(tbl, 2)
    fill_table(
        tbl,
        [
            ["公司", ">1年", ">180天", ">90天", ">30天", "current", "總計"],
            ["模擬再保A", money(-206000, 0), money(-42000, 0), money(-195000, 0), money(-24000, 0), money(-4000, 0), money(-471000, 0)],
            ["模擬經紀B", money(-57000, 0), money(-58000, 0), money(-130000, 0), money(-35000, 0), money(-8000, 0), money(-288000, 0)],
            ["模擬財險C", money(-9000, 0), money(-31000, 0), money(-150000, 0), money(-22000, 0), money(0, 0), money(-212000, 0)],
            ["模擬再保D", money(-1090000, 0), money(-36000, 0), money(-640000, 0), money(-54000, 0), money(0, 0), money(-1820000, 0)],
            ["模擬經紀E", money(-248000, 0), money(-200000, 0), money(-228000, 0), money(-45000, 0), money(-42000, 0), money(-763000, 0)],
            ["模擬壽險F", money(0, 0), money(0, 0), money(0, 0), money(0, 0), money(-12600, 0), money(-12600, 0)],
            ["模擬再保G", money(-780000, 0), money(-175000, 0), money(-193000, 0), money(-104000, 0), money(-18000, 0), money(-1270000, 0)],
            ["總計", money(-2390000, 0), money(-542000, 0), money(-1536000, 0), money(-284000, 0), money(-84600, 0), money(-4836600, 0)],
        ],
    )

    s20 = roots["ppt/slides/slide20.xml"]
    replace_shape(s20, "一、上月工作（2026年9月1日至9月30日）", "一、上月工作（2026年10月1日至10月31日）")
    replace_shape(s20, "二、本月工作計畫（2026年10月1日至10月31日）", "二、本月工作計畫（2026年11月1日至11月30日）")
    for tbl, matrix in zip(
        tables(s20),
        [
            [
                ["序號", "主要工作內容", "主要工作成效"],
                ["1", "測試南灣銀行網上銀行付款", "已完成一筆小額測試入賬"],
                ["2", "跟進代表處註銷文件", "已提交補正材料，等待回執"],
                ["3", "拜訪合作經紀（虛構）", "確認明年合同分入意向保費區間"],
                ["4", "核對10月投資估值", "ETF浮虧已入賬，待下月複盤限額"],
            ],
            [
                ["序號", "主要工作內容", "主要工作成效"],
                ["1", "正式啟用南灣銀行網上銀行", "計畫完成權限與限額設置"],
                ["2", "關閉代表處剩餘賬戶", "計畫取得註銷證明"],
                ["3", "編制2027年預算修訂稿", "計畫提交管理層討論"],
                ["4", "復查汽車險定價", "計畫下調高賠付組合的續保折扣"],
            ],
        ],
    ):
        prepare(tbl, 1)
        restyle_data_rows(tbl, "333F50", "0")
        fill_table(tbl, matrix)

    s21 = roots["ppt/slides/slide21.xml"]
    replace_shape(
        s21,
        "截至9月底，承保虧損688萬港元。私家車全保盈利28.4萬，第三者責任虧損820.6萬。電單車盈利142.2萬。單位：萬港元。",
        "截至10月底，承保虧損742.6萬港元。私家車全保盈利36.2萬，第三者責任虧損880.4萬。電單車盈利168.5萬。單位：萬港元。",
    )
    fill_table(
        tables(s21)[0],
        [
            ["車種", "保障", "毛保費", "滿期保費", "淨佣金", "淨賠付", "承保收入"],
            ["私家車", "全保", money(92.4), money(110.8), money(28.6), money(52.4), money(36.2)],
            ["私家車", "第三者", money(688.5), money(860.4), money(250.2), money(1490.2), money(-880.4)],
            ["電單車", "第三者", money(560.4), money(548.6), money(166.8), money(248.6), money(168.5)],
            ["其他", "第三者", money(7.4), money(7.8), money(1.6), money(4.2), money(1.9)],
            ["合計", "—", money(1349.9), money(1529.2), money(447.5), money(1796.6), money(-742.6)],
        ],
    )

    s22 = roots["ppt/slides/slide22.xml"]
    replace_shape(s22, "2026年9月", "2026年10月")

    out = {}
    for name, root in roots.items():
        body = ET.tostring(root, encoding="utf-8")
        out[name] = b"<?xml version='1.0' encoding='UTF-8' standalone='yes'?>\n" + body
    return out


def main() -> None:
    if not TEMPLATE.exists():
        raise SystemExit(f"missing template: {TEMPLATE}")
    raw: dict[str, bytes] = {}
    text: dict[str, str] = {}
    with zipfile.ZipFile(TEMPLATE) as archive:
        for info in archive.infolist():
            data = archive.read(info.filename)
            raw[info.filename] = data
            if info.filename.startswith("ppt/slides/slide") and info.filename.endswith(".xml"):
                text[info.filename] = data.decode("utf-8")
    updated = build(text)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in raw.items():
            archive.writestr(name, updated.get(name, data))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
