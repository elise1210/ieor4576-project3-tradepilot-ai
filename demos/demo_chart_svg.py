import html
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.skills.chart import run_chart_skill


WIDTH = 960
HEIGHT = 520
MARGIN = 48


def _text(x: int, y: int, value: str, size: int = 16, weight: str = "400", color: str = "#1f2937") -> str:
    safe_value = html.escape(str(value))
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{safe_value}</text>'
    )


def _rect(x: int, y: int, width: int, height: int, fill: str, radius: int = 4) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" />'


def _line(x1: int, y1: int, x2: int, y2: int, color: str = "#64748b", width: int = 2) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" />'


def _polyline(points: list[tuple[int, int]], color: str = "#16a34a", width: int = 4) -> str:
    point_text = " ".join(f"{px},{py}" for px, py in points)
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" />'
    )


def _render_price_panel(chart: dict, x: int, y: int, width: int, height: int) -> list[str]:
    elements = [
        _rect(x, y, width, height, "#f8fafc", 8),
        _text(x + 24, y + 42, chart["title"], 24, "700"),
    ]

    data = chart.get("data", [])
    if len(data) < 2:
        return elements

    prices = [float(point["close"]) for point in data]
    start_price = prices[0]
    end_price = prices[-1]
    min_price = min(prices)
    max_price = max(prices)
    span = max(max_price - min_price, 1.0)

    plot_x1 = x + 88
    plot_x2 = x + width - 56
    plot_y1 = y + height - 86
    plot_y2 = y + 82

    def y_pos(price: float) -> int:
        ratio = (price - min_price) / span
        return int(plot_y1 - ratio * (plot_y1 - plot_y2))

    def x_pos(index: int) -> int:
        if len(data) == 1:
            return plot_x1
        ratio = index / (len(data) - 1)
        return int(plot_x1 + ratio * (plot_x2 - plot_x1))

    line_points = [(x_pos(index), y_pos(price)) for index, price in enumerate(prices)]
    trend_color = "#16a34a" if end_price >= start_price else "#dc2626"

    for grid_index in range(5):
        ratio = grid_index / 4
        grid_y = int(plot_y2 + ratio * (plot_y1 - plot_y2))
        price_value = max_price - ratio * (max_price - min_price)
        elements.extend([
            _line(plot_x1, grid_y, plot_x2, grid_y, "#d7dee8", 1),
            _text(plot_x1 - 64, grid_y + 4, f"${price_value:,.0f}", 11, "400", "#64748b"),
        ])

    elements.extend([
        _line(plot_x1, plot_y2, plot_x1, plot_y1, "#94a3b8", 1),
        _line(plot_x1, plot_y1, plot_x2, plot_y1, "#94a3b8", 1),
        _polyline(line_points, trend_color, 5),
    ])

    for index, point in enumerate(data):
        px, py = line_points[index]
        price = prices[index]
        elements.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{trend_color}" />')

        if index in {0, len(data) - 1}:
            elements.append(_text(px - 36, py - 16, f"${price:,.2f}", 14, "700"))

        label = str(point.get("date", ""))
        if index in {0, len(data) - 1} or len(data) <= 7:
            short_label = label[5:] if len(label) >= 10 else label
            elements.append(_text(px - 18, plot_y1 + 30, short_label, 12, "400", "#475569"))

    change_pct = (end_price - start_price) / start_price if start_price else 0.0
    elements.append(_text(x + 24, y + height - 28, f"7-day change: {change_pct:+.2%}", 18, "700", trend_color))

    return elements


def build_svg(chart_spec: dict) -> str:
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        _rect(0, 0, WIDTH, HEIGHT, "#eef2f7", 0),
        _text(MARGIN, 58, chart_spec["title"], 30, "800", "#111827"),
        _text(MARGIN, 88, "Seven recent trading-day close prices", 15, "400", "#475569"),
    ]

    charts = chart_spec.get("charts", [])
    price_chart = next((chart for chart in charts if chart["id"].endswith("-price")), charts[0])

    elements.extend(_render_price_panel(price_chart, MARGIN, 120, WIDTH - MARGIN * 2, 340))

    elements.append("</svg>")
    return "\n".join(elements)


if __name__ == "__main__":
    sample_evidence = {
        "market": {
            "ticker": "AAPL",
            "current_price": 220.0,
            "start_price_7d": 210.0,
            "trend_7d": 0.0476,
            "trend_label": "upward",
            "volatility": 0.018,
            "ma20": 215.0,
            "history": [
                {"date": "2026-04-23", "close": 210.00},
                {"date": "2026-04-24", "close": 214.20},
                {"date": "2026-04-27", "close": 212.80},
                {"date": "2026-04-28", "close": 216.40},
                {"date": "2026-04-29", "close": 215.30},
                {"date": "2026-04-30", "close": 218.70},
                {"date": "2026-05-01", "close": 220.00},
            ],
        },
    }

    spec = run_chart_skill(
        ticker="AAPL",
        evidence=sample_evidence,
        query="What is AAPL price today?",
        reference_date=date(2026, 5, 1),
    )
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "aapl_chart.svg"
    output_path.write_text(build_svg(spec), encoding="utf-8")

    print(f"Generated chart: {output_path}")
