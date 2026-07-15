"""Run MESIE multi-domain analysis suites and write deliverables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesie.analysis.domain_suites import run_all_suites

OUT_DIR = ROOT / "deliverables" / "suites"
REPORT_JSON = ROOT / "deliverables" / "MESIE_Multi_Domain_Suite_Report.json"
REPORT_MD = ROOT / "deliverables" / "MESIE_Multi_Domain_Suite_Report.md"


def _write_markdown(data: dict) -> None:
    lines = [
        "# MESIE Multi-Domain Analysis Report",
        "",
        f"*Generated {data['generated_at']} — engine v{data['version']}*",
        "",
        "## Executive summary",
        "",
    ]
    for i, s in enumerate(data["executive_summary"], 1):
        lines.append(f"{i}. {s}")
    lines.extend(["", "## Domain suites", ""])

    for suite in data["suites"]:
        lines.append(f"### {suite['title']} (`{suite['domain']}`)")
        lines.append("")
        lines.append(f"**Conclusion:** {suite['plain_conclusion']}")
        lines.append("")
        lines.append(f"- Records: {', '.join(suite['records_used'][:6])}")
        lines.append(f"- Runtime: {suite['elapsed_ms']:.0f} ms")
        if suite.get("metrics"):
            lines.append("- Metrics:")
            for k, v in suite["metrics"].items():
                lines.append(f"  - {k}: {v}")
        if suite.get("matches"):
            lines.append("- Top matches:")
            for m in suite["matches"][:4]:
                if "candidate_id" in m:
                    lines.append(f"  - `{m['candidate_id']}` → {m['score']}")
                else:
                    lines.append(f"  - `{m.get('a')}` vs `{m.get('b')}` → {m['score']}")
        if suite.get("power_data"):
            pd = suite["power_data"]
            if pd.get("hz_ladder_tiers"):
                lines.append("- Hz ladder (power/comm):")
                for t in pd["hz_ladder_tiers"][:4]:
                    lines.append(f"  - Tier {t['tier']} {t['name']}: {t['center_Hz']:.2e} Hz")
        if suite.get("orbital_data"):
            od = suite["orbital_data"]
            if od.get("satellite_link_budgets"):
                lines.append("- Satellite nodes:")
                for lb in od["satellite_link_budgets"][:3]:
                    lines.append(
                        f"  - {lb['node']}: contact {lb['max_contact_s']}s, loss {lb['path_loss_dB']} dB"
                    )
            fwd = od.get("orbital_50d", {}).get("forward_model", {})
            if fwd.get("alert_days_anomaly_elevated"):
                lines.append(f"- Forward alert days: {fwd['alert_days_anomaly_elevated'][:12]}")
        lines.append("")

    lines.extend(
        [
            "## How to re-run",
            "",
            "```bash",
            "python scripts/run_multi_domain_suites.py",
            "python scripts/orbital_edge_50d_analysis.py",
            "```",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = run_all_suites()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_markdown(data)

    for suite in data["suites"]:
        path = OUT_DIR / f"{suite['domain']}_suite.json"
        path.write_text(json.dumps(suite, indent=2), encoding="utf-8")

    print("=== MESIE Multi-Domain Suites ===\n")
    for suite in data["suites"]:
        print(f"[{suite['domain'].upper()}] {suite['plain_conclusion'][:120]}...")
    print(f"\nTotal time: {data['total_elapsed_ms']:.0f} ms")
    print(f"JSON: {REPORT_JSON}")
    print(f"Markdown: {REPORT_MD}")
    print(f"Per-domain: {OUT_DIR}/")


if __name__ == "__main__":
    main()