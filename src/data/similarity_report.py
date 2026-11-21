import difflib
import html
import os


def _highlight(a, b):
    """Markiert Unterschiede zwischen a und b auf Zeichenebene."""
    left, right = [], []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        part_a = html.escape(a[a0:a1])
        part_b = html.escape(b[b0:b1])
        if op == "equal":
            left.append(part_a)
            right.append(part_b)
        else:
            if part_a:
                left.append(f"<del>{part_a}</del>")
            if part_b:
                right.append(f"<ins>{part_b}</ins>")
    return "".join(left), "".join(right)


def write_similarity_report(texts_to_check, highest_similarity, most_similar_text,
                            similarity_threshold, output_folder, split_name,
                            show_from=0.8):
    """Schreibt einen HTML-Bericht und eine sortierte CSV.

    Enthalten sind alle Paare ab `show_from`, also auch knapp behaltene
    Prompts unterhalb der Schwelle (nützlich für die manuelle Prüfung).
    """
    import csv

    rows = [
        (float(sim), text, match)
        for text, sim, match in zip(texts_to_check, highest_similarity, most_similar_text)
        if sim >= show_from
    ]
    rows.sort(key=lambda r: r[0], reverse=True)
    n_removed = sum(1 for r in rows if r[0] >= similarity_threshold)

    # Sortierte CSV (für Nachvollziehbarkeit und Auswertung)
    csv_path = os.path.join(output_folder, f"similar_{split_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["similarity", "removed", "prompt", "most_similar_train_prompt"])
        for sim, text, match in rows:
            writer.writerow([f"{sim:.4f}", sim >= similarity_threshold, text, match])

    # HTML-Bericht (zum Lesen)
    body = []
    for i, (sim, text, match) in enumerate(rows, 1):
        removed = sim >= similarity_threshold
        left, right = _highlight(text, match)
        body.append(
            f'<tr class="{"removed" if removed else "kept"}">'
            f"<td>{i}</td><td>{sim:.3f}</td>"
            f'<td>{"entfernt" if removed else "behalten"}</td>'
            f"<td>{left}</td><td>{right}</td></tr>"
        )

    page = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<title>Ähnliche Prompts – {html.escape(split_name)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; vertical-align: top;
          text-align: left; white-space: pre-wrap; }}
th {{ position: sticky; top: 0; background: #f4f4f4; }}
tr.removed td:nth-child(3) {{ color: #b00020; font-weight: 600; }}
tr.kept td:nth-child(3) {{ color: #1b6e2c; }}
del {{ background: #ffd7d5; text-decoration: none; }}
ins {{ background: #d4f7d4; text-decoration: none; }}
td:nth-child(1), td:nth-child(2), td:nth-child(3) {{ white-space: nowrap; }}
</style></head><body>
<h1>Ähnliche Prompts: {html.escape(split_name)}</h1>
<p>Schwelle: {similarity_threshold} · entfernt: {n_removed} ·
knapp behalten ({show_from} bis {similarity_threshold}): {len(rows) - n_removed} ·
geprüft insgesamt: {len(texts_to_check)}</p>
<p><del>rot</del> = nur im geprüften Prompt, <ins>grün</ins> = nur im Trainingsprompt</p>
<table>
<tr><th>#</th><th>Ähnlichkeit</th><th>Status</th><th>Geprüfter Prompt</th><th>Ähnlichster Trainingsprompt</th></tr>
{"".join(body)}
</table></body></html>"""

    html_path = os.path.join(output_folder, f"similar_{split_name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(page)
    return html_path, csv_path
