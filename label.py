"""Local hand-labelling page (Gradio) for the items from prepare_labels.py.

- aspects: one clause per screen with the pre-labels already set (lexicon aspects + model sentiment);
  confirm with Enter or fix by clicking. Multi-label aspects + "нет аспекта", clause sentiment, sarcasm flag.
- audit:   one test review per screen, blind (no stars, no model answer): негатив / нейтрально / позитив / смешанный.
Every answer is appended to data/labels_{mode}.jsonl at once (last answer per id wins), so a restart resumes
at the first unlabelled item.
Hotkeys (any keyboard layout): 1…0 aspects, − «нет аспекта» / 1…4 classes in audit; Z X C (V) sentiment;
S sarcasm; Enter save and next; Backspace back.
Usage: python label.py aspects|audit [--port 7861]
"""
import argparse
import json
import time
from pathlib import Path

import gradio as gr

from sentiment import ASPECTS

NONE = "нет аспекта"
ASPECT_CHOICES = [*ASPECTS, NONE]
SENT = {"aspects": ["негатив", "нейтрально", "позитив"], "audit": ["негатив", "нейтрально", "позитив", "смешанный"]}
EN = {"негатив": "negative", "нейтрально": "neutral", "позитив": "positive", "смешанный": "mixed"}
RU = {v: k for k, v in EN.items()}

HOTKEYS = """<script>
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA" || (e.target.tagName === "INPUT" && e.target.type === "text")) return;
  if (e.ctrlKey || e.altKey || e.metaKey) return;
  const q = (s) => [...document.querySelectorAll(s)];
  const audit = !!document.querySelector("#mode-audit");
  const digit = e.code.startsWith("Digit") ? (+e.code.slice(5) + 9) % 10 : -1;  // 1→0 … 0→9
  let hit = true;
  if (!audit && digit >= 0) q("#asp input[type=checkbox]")[digit]?.click();
  else if (!audit && (e.code === "Minus" || e.key === "-")) q("#asp input[type=checkbox]")[10]?.click();
  else if (audit && digit >= 0 && digit < 4) q("#sent input[type=radio]")[digit]?.click();
  else if (["KeyZ", "KeyX", "KeyC", "KeyV"].includes(e.code)) q("#sent input[type=radio]")["ZXCV".indexOf(e.code[3])]?.click();
  else if (e.code === "KeyS") q("#sarc input[type=checkbox]")[0]?.click();
  else if (e.code === "Enter") document.querySelector("#save")?.click();
  else if (e.code === "Backspace") document.querySelector("#back")?.click();
  else hit = false;
  if (hit) e.preventDefault();
});
</script>"""


def load(path):
    if not path.exists():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            done[r["id"]] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["aspects", "audit"])
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    items = [json.loads(x) for x in Path(f"data/label_items_{a.mode}.jsonl").read_text(encoding="utf-8").splitlines()]
    out = Path(f"data/labels_{a.mode}.jsonl")
    aspects_mode = a.mode == "aspects"

    def first_open(start=0):
        done = load(out)
        for k in list(range(start, len(items))) + list(range(start)):
            if items[k]["id"] not in done:
                return k
        return min(start, len(items) - 1)

    def show(i):
        done = load(out)
        x = items[i]
        r = done.get(x["id"])
        n = len(done)
        head = f"**{i + 1} / {len(items)}** · размечено {n} из {len(items)}" + (" · ✅ эта уже размечена" if r else "")
        if n == len(items):
            head += " · 🎉 всё размечено, можно закрывать"
        if aspects_mode:
            asp = r["aspects"] if r else (x["pre_aspects"] or [NONE])
            sent = RU[r["label"]] if r else RU[x["pre_label"]]
            body = x["text"]
            ctx = x["review"]
        else:
            asp, sent = [], (RU[r["label"]] if r else None)
            body, ctx = x["text"], ""
        return i, head, body, ctx, gr.update(value=asp), gr.update(value=sent), bool(r and r.get("sarcasm")), ""

    def save(i, asp, sent, sarc):
        # gr.Error stops the chain, so the screen stays on this item
        if aspects_mode and not asp:
            raise gr.Error("Отметьте хотя бы один аспект или «нет аспекта»")
        if not sent:
            raise gr.Error("Выберите тональность")
        rec = {"id": items[i]["id"], "label": EN[sent], "sarcasm": bool(sarc), "ts": round(time.time())}
        if aspects_mode:
            rec["aspects"] = [x for x in asp if x != NONE]
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return first_open(i + 1), ""

    with gr.Blocks(title="Разметка отзывов") as demo:
        gr.HTML(f'<span id="mode-{a.mode}"></span>')
        gr.Markdown("## Разметка аспектов" if aspects_mode else
                    "## Аудит «нейтрального»: какая тональность у отзыва на самом деле? (звёзды и ответ модели скрыты)")
        idx = gr.State(first_open())
        head = gr.Markdown()
        body = gr.Markdown(elem_id="phrase")
        ctx = gr.Textbox(label="Весь отзыв (контекст)", lines=4, interactive=False, visible=aspects_mode)
        asp = gr.CheckboxGroup(ASPECT_CHOICES, label="Аспекты фразы (1…0, «−» — нет аспекта)", elem_id="asp",
                               visible=aspects_mode)
        last_asp = gr.State([])
        sent = gr.Radio(SENT[a.mode], label="Тональность фразы (Z X C)" if aspects_mode else
                        "Тональность отзыва (1–4 или Z X C V)", elem_id="sent")
        sarc = gr.Checkbox(label="Сарказм / ирония (S)", elem_id="sarc")
        msg = gr.Markdown()
        with gr.Row():
            back = gr.Button("← Назад (Backspace)", elem_id="back")
            go = gr.Button("Сохранить и дальше (Enter)", variant="primary", elem_id="save")
            skip = gr.Button("К первой неразмеченной")
        outputs = [idx, head, body, ctx, asp, sent, sarc, msg]
        # a real aspect and «нет аспекта» exclude each other: the one just ticked wins
        asp.input(lambda new, old: [NONE] if NONE in new and NONE not in old else [x for x in new if x != NONE],
                  [asp, last_asp], asp)
        asp.change(lambda x: x, asp, last_asp)
        demo.load(show, idx, outputs)
        go.click(save, [idx, asp, sent, sarc], [idx, msg]).success(show, idx, outputs)
        back.click(lambda i: max(i - 1, 0), idx, idx).then(show, idx, outputs)
        skip.click(lambda: first_open(), None, idx).then(show, idx, outputs)
    demo.launch(server_port=a.port, head=HOTKEYS, css="#phrase { font-size: 20px; line-height: 1.5; }")


if __name__ == "__main__":
    main()
