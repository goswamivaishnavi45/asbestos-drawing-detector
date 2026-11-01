"""Generate synthetic building drawings (vector PDF + degraded scan PNG) with ground truth.

Each page has a floor plan with leader-line material annotations (some rotated 90 deg),
a finish schedule table (仕上表), specification notes and a title block whose
construction year may be written in Japanese era format (昭和53年) or omitted.

    python scripts/generate_synthetic.py --n 30 --out data/synthetic
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asbestos_detector.database import MaterialDB  # noqa: E402
from asbestos_detector.geometry import iou  # noqa: E402
from asbestos_detector.rules import classify  # noqa: E402

FONT = pymupdf.Font("japan")
PAGE_W, PAGE_H = 1191, 842  # A3 landscape, points

ROOMS = {
    "ja": ["事務室", "会議室", "廊下", "倉庫", "機械室", "便所", "休憩室", "受付", "更衣室", "書庫", "電気室", "湯沸室"],
    "en": ["Office", "Meeting Room", "Corridor", "Storage", "Machine Room", "Toilet", "Lounge", "Reception",
           "Locker Room", "Archive", "Electrical Room", "Pantry"],
}
LOC = {"ja": {"floor": "床", "base": "巾木", "wall": "壁", "ceiling": "天井", "roof": "屋根", "exterior": "外壁"},
       "en": {"floor": "Floor", "base": "Base", "wall": "Wall", "ceiling": "Ceiling", "roof": "Roof", "exterior": "Ext Wall"}}
# Which material types plausibly appear at each location.
BY_LOCATION = {
    "floor": ["A016", "A017", "A020", "N005", "N006", "N007", "N010", "N001"],
    "base": ["A016", "A017", "N008", "N010", "N002"],
    "wall": ["A015", "A007", "A008", "A018", "A012", "N008", "N009", "N002", "N001"],
    "ceiling": ["A014", "A015", "A007", "A002", "A001", "A003", "A008", "A005", "N008", "N011"],
    "roof": ["A009", "A010", "N004", "N012"],
    "exterior": ["A011", "A012", "A013", "A018", "A019", "N012", "N010", "N003"],
    "pipe": ["A004", "A021", "A006", "A022"],
}
SPECS = {"ja": ["t=12.5", "t=9.5", "t=6", "厚12mm", "2.0mm", "t=2", "t=25", "", "", ""],
         "en": ["t=12.5", "t=9.5", "t=6", "12mm", "2.0mm", "t=2", "", "", ""]}


def is_ascii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


class Sheet:
    def __init__(self, rng: random.Random, db: MaterialDB, lang: str):
        self.rng, self.db, self.lang = rng, db, lang
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.items: list[dict] = []
        self.occupied: list[tuple] = []
        self.year: int | None = None

    # --- text ---------------------------------------------------------------------
    def measure(self, x, y, text, fs, rotate=False):
        tw = pymupdf.TextWriter(self.page.rect)
        tw.append((x, y), text, font=FONT, fontsize=fs)
        r = tw.text_rect
        if rotate:  # text runs upward from (x, y): offset (dx, dy) -> (dy, -dx)
            bbox = (x + (r.y0 - y), y - (r.x1 - x), x + (r.y1 - y), y - (r.x0 - x))
        else:
            bbox = (r.x0, r.y0, r.x1, r.y1)
        return tw, bbox

    def fits(self, bbox, region=None, pad=4) -> bool:
        x0, y0, x1, y1 = bbox
        if region and not (region[0] <= x0 and region[1] <= y0 and x1 <= region[2] and y1 <= region[3]):
            return False
        return all(x1 + pad < o[0] or o[2] + pad < x0 or y1 + pad < o[1] or o[3] + pad < y0 for o in self.occupied)

    def put(self, x, y, text, fs, kind, rotate=False, check=False, region=None, **meta):
        tw, bbox = self.measure(x, y, text, fs, rotate)
        if check and not self.fits(bbox, region):
            return None
        if rotate:
            tw.write_text(self.page, morph=(pymupdf.Point(x, y), pymupdf.Matrix(90)))
        else:
            tw.write_text(self.page)
        self.occupied.append(bbox)
        item = {"text": text, "bbox_pt": [round(v, 2) for v in bbox], "kind": kind, "rotated": rotate,
                "lang": "en" if is_ascii(text) else "ja", **meta}
        self.items.append(item)
        return item

    # --- materials ------------------------------------------------------------------
    def material_text(self, location: str | None, max_width: float | None = None, fs: float = 7,
                      prefix: bool = True):
        for _ in range(20):
            entry = self.db.get(self.rng.choice(BY_LOCATION[location or "pipe"]))
            if self.lang == "en":
                forms = [entry.name_en] + [a for a in entry.aliases if is_ascii(a)]
            else:
                forms = [entry.name_ja] * 2 + [a for a in entry.aliases if not is_ascii(a)] \
                        + [a for a in entry.aliases if is_ascii(a) and len(a) <= 4]
            body = self.rng.choice(forms)
            if entry.status == "possible" and self.rng.random() < 0.1:
                body = ("石綿含有" + body) if self.lang == "ja" else f"Asbestos-containing {body}"
            elif entry.status != "asbestos" and self.rng.random() < 0.08:
                body = (body + "(ノンアスベスト)") if self.lang == "ja" else f"Non-asbestos {body}"
            spec = self.rng.choice(SPECS[self.lang])
            text = f"{body} {spec}".strip()
            if prefix and location:
                sep = ": " if self.lang == "en" else self.rng.choice([":", "："])
                text = f"{LOC[self.lang][location]}{sep}{text}"
            if max_width is None or FONT.text_length(text, fs) <= max_width:
                return entry, text
        return None, None

    def material_meta(self, entry, text):
        d = classify(entry, text, self.year)
        return {"entry_id": entry.id, "expected_label": d.label}

    # --- drawing parts --------------------------------------------------------------------
    def title_block(self):
        x0, y0, x1, row = 800, 690, 1152, 22
        rng, ja = self.rng, self.lang == "ja"
        year = rng.choice(["showa"] * 4 + ["western"] * 3 + ["heisei"] * 2 + ["none"] * 2)
        if year == "showa":
            n = rng.randint(35, 63)
            self.year, year_text = 1925 + n, f"竣工: 昭和{n}年"
        elif year == "heisei":
            n = rng.randint(1, 20)
            self.year, year_text = 1988 + n, f"竣工: 平成{n}年"
        elif year == "western":
            self.year = rng.randint(1962, 2015)
            year_text = f"竣工年: {self.year}年" if ja else f"Year built: {self.year}"
        else:
            self.year, year_text = None, ("竣工: 不明" if ja else "Year built: unknown")
        rows = [
            f"工事名: {rng.choice(['中央', '駅前', '第二', '北'])}ビル改修工事" if ja else f"Project: {rng.choice(['Central', 'Station', 'North'])} Building Renovation",
            "図面名: 1階平面図・仕上表" if ja else "Drawing: 1F Plan and Finish Schedule",
            year_text,
            f"縮尺 S=1:{rng.choice([100, 200])}   図番 A-{rng.randint(101, 120)}",
            f"日付 {rng.randint(2019, 2025)}.{rng.randint(1, 12):02d}.{rng.randint(1, 28):02d}",
        ]
        y1 = y0 + row * len(rows)
        self.page.draw_rect((x0, y0, x1, y1), width=1.2)
        for i, text in enumerate(rows):
            if i:
                self.page.draw_line((x0, y0 + i * row), (x1, y0 + i * row), width=0.6)
            self.put(x0 + 6, y0 + i * row + 15, text, 8.5, "title")

    def floor_plan(self):
        rng = self.rng
        x0, y0, x1, y1 = 60, 90, rng.randint(520, 590), rng.randint(480, 540)
        cols = rng.choice([2, 3])
        xs = sorted([x0, x1] + [x0 + (x1 - x0) * (i + rng.uniform(-0.12, 0.12)) / cols for i in range(1, cols)])
        ym = y0 + (y1 - y0) * rng.uniform(0.45, 0.6)
        self.page.draw_rect((x0, y0, x1, y1), width=3)
        self.page.draw_line((x0, ym), (x1, ym), width=2)
        for x in xs[1:-1]:
            self.page.draw_line((x, y0), (x, y1), width=2)
        rooms = rng.sample(ROOMS[self.lang], 2 * cols)
        k = 0
        self.room_boxes = []
        for ry0, ry1 in ((y0, ym), (ym, y1)):
            for cx0, cx1 in zip(xs, xs[1:]):
                name = rooms[k]
                k += 1
                tw, b = self.measure(0, 0, name, 9)
                self.put((cx0 + cx1) / 2 - (b[2] - b[0]) / 2, (ry0 + ry1) / 2, name, 9, "room")
                self.room_boxes.append((cx0, ry0, cx1, ry1))
                door_x = rng.uniform(cx0 + 15, cx1 - 40)  # door swing on the corridor wall
                self.page.draw_line((door_x, ym), (door_x + 25, ym), color=(1, 1, 1), width=3)
                self.page.draw_sector((door_x, ym), (door_x + 25, ym), 90 if ry0 == y0 else -90, width=0.5)
        # dimension chain above the plan
        dy = y0 - 22
        self.page.draw_line((x0, dy), (x1, dy), width=0.5)
        for a, b in zip(xs, xs[1:]):
            self.page.draw_line((a, dy - 4), (a, dy + 4), width=0.5)
            mm = str(int(round((b - a) / (x1 - x0) * rng.choice([14400, 16200, 18000]) / 300) * 300))
            self.put((a + b) / 2 - 10, dy - 4, mm, 7, "dim")
        self.page.draw_line((x1, dy - 4), (x1, dy + 4), width=0.5)
        self.plan_box = (x0, y0, x1, y1)

    def annotations(self, n: int):
        rng, (px0, py0, px1, py1) = self.rng, self.plan_box
        placed = 0
        for _ in range(n * 30):
            if placed >= n:
                break
            room = rng.choice(self.room_boxes)
            loc = rng.choice(["floor", "wall", "ceiling", "ceiling", None])
            entry, text = self.material_text(loc, fs=7.5)
            if not entry:
                continue
            rotate = rng.random() < 0.25
            x = rng.uniform(room[0] + 8, room[2] - 20)
            y = rng.uniform(room[1] + 20, room[3] - 8)
            region = (px0 + 4, py0 + 4, px1 - 4, py1 - 4)
            item = self.put(x, y, text, 7.5, "material", rotate=rotate, check=True, region=region,
                            **self.material_meta(entry, text))
            if item:
                b = item["bbox_pt"]
                tx, ty = rng.uniform(room[0] + 5, room[2] - 5), rng.uniform(room[1] + 5, room[3] - 5)
                sx, sy = (b[0], b[3] + 2) if not rotate else (b[2] + 2, b[3])
                self.page.draw_line((sx, sy), (tx, ty), width=0.4)
                self.page.draw_circle((tx, ty), 1.5, fill=(0, 0, 0))
                placed += 1

    def finish_schedule(self):
        rng, ja = self.rng, self.lang == "ja"
        x0, y0, row = 640, 100, 22
        widths = [64, 112, 112, 112, 112]
        headers = ["室名", "床", "巾木", "壁", "天井"] if ja else ["Room", "Floor", "Base", "Wall", "Ceiling"]
        locs = [None, "floor", "base", "wall", "ceiling"]
        n_rows = rng.randint(4, 6)
        xs = [x0 + sum(widths[:i]) for i in range(len(widths) + 1)]
        y1 = y0 + row * (n_rows + 1)
        self.put(x0, y0 - 10, "仕上表" if ja else "FINISH SCHEDULE", 10, "title")
        self.page.draw_rect((x0, y0, xs[-1], y1), width=1.2)
        for x in xs[1:-1]:
            self.page.draw_line((x, y0), (x, y1), width=0.6)
        for r in range(1, n_rows + 1):
            self.page.draw_line((x0, y0 + r * row), (xs[-1], y0 + r * row), width=0.6 if r > 1 else 1.0)
        for c, h in enumerate(headers):
            self.put(xs[c] + 5, y0 + 15, h, 8, "header")
        rooms = rng.sample(ROOMS[self.lang], n_rows)
        for r in range(n_rows):
            base_y = y0 + (r + 1) * row + 15
            self.put(xs[0] + 5, base_y, rooms[r], 7, "room")
            for c in range(1, 5):
                entry, text = self.material_text(locs[c], max_width=widths[c] - 12, fs=7, prefix=False)
                if entry:
                    self.put(xs[c] + 5, base_y, text, 7, "material", **self.material_meta(entry, text))
        self.schedule_bottom = y1

    def notes(self):
        rng, ja = self.rng, self.lang == "ja"
        x, y = 60, 600
        self.put(x, y, "特記事項" if ja else "GENERAL NOTES", 9, "title")
        for i, loc in enumerate(rng.sample(["roof", "exterior", "exterior", "pipe"], rng.randint(2, 4))):
            entry, text = self.material_text(loc if loc != "pipe" else None, fs=8)
            if loc == "pipe":
                text = ("配管: " if ja else "Piping: ") + text
            if entry:
                self.put(x + 8, y + 22 + i * 18, f"{i + 1}. {text}", 8, "material", **self.material_meta(entry, text))

    def tighten_boxes(self):
        """TextWriter rects include side bearings; use the extracted glyph extents as ground truth."""
        extracted = []
        for block in self.page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                extracted.append(("".join(s["text"] for s in line["spans"]).strip(), line["bbox"]))
        for it in self.items:
            same = [b for t, b in extracted if t == it["text"].strip()]
            if same:
                best = max(same, key=lambda b: iou(tuple(it["bbox_pt"]), tuple(b)))
                if iou(tuple(it["bbox_pt"]), tuple(best)) > 0.2:
                    it["bbox_pt"] = [round(v, 2) for v in best]

    def build(self, n_annotations: int):
        self.page.draw_rect((20, 20, PAGE_W - 20, PAGE_H - 20), width=1.5)
        self.title_block()  # first, so the construction year is known when labelling materials
        self.floor_plan()
        self.put(60, 570, "1階平面図  S=1:100" if self.lang == "ja" else "1F PLAN  S=1:100", 10, "title")
        self.finish_schedule()
        self.annotations(n_annotations)
        self.notes()
        self.tighten_boxes()


def degrade(gray: np.ndarray, rng: random.Random) -> tuple[np.ndarray, float]:
    """Simulate a photocopied / scanned sheet."""
    h, w = gray.shape
    angle = rng.uniform(-2.0, 2.0)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(gray, m, (w, h), borderValue=255).astype(np.float32)
    img = img * rng.uniform(0.75, 0.95) + rng.uniform(15, 45)  # faded toner, grey paper
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    img += nrng.normal(0, rng.uniform(3, 8), img.shape)
    speck = nrng.random(img.shape) < 0.0006
    img[speck] = 40
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(50, 85)])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE), angle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db = MaterialDB.load()
    rng = random.Random(args.seed)
    for i in range(args.n):
        lang = rng.choice(["ja", "ja", "en"])
        sheet = Sheet(rng, db, lang)
        sheet.build(n_annotations=rng.randint(3, 6))
        sid = f"drawing_{i:03d}"
        sheet.doc.save(out / f"{sid}.pdf")
        dpi = rng.choice([180, 200, 220])
        pix = sheet.page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        clean = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        scan, angle = degrade(clean, rng)
        cv2.imencode(".png", scan)[1].tofile(str(out / f"{sid}_scan.png"))
        gt = {"id": sid, "lang": lang, "build_year": sheet.year, "scan_dpi": dpi, "scan_skew_deg": round(angle, 3),
              "items": sheet.items}
        (out / f"{sid}.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
        n_mat = sum(1 for it in sheet.items if it["kind"] == "material")
        print(f"{sid}: lang={lang} year={sheet.year} items={len(sheet.items)} materials={n_mat} dpi={dpi}")


if __name__ == "__main__":
    main()
