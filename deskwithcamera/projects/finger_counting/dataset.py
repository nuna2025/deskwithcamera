"""Data set for the trained finger model:  video session  ->  REVIEW (you look at every frame)  ->  CONFIRM  ->  final data set.

Run from the project folder (where main.py is):

    python -m projects.finger_counting.dataset status             what exists, what is confirmed, how many frames per shape
    python -m projects.finger_counting.dataset build [session]    1. cut frames out of a session video, find the hand points
                                                                     (frames without a hand get a slow second search in tiles; --no-retry skips it),
                                                                     write data/review/<session>/gallery.html
    python -m projects.finger_counting.dataset confirm [session]  2. after you looked at the gallery: add the good frames
                                                                     to data/dataset/dataset.npz (the final data)

Without [session] the newest session is used. Nothing reaches the final data until you run "confirm".
Frames you do not want: click them in gallery.html (red) and save exclude.txt into the review folder -
or simply delete their .jpg files. Confirm leaves those out.
"""
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config  # noqa: F401  (core.opts reads it)
from core.opts import opt
from projects.finger_counting import hands as H
from projects.finger_counting.model import DATA_DIR
from projects.finger_counting.recorder import name_of

VIDEOS = DATA_DIR / "videos"
REVIEW = DATA_DIR / "review"
FINAL = DATA_DIR / "dataset"
SKIP_START = 0.5              # seconds after the start of a recording that are not used (the hand is still settling)
STRIDE = 2                    # every 2nd video frame (neighbouring frames are almost identical)
NOHAND_EVERY = 3              # frames without a usable hand: only every 3rd is saved as a picture (to look at)
JPG_WIDTH = 960
MARGIN = 0.03
WARN_FEW = 30                 # a shape with fewer usable frames than this is reported

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
FINGER_COLORS = [(0, 128, 255), (0, 200, 0), (255, 160, 0), (220, 0, 220), (0, 0, 230)]     # thumb .. pinky (BGR)


# ---------------------------------------------------------------------------------------------------------------------
def _all_folders():
    return sorted(p for p in VIDEOS.iterdir() if (p / "session.mp4").exists()) if VIDEOS.exists() else []


def _sessions():
    """Finished sessions only. A session that was stopped by force (no session.json) has a video that may be cut off."""
    return [p for p in _all_folders() if (p / "session.json").exists() and (p / "labels.csv").exists()]


def _state(name):
    rv = REVIEW / name
    if (FINAL / "sessions.json").exists() and name in json.loads((FINAL / "sessions.json").read_text(encoding="utf-8")):
        return "confirmed"
    return "built (waiting for your confirm)" if (rv / "candidates.npz").exists() else "new"


def _pick_session(arg, want_new=False):
    sessions = _sessions()
    if not sessions:
        sys.exit(f"No session found in {VIDEOS}. Film one first (key V).")
    if arg:
        p = VIDEOS / arg
        if p.exists() and p not in sessions:
            sys.exit(f"{arg} was not finished properly (no session.json: the program was closed during filming). "
                     "Its video may be cut off, so it is not used.")
        if not p.exists():
            sys.exit(f"No such session: {arg}   (existing: {', '.join(s.name for s in sessions)})")
        return p
    if want_new:
        new = [s for s in sessions if _state(s.name) == "new"]
        return new[-1] if new else sessions[-1]
    built = [s for s in sessions if _state(s.name).startswith("built")]
    return built[-1] if built else sessions[-1]


def _zone(hd, Hm, top):
    """Where the hand is on the table: "ok", "band" (a point in the button band) or "outside" (wrist off the table).
    Only for your information: the hand-local features do not depend on the position, so such frames are kept."""
    if Hm is None:
        return "ok"
    n = H.to_norm(Hm, hd.pts)
    if not (-MARGIN <= n[0, 0] <= 1 + MARGIN and -MARGIN <= n[0, 1] <= 1 + MARGIN):
        return "outside"
    return "band" if n[:, 1].min() < top else "ok"


def _pick_hand(hands, Hm, top):
    """The hand that is filmed: the biggest one. -> (hand, zone) or (None, "no hand found")"""
    if not hands:
        return None, "no hand found"
    hd = max(hands, key=lambda h: H.palm_size(h.pts))
    return hd, _zone(hd, Hm, top)


def _overlay(frame, hand, text):
    import cv2
    img = frame.copy()
    if hand is not None:
        p = hand.pts
        for a, b in BONES:
            cv2.line(img, (int(p[a, 0]), int(p[a, 1])), (int(p[b, 0]), int(p[b, 1])), (255, 255, 255), 2)
        for i in range(21):
            col = (255, 255, 255) if i == 0 else FINGER_COLORS[(i - 1) // 4]
            cv2.circle(img, (int(p[i, 0]), int(p[i, 1])), 6, col, -1)
    h, w = img.shape[:2]
    img = cv2.resize(img, (JPG_WIDTH, int(h * JPG_WIDTH / w)), interpolation=cv2.INTER_AREA)
    cv2.putText(img, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(img, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


# ---------------------------------------------------------------------------------------------------------------------
def build(arg=None, force=False, tracker=None, retry=True):
    import cv2
    folder = _pick_session(arg, want_new=True)
    name = folder.name
    out = REVIEW / name
    if _state(name) == "confirmed" and not force:
        sys.exit(f"{name} is already confirmed. (Build again with --force: your old exclusions in the review folder are lost.)")
    if _state(name).startswith("built") and (out / "exclude.txt").exists() and not force:
        sys.exit(f"{name} was built before and you already have an exclude.txt there. Use --force to build again.")
    info = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    cal = info.get("calibration") or {}
    Hm = np.array(cal["H"], float) if cal.get("H") else None
    top = opt("HUD_FRAC")
    rows = list(csv.DictReader(open(folder / "labels.csv", encoding="utf-8")))

    tr = tracker
    if tr is None:
        tr = H.HandTracker()
        tr._load()                                   # blocking: downloads the hand model the first time
        if not tr.ready:
            sys.exit(f"Hand model not available: {tr.message}")

    wanted = {}                                      # video frame number -> index of the row in labels.csv
    skipped_marked = []
    for i, r in enumerate(rows):
        if r.get("marked") == "redo":
            skipped_marked.append(f"take {r['take']} step {r['step']} {r['label']}")
            continue
        a = int(round((float(r["rec_start_s"]) + SKIP_START) * fps))
        b = int(float(r["rec_end_s"]) * fps)
        for f in range(a, b, STRIDE):
            wanted[f] = i

    if out.exists():
        import shutil
        shutil.rmtree(out)
    (out / "frames").mkdir(parents=True)
    cap = cv2.VideoCapture(str(folder / "session.mp4"))
    if not cap.isOpened():
        sys.exit("Could not open session.mp4")
    last = max(wanted) if wanted else -1
    print(f"[dataset] {name}: {len(rows)} recordings, {len(wanted)} frames to look at ...")

    recs = []                                        # one dict per frame
    f = -1
    while f < last:
        ok = cap.grab()
        f += 1
        if not ok:
            print(f"[dataset] the video ends at frame {f} (stopped early?)")
            break
        if f not in wanted:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        r = rows[wanted[f]]
        bits, take, step = r["label"], int(r["take"]), int(r["step"])
        fid = f"t{take}_s{step:02d}_f{f:06d}"
        found = tr.detect(frame, ts_ms=f * 1000.0 / fps)
        hand, why = _pick_hand(found, Hm, top)                                      # why = zone, or "no hand found"
        how = "normal"
        if hand is None and retry and hasattr(tr, "detect_zoom"):                    # nothing found: search in tiles
            hand, why2 = _pick_hand(tr.detect_zoom(frame), Hm, top)
            if hand is not None:
                how, why = "zoom", why2
        rec = dict(id=fid, bits=bits, take=take, step=step, frame=f, ok=hand is not None, why=why, how=how, rule="", palm=0.0,
                   pts=np.zeros((21, 3), np.float32), file="")
        sub = f"t{take}_s{step:02d}_{bits}"
        if hand is not None:
            rec["rule"] = "".join("1" if v else "0" for v in H.finger_states(hand.pts))
            rec["pts"] = hand.pts.astype(np.float32)
            rec["palm"] = H.palm_size(hand.pts)
            img = _overlay(frame, hand, f"{bits}  (rule says {rec['rule']})" + ("  [zoom]" if how == "zoom" else "") + ("" if why == "ok" else f"  [{why}]"))
            rec["file"] = f"frames/{sub}/{fid}.jpg"
        elif len([x for x in recs if x["step"] == step and x["take"] == take and not x["ok"]]) % NOHAND_EVERY == 0:
            img = _overlay(frame, None, f"{bits}  - no hand found")
            rec["file"] = f"frames/{sub}/{fid}_nohand.jpg"
        else:
            img = None
        if img is not None:
            (out / "frames" / sub).mkdir(exist_ok=True)
            cv2.imwrite(str(out / rec["file"]), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        recs.append(rec)
        if len(recs) % 200 == 0:
            print(f"[dataset]   {len(recs)} / {len(wanted)}   (hand found so far: {100 * sum(x['ok'] for x in recs) // len(recs)} %)")
    cap.release()

    np.savez_compressed(out / "candidates.npz",
                        id=np.array([r["id"] for r in recs]), bits=np.array([r["bits"] for r in recs]),
                        take=np.array([r["take"] for r in recs]), step=np.array([r["step"] for r in recs]),
                        frame=np.array([r["frame"] for r in recs]), ok=np.array([r["ok"] for r in recs]), how=np.array([r["how"] for r in recs]),
                        zone=np.array([r["why"] for r in recs]),
                        rule=np.array([r["rule"] for r in recs]), palm=np.array([r["palm"] for r in recs], np.float32),
                        path=np.array([r["file"] for r in recs]), P=np.array([r["pts"] for r in recs], np.float32).reshape(-1, 21, 3))
    _gallery(out, name, rows, recs)
    (out / "build.json").write_text(json.dumps(dict(session=name, built=datetime.now().isoformat(timespec="seconds"), fps=fps,
                                                    skip_start=SKIP_START, stride=STRIDE, frames=len(recs),
                                                    marked_redo_left_out=skipped_marked), indent=2), encoding="utf-8")
    ok_n = sum(r["ok"] for r in recs)
    print(f"\n[dataset] {len(recs)} frames, hand found in {ok_n} ({100 * ok_n / max(len(recs), 1):.0f} %).")
    zoom_n = sum(r["how"] == "zoom" for r in recs)
    got = [r for r in recs if r["ok"]]
    print(f"[dataset]   found directly: {ok_n - zoom_n}   found by the tile search: {zoom_n}   MediaPipe found nothing: {len(recs) - ok_n}")
    print(f"[dataset]   where the hand is:  on the table {sum(r['why'] == 'ok' for r in got)},  "
          f"a point in the button band {sum(r['why'] == 'band' for r in got)},  wrist off the table {sum(r['why'] == 'outside' for r in got)}   (all kept)")
    per = {}
    for r in recs:
        per.setdefault(r["bits"], [0, 0])
        per[r["bits"]][0] += 1
        per[r["bits"]][1] += r["ok"]
    rate = sorted(((v[1] / v[0], b) for b, v in per.items()))
    print("[dataset]   hardest shapes (share with a hand found): " + ",  ".join(f"{b} {100 * x:.0f} %" for x, b in rate[:6]))
    if skipped_marked:
        print(f"[dataset] left out (you marked them with X): {len(skipped_marked)}  e.g. {skipped_marked[0]}")
    bad = {}
    for r in recs:
        if not r["ok"]:
            k = (r["take"], r["step"], r["bits"])
            bad[k] = bad.get(k, 0) + 1
    worst = sorted(bad.items(), key=lambda kv: -kv[1])[:5]
    if worst:
        print("[dataset] most frames without a hand:  " + ",  ".join(f"take {t} #{s} {b} ({n})" for (t, s, b), n in worst))
    print(f"\nNow look at it:   {out / 'gallery.html'}\nWhen everything is fine:   python -m projects.finger_counting.dataset confirm {name}")


def _gallery(out, name, rows, recs):
    groups = {}
    for r in recs:
        groups.setdefault((r["take"], r["step"], r["bits"]), []).append(r)
    html = [f"""<!doctype html><meta charset="utf-8"><title>{name}</title>
<style>
:root{{--w:340px}}
body{{font:14px system-ui,sans-serif;margin:0;background:#f4f1e6;color:#222}}
#bar{{position:sticky;top:0;background:#fff;border-bottom:2px solid #1b2a52;padding:4px 12px;font-size:13px;z-index:5;display:flex;gap:18px;align-items:center;flex-wrap:wrap}}
button{{font:inherit;padding:5px 12px;cursor:pointer}}
h2{{margin:22px 14px 4px;font-size:17px}} h2 small{{font-weight:normal;color:#555}}
.grid{{display:flex;flex-wrap:wrap;gap:6px;margin:0 14px}}
figure{{margin:0;width:var(--w);border:4px solid #fff;background:#fff;cursor:pointer;position:relative}}
figure img{{width:100%;display:block}} figcaption{{font-size:11px;padding:2px 3px;color:#444}}
figure.rd{{border-color:#f0a020}} figure.zn{{border-color:#2a7de1}} figure.nohand{{border-color:#999;opacity:.55;cursor:default}}
figure.ex{{border-color:#d00}} figure.ex::after{{content:"\\2715";position:absolute;top:0;left:0;right:0;bottom:0;color:#d00;font-size:90px;text-align:center;line-height:120px;opacity:.7}}
.few{{color:#d00;font-weight:bold}} .legend span{{display:inline-block;padding:1px 8px;border:3px solid;margin-right:6px}}
body.only figure:not(.rd):not(.nohand):not(.zn){{display:none}}
</style>
<div id="bar"><b>{name}</b>
<span class="legend"><span style="border-color:#f0a020">rule says something else</span><span style="border-color:#2a7de1">hand partly in the button band / off the table (kept)</span><span style="border-color:#999">no hand found</span><span style="border-color:#d00">excluded by you</span></span>
<label><input type="checkbox" onchange="document.body.classList.toggle('only',this.checked)"> only suspicious frames</label>
<span>excluded: <b id="n">0</b></span> <button onclick="save()">Save exclude.txt</button>
<label>size <input type="range" min="150" max="900" value="340" oninput="document.documentElement.style.setProperty('--w',this.value+'px')"></label></div>
<p style="margin:8px 14px;font-size:13px">Click a frame = exclude / include. Save exclude.txt, then move it into this folder (or just delete the .jpg files).
Colored points: thumb orange, index green, middle blue-ish, ring pink, pinky red. A frame is good when the points sit on the real joints and the label matches your hand.</p>"""]
    for (take, step, bits), items in sorted(groups.items()):
        usable = sum(r["ok"] for r in items)
        rd = sum(1 for r in items if r["ok"] and r["rule"] != bits)
        note = next((x["note"] for x in rows if int(x["take"]) == take and int(x["step"]) == step), "")
        few = f' <span class="few">only {usable} usable</span>' if usable < WARN_FEW else ""
        html.append(f'<h2><span>Take {take} #{step}: {bits} = {name_of(bits)}{" (" + note + ")" if note else ""}</span> '
                    f'<small>{usable} / {len(items)} with a hand, {rd} where the rule disagrees</small>{few} '
                    f'<button onclick="allOf(this.parentNode.nextElementSibling)">exclude / include all</button></h2><div class="grid">')
        for r in items:
            if not r["file"]:
                continue
            cls = "nohand" if not r["ok"] else ("rd" if r["rule"] != r["bits"] else ("zn" if r["why"] != "ok" else ""))
            cap = f"f{r['frame']}" + (" - no hand found" if not r["ok"] else f" - rule {r['rule']}" + ("" if r["why"] == "ok" else f" - {r['why']}"))
            click = "" if not r["ok"] else ' onclick="tog(this)"'
            html.append(f'<figure class="{cls}" data-id="{r["id"]}"{click}><img loading="lazy" src="{r["file"]}"><figcaption>{cap}</figcaption></figure>')
        html.append("</div>")
    html.append("""<script>
function cnt(){document.getElementById('n').textContent=document.querySelectorAll('figure.ex').length}
function tog(f){f.classList.toggle('ex');cnt()}
function allOf(g){const fs=[...g.querySelectorAll('figure:not(.nohand)')];const on=fs.some(f=>!f.classList.contains('ex'));fs.forEach(f=>f.classList.toggle('ex',on));cnt()}
function save(){const ids=[...document.querySelectorAll('figure.ex')].map(f=>f.dataset.id).join('\\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([ids+'\\n'],{type:'text/plain'}));a.download='exclude.txt';a.click()}
</script>""")
    (out / "gallery.html").write_text("\n".join(html), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------------
def confirm(arg=None):
    folder = _pick_session(arg)
    name = folder.name
    rv = REVIEW / name
    if not (rv / "candidates.npz").exists():
        sys.exit(f"{name} has not been built yet:  python -m projects.finger_counting.dataset build {name}")
    with np.load(rv / "candidates.npz") as z:
        d = {k: z[k] for k in z.files}
    ids = [str(x) for x in d["id"]]
    drop = set()
    if (rv / "exclude.txt").exists():
        drop |= {ln.strip() for ln in (rv / "exclude.txt").read_text(encoding="utf-8").splitlines() if ln.strip()}
    n_listed = len(drop)
    gone = 0
    for i, fid in enumerate(ids):
        fn = str(d["path"][i])
        if bool(d["ok"][i]) and fn and not (rv / fn).exists():           # you deleted the picture
            drop.add(fid)
            gone += 1
    keep = np.array([bool(d["ok"][i]) and ids[i] not in drop for i in range(len(ids))])

    if "P" not in d:
        sys.exit(f"{name} was built with an older version (no hand points saved). Build it again:  "
                 f"python -m projects.finger_counting.dataset build {name} --force")
    new = dict(P=d["P"][keep], bits=d["bits"][keep], take=d["take"][keep], step=d["step"][keep], id=d["id"][keep],
               palm=d["palm"][keep], rule=d["rule"][keep], session=np.array([name] * int(keep.sum())))
    FINAL.mkdir(parents=True, exist_ok=True)
    f = FINAL / "dataset.npz"
    if f.exists():
        with np.load(f) as z:                                           # read everything and CLOSE the file (Windows cannot replace an open file)
            old = {k: z[k] for k in z.files}
        if "P" not in old:                                              # data set of an older version: start again
            f.replace(FINAL / "dataset_old.npz")
            (FINAL / "sessions.json").unlink(missing_ok=True)
            print("[dataset] the old final data set had no hand points; it was moved to dataset_old.npz. Confirm all sessions again.")
            old = {k: v[:0] for k, v in new.items()}
            meta_reset = True
        mask = old["session"] != name                                   # confirming again replaces this session
        for k in new:
            new[k] = np.concatenate([old[k][mask], new[k]])
    tmp = FINAL / "dataset.tmp.npz"
    np.savez_compressed(tmp, **new)
    for attempt in range(10):                                           # a virus scanner / sync program may hold the file for a moment
        try:
            tmp.replace(f)
            break
        except PermissionError:
            if attempt == 9:
                sys.exit(f"Could not replace {f} (is it open in another program?). The new data is in {tmp}.")
            time.sleep(0.5)
    sj = FINAL / "sessions.json"
    meta = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    meta[name] = dict(confirmed=datetime.now().isoformat(timespec="seconds"), frames_kept=int(keep.sum()),
                      excluded_by_you=n_listed + gone, no_hand=int((~d["ok"]).sum()))
    sj.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[dataset] {name}: kept {int(keep.sum())} frames. Left out: {int((~d['ok']).sum())} without a usable hand, "
          f"{n_listed} in exclude.txt, {gone} deleted pictures.")
    status()


def zonecheck(arg=None):
    """Draws where the program thinks the table (green) and the end of the button band (red line) are, on a few
    frames of the session, together with the hand points. -> data/review/<session>/zone_check.jpg"""
    import cv2
    folder = _pick_session(arg)
    rv = REVIEW / folder.name
    if not (rv / "candidates.npz").exists():
        sys.exit("Build this session first.")
    info = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    cal = info.get("calibration") or {}
    if not cal.get("H"):
        sys.exit("This session has no calibration matrix.")
    Hm = np.array(cal["H"], float)
    Hi = np.linalg.inv(Hm)
    top = opt("HUD_FRAC")
    with np.load(rv / "candidates.npz") as z:
        d = {k: z[k] for k in z.files}
    cap = cv2.VideoCapture(str(folder / "session.mp4"))
    print(f"calibration: camera size {cal.get('cam_size')}, video size {int(cap.get(3))} x {int(cap.get(4))}, band bottom = {top}")
    idx = [i for i in range(len(d["id"])) if d["ok"][i]]
    pick = [idx[int(k)] for k in np.linspace(0, len(idx) - 1, 6)]
    tiles = []
    for i in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(d["frame"][i]))
        ok, frame = cap.read()
        if not ok:
            continue
        def to_cam(pt):
            v = Hi @ np.array([pt[0], pt[1], 1.0])
            return int(v[0] / v[2]), int(v[1] / v[2])
        box = np.array([to_cam(c) for c in ((0, 0), (1, 0), (1, 1), (0, 1))])
        cv2.polylines(frame, [box], True, (0, 255, 0), 3)
        cv2.line(frame, to_cam((0, top)), to_cam((1, top)), (0, 0, 255), 3)
        label = f"{d['bits'][i]}  zone={d['zone'][i]}"
        cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 6)
        cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        tiles.append(cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1]))))
    cap.release()
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    mosaic = np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)])
    cv2.imwrite(str(rv / "zone_check.jpg"), mosaic)
    print(f"saved: {rv / 'zone_check.jpg'}   (green = table as the program sees it, red line = end of the button band)")


def status():
    sessions = _sessions()
    print("\nSessions (data/videos):")
    for s in sessions:
        info = json.loads((s / "session.json").read_text(encoding="utf-8")) if (s / "session.json").exists() else {}
        print(f"  {s.name:28s} {info.get('seconds', 0) / 60:5.1f} min   {_state(s.name)}")
    if not sessions:
        print("  (none)")
    for p in _all_folders():
        if p not in sessions:
            print(f"  {p.name:28s} not finished (no session.json) - ignored")
    f = FINAL / "dataset.npz"
    if not f.exists():
        print("\nFinal data set: still empty.")
        return
    with np.load(f) as z:
        d = {k: z[k] for k in z.files}
    if "P" not in d:
        print("\nFinal data set: old version (no hand points). Build and confirm the sessions again.")
        return
    print(f"\nFinal data set (data/dataset/dataset.npz): {len(d['P'])} frames from {len(set(d['session']))} session(s)")
    pats = {}
    for b, t in zip(d["bits"], d["take"]):
        pats.setdefault(str(b), {}).setdefault(int(t), 0)
        pats[str(b)][int(t)] += 1
    print("  shape   fingers                                take1  take2")
    for n in range(6):
        for b in sorted(p for p in pats if p.count("1") == n):
            c = pats[b]
            low = "   <-- few" if min(c.get(1, 0), c.get(2, 0)) < WARN_FEW else ""
            print(f"  {b}   {name_of(b):38s} {c.get(1, 0):5d}  {c.get(2, 0):5d}{low}")
    missing = [f"{''.join(c)}" for c in __import__('itertools').product("01", repeat=5) if "".join(c) not in pats]
    if missing:
        print("  not in the data yet:", ", ".join(missing))


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    force = "--force" in argv
    retry = "--no-retry" not in argv
    cmd = args[0] if args else "status"
    arg = args[1] if len(args) > 1 else None
    if cmd == "build":
        build(arg, force, retry=retry)
    elif cmd == "confirm":
        confirm(arg)
    elif cmd == "status":
        status()
    elif cmd == "zonecheck":
        zonecheck(arg)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
