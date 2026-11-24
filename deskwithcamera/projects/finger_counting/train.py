"""Train the finger model from the confirmed data set (data/dataset/dataset.npz).

    python -m projects.finger_counting.train                 try 3 kinds of model, test them honestly, save the best one
    python -m projects.finger_counting.train --model forest  use only this kind (forest | extra | mlp)
    python -m projects.finger_counting.train --no-save       only test, do not touch the saved model

How it is tested (the honest way): the model is trained on one filming session and tested on ANOTHER one it has never seen
(two sessions: A -> B and B -> A; with only one session: take 1 -> take 2). A model never gets tested on frames of a video it learned from.

The model answers per finger (5 times yes / no: thumb index middle ring pinky), so a pattern it has never seen
as a whole can still be right finger by finger. The number of fingers is the sum.
At the end the best kind is trained on ALL data and saved as data/model.joblib. The app loads it by itself
(Finger counting -> method "Model").
"""
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np

from projects.finger_counting import hands as H
from projects.finger_counting.model import DATA_DIR
from projects.finger_counting.recorder import name_of

FINAL = DATA_DIR / "dataset"
KINDS = ("forest", "extra", "mlp")


def load():
    f = FINAL / "dataset.npz"
    if not f.exists():
        sys.exit("No final data set yet. Run:  python -m projects.finger_counting.dataset build / confirm")
    with np.load(f) as z:
        d = {k: z[k] for k in z.files}
    if "P" not in d:
        sys.exit("The data set is from an older version (no hand points). Build (--force) and confirm the sessions again.")
    ex = FINAL / "exclude.txt"                                            # frames you removed in disagreements.html: "session/id"
    if ex.exists():
        gone = {ln.strip() for ln in ex.read_text(encoding="utf-8").splitlines() if ln.strip()}
        keep = np.array([f"{s}/{i}" not in gone for s, i in zip(d["session"], d["id"])])
        print(f"[train] {int((~keep).sum())} frames left out (exclude.txt)")
        d = {k: v[keep] for k, v in d.items()}
    d["X"] = np.array([H.features(p) for p in d["P"]], dtype=np.float32)      # features are computed here, so they can be improved any time
    return d


def to_y(bits):
    return np.array([[int(c) for c in str(b)] for b in bits], dtype=int)


def to_bits(y):
    return ["".join(str(int(v)) for v in row) for row in y]


def augment(X, Y):
    """Every recording also as its mirror image (left hand <-> right hand)."""
    Xm = np.array([H.mirrored(f) for f in X], dtype=np.float32)
    return np.vstack([X, Xm]), np.vstack([Y, Y])


def make(kind):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    if kind == "forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)
    if kind == "extra":
        return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0)
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=True,
                                                         random_state=0))


def folds(d):
    """[(name, train mask, test mask)]: always train and test on different videos."""
    sess = sorted(set(str(s) for s in d["session"]))
    if len(sess) >= 2:
        return [(f"test on {s[-8:]}", d["session"] != s, d["session"] == s) for s in sess]
    return [("take 1 -> take 2", d["take"] == 1, d["take"] == 2), ("take 2 -> take 1", d["take"] == 2, d["take"] == 1)]


def evaluate(kind, d, Y, fl):
    """-> y_true, y_pred, y_rule (pooled over all folds)"""
    t, p, r = [], [], []
    for name, tr, te in fl:
        if not tr.any() or not te.any():
            continue
        Xa, Ya = augment(d["X"][tr], Y[tr])
        clf = make(kind).fit(Xa, Ya)
        p.append(np.asarray(clf.predict(d["X"][te])).reshape(te.sum(), 5).astype(int))
        t.append(Y[te])
        r.append(to_y(d["rule"][te]))
    return np.vstack(t), np.vstack(p), np.vstack(r)


def scores(t, p):
    exact = float((p == t).all(axis=1).mean())
    count = float((p.sum(axis=1) == t.sum(axis=1)).mean())
    return exact, count, (p == t).mean(axis=0)


EVERYDAY = {"00000", "01000", "10000", "01100", "11000", "01110", "11100", "01111", "11111"}      # how people usually count 0 .. 5


def disagreements(d, t, p, fl):
    """A gallery of the frames the model got wrong on videos it did not learn from: look at them, remove the wrong labels."""
    ids = []
    for name, tr, te in fl:
        if tr.any() and te.any():
            ids += [(str(d["session"][i]), str(d["id"][i])) for i in np.where(te)[0]]
    tb, pb = to_bits(t), to_bits(p)
    paths = {}
    for sess in set(s for s, _ in ids):
        f = DATA_DIR / "review" / sess / "candidates.npz"
        if f.exists():
            with np.load(f) as z:
                paths[sess] = dict(zip([str(x) for x in z["id"]], [str(x) for x in z["path"]]))
    groups = {}
    for (sess, fid), a, b in zip(ids, tb, pb):
        if a != b and paths.get(sess, {}).get(fid):
            groups.setdefault(a, []).append((sess, fid, b))
    rng = np.random.default_rng(0)
    h = ["""<!doctype html><meta charset="utf-8"><title>disagreements</title><style>
:root{--w:340px}
body{font:14px system-ui,sans-serif;margin:0;background:#f4f1e6}
#bar{position:sticky;top:0;background:#fff;border-bottom:2px solid #1b2a52;padding:4px 12px;z-index:5;font-size:13px;display:flex;gap:14px;align-items:center}
#bar button{font-size:13px;padding:2px 8px}
#help{margin:8px 14px;color:#444;font-size:13px}
h2{margin:18px 14px 4px;font-size:15px}.g{display:flex;flex-wrap:wrap;gap:6px;margin:0 14px}
figure{margin:0;width:var(--w);border:4px solid #fff;background:#fff;cursor:pointer;position:relative}figure img{width:100%;display:block}
figcaption{font-size:12px;padding:2px 3px}figure.ex{border-color:#d00}figure.ex::after{content:"\\2715";position:absolute;inset:0;color:#d00;font-size:90px;text-align:center;line-height:130px;opacity:.7}
</style>
<div id="bar"><b>Wrong frames</b><span>excluded: <b id="n">0</b></span><button onclick="save()">Save exclude.txt</button>
<label>size <input type="range" min="150" max="900" value="340" oninput="document.documentElement.style.setProperty('--w',this.value+'px')"></label></div>
<div id="help">Frames the model got wrong on a video it did not learn from. Look at the real hand: if the LABEL (title) is wrong, click the frame (red).
Then Save exclude.txt and put the file into data/dataset/. The size slider makes the pictures bigger or smaller (no browser zoom needed).</div>"""]
    for a, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        pick = [items[i] for i in rng.permutation(len(items))[:40]]
        h.append(f'<h2>asked for {a} = {name_of(a)} &nbsp;<small>({len(items)} wrong frames, {len(pick)} shown)</small></h2><div class="g">')
        for sess, fid, b in pick:
            h.append(f'<figure data-id="{sess}/{fid}" onclick="this.classList.toggle(\'ex\');cnt()"><img loading="lazy" '
                     f'src="../review/{sess}/{paths[sess][fid]}"><figcaption>model said {b} ({name_of(b)})</figcaption></figure>')
        h.append("</div>")
    h.append("""<script>function cnt(){document.getElementById('n').textContent=document.querySelectorAll('figure.ex').length}
function save(){const ids=[...document.querySelectorAll('figure.ex')].map(f=>f.dataset.id).join('\\n');const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([ids+'\\n'],{type:'text/plain'}));a.download='exclude.txt';a.click()}</script>""")
    (FINAL / "disagreements.html").write_text("\n".join(h), encoding="utf-8")
    return FINAL / "disagreements.html"


def main(argv):
    try:
        import joblib
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit("scikit-learn missing:  pip install scikit-learn joblib")
    kinds = KINDS
    if "--model" in argv:
        kinds = (argv[argv.index("--model") + 1],)
    save = "--no-save" not in argv
    d = load()
    Y = to_y(d["bits"])
    fl = folds(d)
    n_sess = len(set(str(s) for s in d["session"]))
    out = [f"Data: {len(Y)} frames, {n_sess} session(s)    {datetime.now():%Y-%m-%d %H:%M}",
           "Test: " + ", ".join(n for n, _, _ in fl) + "   (the model never sees the video it is tested on)", ""]
    results = {}
    for kind in kinds:
        t0 = time.time()
        t, p, r = evaluate(kind, d, Y, fl)
        results[kind] = (t, p, r)
        ex, co, fa = scores(t, p)
        out.append(f"{kind:7s} exact pattern {ex:6.1%}   number of fingers {co:6.1%}   per finger "
                   + " ".join(f"{v:5.1%}" for v in fa) + f"    ({time.time() - t0:.0f} s)")
        print(out[-1])
    t, p, r = next(iter(results.values()))
    rex, rco, rfa = scores(t, r)
    out.append(f"{'rule':7s} exact pattern {rex:6.1%}   number of fingers {rco:6.1%}   per finger "
               + " ".join(f"{v:5.1%}" for v in rfa) + "    (the old hand-made rule, for comparison)")
    out.append("        (per finger order: thumb index middle ring pinky)")
    best = max(results, key=lambda k: scores(results[k][0], results[k][1])[0])
    out.append(f"\nBest: {best}")
    t, p, r = results[best]
    tb, pb = to_bits(t), to_bits(p)

    per = {}
    for a, b in zip(tb, pb):
        per.setdefault(a, [0, 0])
        per[a][0] += 1
        per[a][1] += a == b
    worst = sorted(per.items(), key=lambda kv: kv[1][1] / kv[1][0])[:10]
    out.append("\nShapes the model gets wrong most often (share correct):")
    for b, (n, ok) in worst:
        wrong = Counter(q for a, q in zip(tb, pb) if a == b and q != b).most_common(2)
        what = ",  ".join(f"{q} ({name_of(q)}) {c}x" for q, c in wrong)
        out.append(f"  {b}  {name_of(b):34s} {ok / n:6.1%}   mistaken for: {what or '-'}")
    byc = []
    for n in range(6):
        m = t.sum(axis=1) == n
        if m.any():
            byc.append(f"{n}: {float((p[m].sum(axis=1) == n).mean()):.0%}")
    out.append("\nNumber of fingers right, by real number:  " + "   ".join(byc))
    ev = np.array([a in EVERYDAY for a in tb])
    if ev.any():
        out.append(f"Everyday counting poses only ({', '.join(sorted(EVERYDAY))}):  pattern {float((p[ev] == t[ev]).all(axis=1).mean()):.1%},  "
                   f"number {float((p[ev].sum(axis=1) == t[ev].sum(axis=1)).mean()):.1%}   (rule: {float((r[ev].sum(axis=1) == t[ev].sum(axis=1)).mean()):.1%})")
    mix = Counter((a, q) for a, q in zip(tb, pb) if a != q).most_common(8)
    out.append("\nMost frequent mix-ups (real -> predicted):")
    for (a, q), c in mix:
        out.append(f"  {a} -> {q}   {c}x   ({name_of(a)}  ->  {name_of(q)})")
    text = "\n".join(out)
    print("\n" + text)
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "train_report.txt").write_text(text, encoding="utf-8")
    gal = disagreements(d, t, p, fl)
    print(f"\nWrong frames to look at: {gal}")

    if not save:
        print("\n(--no-save: the saved model was not changed)")
        return
    Xa, Ya = augment(d["X"], Y)
    clf = make(best).fit(Xa, Ya)
    ex, co, _ = scores(*results[best][:2])
    info = f"{best}, {len(Y)} frames, {n_sess} session(s); tested on unseen videos: pattern {ex:.0%}, number {co:.0%}"
    joblib.dump(dict(clf=clf, kind="bits", feature_n=H.FEATURE_N, info=info, model=best), DATA_DIR / "model.joblib")
    print(f"\nSaved: {DATA_DIR / 'model.joblib'}\n  {info}\nStart the app: Finger counting uses it automatically (method 'Model').")


if __name__ == "__main__":
    main(sys.argv[1:])
