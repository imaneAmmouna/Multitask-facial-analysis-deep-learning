# ====================================================================
# app_face_detection.py — Interface de démonstration MTL Face Analysis
# Streamlit — Image / Vidéo / Webcam + SORT Tracking
# ====================================================================
import os
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import streamlit as st
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────
# CONFIGURATION PAGE
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="MTL Face Analysis",
    page_icon="🎭",
    layout="wide",
)

# ──────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────
IMG_SIZE            = 112
AGE_MAX             = 100.0
MODEL_PATH          = r"E:\Soft7Art\faces_detection_app\mtl_model.pth"
LOCK_AFTER_N_FRAMES = 5
PREVIEW_EVERY_N     = 10   # frames capturées pour la grille d'aperçu

GENDER_LABELS = {0: "Homme", 1: "Femme"}
GENDER_ABBR   = {0: "H",     1: "F"}

EMOTION_LABELS = {
    0: "Surprise", 1: "Peur",      2: "Dégoût",
    3: "Joie",     4: "Tristesse", 5: "Colère", 6: "Neutre",
}
EMOTION_COLORS_BGR = {
    0: (183,  74,  83), 1: (221, 138,  55), 2: (126,  83, 212),
    3: ( 39, 159, 239), 4: (117, 158,  29), 5: ( 74,  75, 226),
    6: (128, 135, 136),
}
EMOTION_COLORS_HEX = {
    0: "#534AB7", 1: "#378ADD", 2: "#D4537E",
    3: "#EF9F27", 4: "#1D9E75", 5: "#E24B4A", 6: "#888780",
}

# ──────────────────────────────────────────────────────
# ARCHITECTURE MTL
# ──────────────────────────────────────────────────────
class MTLFaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone    = nn.Sequential(*list(resnet.children())[:-1])
        self.head_gender = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(2048, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 2))
        self.head_age = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(2048, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid())
        self.head_emotion = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(2048, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 7))

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.head_gender(f), self.head_age(f), self.head_emotion(f)


# ──────────────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MTLFaceModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

@st.cache_resource
def load_detector():
    from insightface.app import FaceAnalysis
    detector = FaceAnalysis(
        name="buffalo_sc",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection"])
    detector.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    return detector


# ──────────────────────────────────────────────────────
# TRANSFORM
# ──────────────────────────────────────────────────────
mtl_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────
# ALIGNEMENT FACIAL
# ──────────────────────────────────────────────────────
def align_face(img_rgb, lmks, size=112):
    dst = np.array([
        [38.29, 51.70], [73.53, 51.50], [56.02, 71.74],
        [41.55, 92.37], [70.73, 92.20],
    ], dtype=np.float32)
    src = np.array(lmks, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img_rgb, M, (size, size),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


# ──────────────────────────────────────────────────────
# PRÉDICTION MTL
# ──────────────────────────────────────────────────────
def predict_face(face_rgb, model, device):
    img    = Image.fromarray(face_rgb)
    tensor = mtl_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        og, oa, oe = model(tensor)
    pg  = torch.softmax(og, 1)[0]
    pe  = torch.softmax(oe, 1)[0]
    gid = int(pg.argmax());  gc = float(pg[gid])
    eid = int(pe.argmax());  ec = float(pe[eid])
    age = max(1, min(100, round(float(oa.squeeze()) * AGE_MAX)))
    return {
        "gender_id": gid, "gender": GENDER_LABELS[gid], "gender_conf": gc,
        "age": age,
        "emotion_id": eid, "emotion": EMOTION_LABELS[eid], "emotion_conf": ec,
    }


# ──────────────────────────────────────────────────────
# MÉMOIRE STABLE PAR PERSONNE
# Âge + Genre : verrouillés après N frames
# Émotion     : toujours live
# ──────────────────────────────────────────────────────
class PersonMemory:
    def __init__(self, lock_after=LOCK_AFTER_N_FRAMES):
        self.lock_after = lock_after
        self._data      = {}

    def reset(self):
        self._data = {}

    def update(self, tid, raw_preds):
        if tid not in self._data:
            self._data[tid] = {
                "ages": [], "gender_ids": [], "gender_confs": [],
                "locked": False,
                "age": None, "gender_id": None,
                "gender": None, "gender_conf": None,
            }
        mem = self._data[tid]

        if not mem["locked"]:
            mem["ages"].append(raw_preds["age"])
            mem["gender_ids"].append(raw_preds["gender_id"])
            mem["gender_confs"].append(raw_preds["gender_conf"])

            if len(mem["ages"]) >= self.lock_after:
                best_gid  = Counter(mem["gender_ids"]).most_common(1)[0][0]
                best_age  = round(sum(mem["ages"]) / len(mem["ages"]))
                best_conf = float(np.mean([
                    c for g, c in zip(mem["gender_ids"], mem["gender_confs"])
                    if g == best_gid]))
                mem.update({"age": best_age, "gender_id": best_gid,
                            "gender": GENDER_LABELS[best_gid],
                            "gender_conf": best_conf, "locked": True})
            else:
                mem.update({"age": raw_preds["age"],
                            "gender_id": raw_preds["gender_id"],
                            "gender": raw_preds["gender"],
                            "gender_conf": raw_preds["gender_conf"]})

        return {
            "gender_id":   mem["gender_id"],
            "gender":      mem["gender"],
            "gender_conf": mem["gender_conf"],
            "age":         mem["age"],
            "emotion_id":  raw_preds["emotion_id"],
            "emotion":     raw_preds["emotion"],
            "emotion_conf":raw_preds["emotion_conf"],
            "locked":      mem["locked"],
        }


# ──────────────────────────────────────────────────────
# TRACKER SORT SIMPLIFIÉ
# ──────────────────────────────────────────────────────
class SimpleSORT:
    def __init__(self, max_age=5, min_hits=1, iou_thresh=0.3):
        self.max_age    = max_age
        self.min_hits   = min_hits
        self.iou_thresh = iou_thresh
        self.trackers   = []
        self.next_id    = 1

    def _iou(self, b1, b2):
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, detections):
        if len(detections) == 0:
            for t in self.trackers: t["age"] += 1
            self.trackers = [t for t in self.trackers if t["age"] <= self.max_age]
            return []
        if len(self.trackers) == 0:
            for det in detections:
                self.trackers.append({"id": self.next_id, "bbox": det, "age": 0, "hits": 1})
                self.next_id += 1
            return [(t["id"], t["bbox"]) for t in self.trackers]

        used_det = set()
        for ti, t in enumerate(self.trackers):
            best_iou, best_di = self.iou_thresh, -1
            for di, det in enumerate(detections):
                if di in used_det: continue
                iou = self._iou(t["bbox"], det)
                if iou > best_iou: best_iou = iou; best_di = di
            if best_di >= 0:
                self.trackers[ti].update({"bbox": detections[best_di], "age": 0})
                self.trackers[ti]["hits"] += 1
                used_det.add(best_di)
            else:
                self.trackers[ti]["age"] += 1

        for di, det in enumerate(detections):
            if di not in used_det:
                self.trackers.append({"id": self.next_id, "bbox": det, "age": 0, "hits": 1})
                self.next_id += 1

        self.trackers = [t for t in self.trackers if t["age"] <= self.max_age]
        return [(t["id"], t["bbox"]) for t in self.trackers if t["hits"] >= self.min_hits]


# ──────────────────────────────────────────────────────
# DESSIN DES ANNOTATIONS
# Format : "ID1 | H | 48 ans" + "Neutre (76%)"
# ──────────────────────────────────────────────────────
def draw_annotations(img_ann, results):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for res in results:
        x1, y1b, x2, y2b = res["bbox"]
        color       = EMOTION_COLORS_BGR[res["emotion_id"]]
        gender_abbr = GENDER_ABBR.get(res.get("gender_id", 0), "?")
        tid_label   = f"ID{res['track_id']}" if res.get("track_id", -1) >= 0 else "?"

        lines = [
            f"{tid_label} | {gender_abbr} | {res['age']} ans",
            f"{res['emotion']} ({res['emotion_conf']*100:.0f}%)",
        ]

        cv2.rectangle(img_ann, (x1, y1b), (x2, y2b), color, 2)

        for li, line in enumerate(lines):
            ty = max(y1b - (len(lines) - li) * 22, 22)
            (tw, th), _ = cv2.getTextSize(line, font, 0.52, 1)
            cv2.rectangle(img_ann, (x1, ty - th - 4), (x1 + tw + 8, ty + 4), color, -1)
            cv2.putText(img_ann, line, (x1 + 4, ty), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return img_ann


# ──────────────────────────────────────────────────────
# PIPELINE SUR UNE FRAME
# ──────────────────────────────────────────────────────
def run_pipeline(img_bgr, model, device, detector,
                 tracker=None, memory=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    img_ann = img_rgb.copy()

    faces_detected = detector.get(img_bgr)
    results = []

    for face in faces_detected:
        bbox = face.bbox.astype(int)
        x1 = max(0, bbox[0]); y1 = max(0, bbox[1])
        x2 = min(w, bbox[2]); y2 = min(h, bbox[3])
        lmks = face.kps.astype(int).tolist()

        aligned = align_face(img_rgb, lmks)
        if aligned is None:
            bw = x2 - x1; bh = y2 - y1
            crop = img_rgb[max(0, y1-int(bh*.1)):min(h, y2+int(bh*.1)),
                           max(0, x1-int(bw*.1)):min(w, x2+int(bw*.1))]
            aligned = (cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                       if crop.size > 0
                       else np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8))

        preds = predict_face(aligned, model, device)
        results.append({"bbox": [x1, y1, x2, y2], "face": aligned,
                        "track_id": -1, **preds})

    # Tracking
    if tracker is not None and results:
        bboxes  = [r["bbox"] for r in results]
        tracked = tracker.update(bboxes)
        for tid, tbbox in tracked:
            best_i = min(range(len(results)),
                         key=lambda i: abs(results[i]["bbox"][0]-tbbox[0]) +
                                       abs(results[i]["bbox"][1]-tbbox[1]))
            results[best_i]["track_id"] = tid

    # Verrouillage âge/genre
    for res in results:
        tid = res["track_id"]
        if memory is not None and tid >= 0:
            stable = memory.update(tid, res)
            res.update({k: stable[k] for k in
                        ["gender_id","gender","gender_conf","age",
                         "emotion_id","emotion","emotion_conf","locked"]})
        else:
            res["locked"] = False

    img_ann = draw_annotations(img_ann, results)
    return img_ann, results


# ──────────────────────────────────────────────────────
# TRAITEMENT VIDÉO AVEC predict_every_n_frames
# ──────────────────────────────────────────────────────
def process_video_streamlit(tmp_path, out_path, model, device, detector,
                             use_tracking, predict_every_n, progress,
                             stats_slot, preview_slots):
    cap      = cv2.VideoCapture(tmp_path)
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_src    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_src, (w_src, h_src))

    tracker       = SimpleSORT()   if use_tracking else None
    memory        = PersonMemory() if use_tracking else None

    times          = []
    frame_idx      = 0
    all_results    = []
    last_results   = []          # résultats propagés entre les frames
    last_img_ann   = None
    preview_buffer = []          # (frame_idx, img_ann) pour la grille finale
    t_start        = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # ── Prédire seulement tous les N frames ──────────
        if frame_idx % predict_every_n == 0:
            img_ann, results = run_pipeline(frame, model, device,
                                            detector, tracker, memory)
            last_results = results
            last_img_ann = img_ann
        else:
            # Propager les derniers résultats sur la frame courante
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_ann = img_rgb.copy()
            img_ann = draw_annotations(img_ann, last_results)
            results = last_results

        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        all_results.append(results)

        writer.write(cv2.cvtColor(img_ann, cv2.COLOR_RGB2BGR))

        # ── Stats temps réel ─────────────────────────────
        recent    = times[-20:] if len(times) >= 20 else times
        fps_cur   = 1000 / np.mean(recent) if recent else 0
        remaining = (n_frames - frame_idx) / fps_cur if fps_cur > 0 else 0

        stats_slot.markdown(
            f"**Frame {frame_idx}/{n_frames}** &nbsp;|&nbsp; "
            f"**{fps_cur:.1f} FPS** &nbsp;|&nbsp; "
            f"ETA : {remaining:.0f}s &nbsp;|&nbsp; "
            f"Visages : {len(results)}"
        )

        # ── Capture pour grille d'aperçu ─────────────────
        if frame_idx % PREVIEW_EVERY_N == 0:
            preview_buffer.append((frame_idx, img_ann.copy()))
            # Afficher les 3 dernières frames en temps réel
            recent_previews = preview_buffer[-3:]
            for slot_i, slot in enumerate(preview_slots):
                if slot_i < len(recent_previews):
                    fidx, fimg = recent_previews[slot_i]
                    slot.image(fimg, caption=f"Frame {fidx}",
                               use_container_width=True)
                else:
                    slot.empty()

        progress.progress(min(frame_idx / max(n_frames, 1), 1.0))
        frame_idx += 1

    cap.release()
    writer.release()

    total_time = time.perf_counter() - t_start
    avg_fps    = frame_idx / total_time if total_time > 0 else 0

    return all_results, preview_buffer, avg_fps, frame_idx, total_time


# ──────────────────────────────────────────────────────
# GRILLE D'APERÇU FINALE (style show_sample_frames)
# ──────────────────────────────────────────────────────
def show_frames_grid(preview_buffer, title="Aperçu des frames annotées"):
    if not preview_buffer:
        return

    n      = len(preview_buffer)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(18, 6 * n_rows))
    axes = np.array(axes).flatten()

    for i, (fidx, fimg) in enumerate(preview_buffer):
        axes[i].imshow(fimg)
        axes[i].set_title(f"Frame {fidx}", fontsize=10)
        axes[i].axis("off")

    # Masquer les cases vides
    for j in range(len(preview_buffer), len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────
# RÉSUMÉ STATISTIQUE
# ──────────────────────────────────────────────────────
def show_summary(all_results_list, use_tracking=True):
    all_faces = [f for frame_faces in all_results_list for f in frame_faces]
    if not all_faces:
        st.info("Aucun visage détecté durant la session.")
        return

    st.subheader("📊 Résumé des résultats")

    if use_tracking:
        per_id = defaultdict(list)
        for f in all_faces:
            per_id[f.get("track_id", -1)].append(f)
        deduped = []
        for tid, faces in per_id.items():
            locked = [f for f in faces if f.get("locked", False)]
            src    = locked if locked else faces
            best_g = Counter([f["gender"]  for f in src]).most_common(1)[0][0]
            deduped.append({
                "track_id": tid,
                "gender":   best_g,
                "gender_conf": float(np.mean([f["gender_conf"] for f in src
                                              if f["gender"] == best_g])),
                "emotion":  Counter([f["emotion"] for f in faces]).most_common(1)[0][0],
                "age":      round(sum(f["age"] for f in src) / len(src)),
                "n_frames": len(faces),
            })
        n_unique  = len([d for d in deduped if d["track_id"] >= 0])
        stats_src = deduped
        label     = "Personnes uniques détectées"
    else:
        deduped = [{"track_id": i, **{k: f[k] for k in
                    ["gender","gender_conf","emotion","age"]}, "n_frames": 1}
                   for i, f in enumerate(all_faces)]
        n_unique  = len(all_faces)
        stats_src = deduped
        label     = "Détections totales"

    c1, c2, c3 = st.columns(3)
    c1.metric(label, n_unique)
    top_e = Counter([d["emotion"] for d in stats_src]).most_common(1)
    c2.metric("Émotion dominante", top_e[0][0] if top_e else "—")
    avg_a = sum(d["age"] for d in stats_src) / len(stats_src) if stats_src else 0
    c3.metric("Âge moyen", f"{avg_a:.1f} ans")

    c4, c5 = st.columns(2)
    with c4:
        st.markdown("**Distribution des émotions**")
        st.bar_chart(pd.Series([d["emotion"] for d in stats_src]).value_counts())
    with c5:
        st.markdown("**Distribution du genre**")
        st.bar_chart(pd.Series([d["gender"] for d in stats_src]).value_counts())

    rows = [{"ID": d["track_id"] if d["track_id"] >= 0 else "—",
             "Genre": d["gender"],
             "Conf.": f"{d['gender_conf']*100:.0f}%",
             "Âge":   f"{d['age']} ans",
             "Émotion": d["emotion"],
             "Frames": d["n_frames"]}
            for d in sorted(stats_src, key=lambda x: x["track_id"])]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ──────────────────────────────────────────────────────
# INTERFACE STREAMLIT
# ──────────────────────────────────────────────────────
def main():
    st.title("🎭 Système de reconnaissance faciale multi-tâches")
    st.markdown(
        "Détection simultanée du **genre**, de l'**âge** et de l'**émotion** "
        "via un modèle MTL ResNet50 + RetinaFace.")

    with st.spinner("Chargement du modèle MTL..."):
        model, device = load_model()
    with st.spinner("Chargement de RetinaFace..."):
        detector = load_detector()

    st.success(f"Modèle chargé — Device : {device}")

    # ── Sidebar ──────────────────────────────────────
    with st.sidebar:
        st.header("📊 Performances du modèle")
        st.metric("Genre Accuracy",   "91.5%")
        st.metric("Genre F1-macro",   "0.914")
        st.metric("MAE Âge",          "6.74 ans")
        st.metric("Émotion Accuracy", "65.2%")
        st.metric("Émotion F1-macro", "0.628")
        st.divider()
        st.header("🏗️ Architecture")
        st.markdown("""
        - **Backbone** : ResNet50 (26.2M params)
        - **Détecteur** : RetinaFace ONNX
        - **Tâches** : Genre + Âge + Émotion
        - **Dataset** : UTKFace + IMDB-Wiki + RAF-DB + AffectNet
        """)
        st.divider()
        st.header("🎨 Couleurs émotions")
        for eid, label in EMOTION_LABELS.items():
            color = EMOTION_COLORS_HEX[eid]
            st.markdown(f'<span style="color:{color};font-weight:bold;">● {label}</span>',
                        unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📷 Image statique", "🎬 Vidéo", "📹 Webcam temps réel"])

    # ════════════════════════════════════════════════
    # ONGLET 1 — IMAGE STATIQUE
    # ════════════════════════════════════════════════
    with tab1:
        st.subheader("Upload d'une image")
        uploaded = st.file_uploader("Choisir une image",
                                    type=["jpg","jpeg","png"], key="img_upload")
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                         caption="Image originale", use_container_width=True)
            with st.spinner("Analyse en cours..."):
                t0 = time.perf_counter()
                img_ann, results = run_pipeline(img_bgr, model, device, detector)
                elapsed = (time.perf_counter() - t0) * 1000
            with col2:
                st.image(img_ann, caption="Résultats", use_container_width=True)

            st.markdown(f"**{elapsed:.0f} ms · {1000/elapsed:.1f} FPS · "
                        f"{len(results)} visage(s)**")

            if results:
                st.subheader("Détail par visage")
                cols = st.columns(min(len(results), 4))
                for i, (res, col) in enumerate(zip(results, cols)):
                    with col:
                        color = EMOTION_COLORS_HEX[res["emotion_id"]]
                        g_abbr = GENDER_ABBR.get(res.get("gender_id", 0), "?")
                        st.markdown(
                            f"<div style='border:2px solid {color};"
                            f"border-radius:8px;padding:10px;'>"
                            f"<b>Visage {i+1}</b><br>"
                            f"👤 {res['gender']} ({res['gender_conf']*100:.0f}%)<br>"
                            f"🎂 {res['age']} ans<br>"
                            f"<span style='color:{color};font-weight:bold;'>"
                            f"😊 {res['emotion']} ({res['emotion_conf']*100:.0f}%)"
                            f"</span></div>", unsafe_allow_html=True)
                        st.image(res["face"], caption=f"Patch {i+1}",
                                 use_container_width=True)

                st.subheader("Confiances émotions — Visage 1")
                with torch.no_grad():
                    img_t = mtl_transform(
                        Image.fromarray(results[0]["face"])
                    ).unsqueeze(0).to(device)
                    _, _, oe = model(img_t)
                    probs = torch.softmax(oe, 1)[0].cpu().numpy()
                df_probs = pd.DataFrame({
                    "Émotion": list(EMOTION_LABELS.values()),
                    "Confiance": [float(p) for p in probs],
                }).sort_values("Confiance", ascending=False)
                st.bar_chart(df_probs.set_index("Émotion"))

    # ════════════════════════════════════════════════
    # ONGLET 2 — VIDÉO
    # ════════════════════════════════════════════════
    with tab2:
        st.subheader("Analyse d'une vidéo")
        video_file = st.file_uploader("Choisir une vidéo",
                                      type=["mp4","avi","mov"], key="vid_upload")

        col_a, col_b = st.columns(2)
        with col_a:
            use_tracking = st.checkbox("Activer le tracking SORT", value=True)
        with col_b:
            predict_every_n = st.slider(
                "Prédire tous les N frames", min_value=1, max_value=10,
                value=5, help="1 = chaque frame · 5 = toutes les 5 frames (plus rapide)")

        if video_file and st.button("▶ Analyser la vidéo"):
            import tempfile
            temp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(temp_dir, "input_video.mp4")
            out_path = os.path.join(temp_dir, "output_video.mp4")

            with open(tmp_path, "wb") as f:
                f.write(video_file.read())

            cap      = cv2.VideoCapture(tmp_path)
            fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w_src    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_src    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            st.info(f"Vidéo : {w_src}×{h_src} · {fps_src:.0f} FPS · {n_frames} frames")

            progress    = st.progress(0)
            stats_slot  = st.empty()

            st.markdown("### 🎞️ Aperçu en temps réel")
            preview_cols  = st.columns(3)
            preview_slots = [col.empty() for col in preview_cols]

            all_results, preview_buffer, avg_fps, frame_idx, total_time = \
                process_video_streamlit(
                    tmp_path, out_path, model, device, detector,
                    use_tracking, predict_every_n,
                    progress, stats_slot, preview_slots)

            progress.progress(1.0)
            stats_slot.markdown(
                f"✅ **Terminé** — {frame_idx} frames · "
                f"{total_time:.1f}s · **{avg_fps:.1f} FPS moyen**")

            # ── Grille finale complète ────────────────
            st.markdown("### 🖼️ Vidéo annotée — aperçu frames")
            show_frames_grid(preview_buffer)

            # ── Résumé stats ─────────────────────────
            show_summary(all_results, use_tracking=use_tracking)

            # ── Téléchargement ────────────────────────
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Télécharger la vidéo annotée",
                    data=f.read(), file_name="video_annotee.mp4", mime="video/mp4")

    # ════════════════════════════════════════════════
    # ONGLET 3 — WEBCAM
    # ════════════════════════════════════════════════
    with tab3:
        st.subheader("Analyse temps réel — Webcam")
        st.info("Requiert une exécution locale.")

        use_tracking_cam = st.checkbox("Activer SORT tracking", value=True, key="track_cam")
        run = st.checkbox("▶ Démarrer la webcam", value=False)

        if run:
            cap     = cv2.VideoCapture(0)
            tracker = SimpleSORT()   if use_tracking_cam else None
            memory  = PersonMemory() if use_tracking_cam else None

            frame_slot = st.empty()
            fps_slot   = st.empty()
            stop_btn   = st.button("⏹ Arrêter")

            times       = []
            cam_results = []

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.error("Impossible de lire la webcam")
                    break

                t0 = time.perf_counter()
                img_ann, results = run_pipeline(frame, model, device,
                                                detector, tracker, memory)
                elapsed = (time.perf_counter() - t0) * 1000
                times.append(elapsed)
                cam_results.append(results)

                frame_slot.image(img_ann, use_container_width=True)
                fps_slot.markdown(
                    f"**{1000/elapsed:.1f} FPS** · {elapsed:.0f}ms · {len(results)} visage(s)")

            cap.release()
            if times:
                st.info(f"Session terminée — FPS moyen : {1000/np.mean(times):.1f}")
            show_summary(cam_results, use_tracking=use_tracking_cam)


if __name__ == "__main__":
    main()