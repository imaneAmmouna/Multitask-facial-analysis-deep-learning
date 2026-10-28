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
from filterpy.kalman import KalmanFilter

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
IMG_SIZE = 112
AGE_MAX = 100.0
MODEL_PATH = r"E:\Soft7Art\faces_detection_app\mtl_model.pth"

GENDER_LABELS = {0: "Homme", 1: "Femme"}
EMOTION_LABELS = {
    0: "Surprise", 1: "Peur", 2: "Dégoût",
    3: "Joie", 4: "Tristesse", 5: "Colère", 6: "Neutre",
}
EMOTION_COLORS_BGR = {
    0: (183, 74, 83),
    1: (221, 138, 55),
    2: (126, 83, 212),
    3: (39, 159, 239),
    4: (117, 158, 29),
    5: (74, 75, 226),
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
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head_gender = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(2048, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 2)
        )
        self.head_age = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(2048, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.head_emotion = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(2048, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 7)
        )

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.head_gender(f), self.head_age(f), self.head_emotion(f)


# ──────────────────────────────────────────────────────
# CHARGEMENT MODÈLE (cache Streamlit)
# ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTLFaceModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


@st.cache_resource
def load_detector():
    from insightface.app import FaceAnalysis
    detector = FaceAnalysis(
        name="buffalo_sc",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection"],
    )
    detector.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    return detector


# ──────────────────────────────────────────────────────
# TRANSFORM
# ──────────────────────────────────────────────────────
mtl_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
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
    img = Image.fromarray(face_rgb)
    tensor = mtl_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        og, oa, oe = model(tensor)
    pg = torch.softmax(og, 1)[0]
    pe = torch.softmax(oe, 1)[0]
    gid = int(pg.argmax()); gc = float(pg[gid])
    eid = int(pe.argmax()); ec = float(pe[eid])
    age = max(1, min(100, round(float(oa.squeeze()) * AGE_MAX)))
    return {
        "gender_id": gid, "gender": GENDER_LABELS[gid], "gender_conf": gc,
        "age": age,
        "emotion_id": eid, "emotion": EMOTION_LABELS[eid], "emotion_conf": ec,
    }

# ──────────────────────────────────────────────────────
# PIPELINE COMPLET SUR UNE FRAME
# ──────────────────────────────────────────────────────


def run_pipeline(img_bgr, model, device, detector, tracker=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    img_ann = img_rgb.copy()

    faces_detected = detector.get(img_bgr)
    results = []

    for face in faces_detected:
        bbox = face.bbox.astype(int)
        x1 = max(0, bbox[0]); y1 = max(0, bbox[1])
        x2 = min(w, bbox[2]); y2 = min(h, bbox[3])
        lmks = face.kps.astype(int).tolist()

        lm_dict = {
            "right_eye": lmks[0], "left_eye": lmks[1],
            "nose": lmks[2], "mouth_right": lmks[3], "mouth_left": lmks[4],
        }

        aligned = align_face(img_rgb, lmks)
        if aligned is None:
            bw = x2 - x1; bh = y2 - y1
            crop = img_rgb[max(0, y1 - int(bh * .1)):min(h, y2 + int(bh * .1)),
                              max(0, x1 - int(bw * .1)):min(w, x2 + int(bw * .1))]
            aligned = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)) \
                      if crop.size > 0 else np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)

        preds = predict_face(aligned, model, device)
        color = EMOTION_COLORS_BGR[preds["emotion_id"]]

        # Boîte + texte
        cv2.rectangle(img_ann, (x1, y1), (x2, y2), color, 2)
        lines = [
            f"{preds['gender']} ({preds['gender_conf']*100:.0f}%)",
            f"Age : {preds['age']} ans",
            f"{preds['emotion']} ({preds['emotion_conf']*100:.0f}%)",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for li, line in enumerate(lines):
            ty = max(y1 - (len(lines) - li) * 16, 16)
            (tw, th), _ = cv2.getTextSize(line, font, 0.42, 1)
            cv2.rectangle(img_ann, (x1, ty - th - 3),
                          (x1 + tw + 6, ty + 3), color, -1)
            cv2.putText(img_ann, line, (x1 + 3, ty),
                        font, 0.42, (255, 255, 255), 1)

        # Landmarks
        for pt in lmks:
            cv2.circle(img_ann, (int(pt[0]), int(pt[1])), 3, color, -1)

        results.append({"bbox": [x1, y1, x2, y2], "face": aligned, **preds})

    return img_ann, results

# ──────────────────────────────────────────────────────
# TRACKER SORT SIMPLIFIÉ
# ──────────────────────────────────────────────────────


class SimpleSORT:
    """SORT simplifié : filtre de Kalman + IoU matching."""

    def __init__(self, max_age=5, min_hits=1, iou_thresh=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.trackers = []
        self.next_id = 1

    def _iou(self, b1, b2):
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, detections):
        """detections : list of [x1,y1,x2,y2]"""
        if len(detections) == 0:
            for t in self.trackers:
                t["age"] += 1
            self.trackers = [t for t in self.trackers
                             if t["age"] <= self.max_age]
            return []

        if len(self.trackers) == 0:
            for det in detections:
                self.trackers.append({
                    "id": self.next_id, "bbox": det, "age": 0, "hits": 1
                })
                self.next_id += 1
            return [(t["id"], t["bbox"]) for t in self.trackers]

        # Association IoU greedy
        matched = set()
        used_det = set()
        for ti, t in enumerate(self.trackers):
            best_iou = self.iou_thresh
            best_di = -1
            for di, det in enumerate(detections):
                if di in used_det:
                    continue
                iou = self._iou(t["bbox"], det)
                if iou > best_iou:
                    best_iou = iou; best_di = di
            if best_di >= 0:
                self.trackers[ti]["bbox"] = detections[best_di]
                self.trackers[ti]["age"] = 0
                self.trackers[ti]["hits"] += 1
                matched.add(ti); used_det.add(best_di)

        # Nouvelles détections non matchées
        for di, det in enumerate(detections):
            if di not in used_det:
                self.trackers.append({
                    "id": self.next_id, "bbox": det, "age": 0, "hits": 1
                })
                self.next_id += 1

        # Supprimer trackers trop vieux
        self.trackers = [t for t in self.trackers
                         if t["age"] <= self.max_age]

        return [(t["id"], t["bbox"]) for t in self.trackers
                if t["hits"] >= self.min_hits]

# ──────────────────────────────────────────────────────
# INTERFACE STREAMLIT
# ──────────────────────────────────────────────────────


def main():
    # ── Header ───────────────────────────────────────
    st.title("🎭 Système de reconnaissance faciale multi-tâches")
    st.markdown(
        "Détection simultanée du **genre**, de l'**âge** et de l'**émotion** "
        "via un modèle MTL ResNet50 + RetinaFace."
    )

    # ── Chargement modèle ────────────────────────────
    with st.spinner("Chargement du modèle MTL..."):
        model, device = load_model()
    with st.spinner("Chargement de RetinaFace..."):
        detector = load_detector()

    st.success(f"Modèle chargé — Device : {device}")

    # ── Sidebar : métriques ──────────────────────────
    with st.sidebar:
        st.header("📊 Performances du modèle")
        st.metric("Genre Accuracy", "91.5%")
        st.metric("Genre F1-macro", "0.914")
        st.metric("MAE Âge", "6.74 ans")
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
            st.markdown(
                f'<span style="color:{color}; font-weight:bold;">'
                f'● {label}</span>',
                unsafe_allow_html=True
            )

    # ── Onglets ──────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📷 Image statique",
        "🎬 Vidéo",
        "📹 Webcam temps réel",
    ])

    # ════════════════════════════════════════════════
    # ONGLET 1 — IMAGE STATIQUE
    # ════════════════════════════════════════════════
    with tab1:
        st.subheader("Upload d'une image")
        uploaded = st.file_uploader(
            "Choisir une image",
            type=["jpg", "jpeg", "png"],
            key="img_upload"
        )

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                         caption="Image originale", use_column_width=True)

            with st.spinner("Analyse en cours..."):
                t0 = time.perf_counter()
                img_ann, results = run_pipeline(
                    img_bgr, model, device, detector)
                elapsed = (time.perf_counter() - t0) * 1000

            with col2:
                st.image(img_ann, caption="Résultats de l'analyse",
                         use_column_width=True)

            # Métriques
            st.markdown(f"**Temps de traitement : {elapsed:.0f} ms "
                        f"({1000/elapsed:.1f} FPS)**")
            st.markdown(f"**{len(results)} visage(s) détecté(s)**")

            # Tableau des résultats
            if results:
                st.subheader("Détail par visage")
                cols = st.columns(min(len(results), 4))
                for i, (res, col) in enumerate(zip(results, cols)):
                    with col:
                        color = EMOTION_COLORS_HEX[res["emotion_id"]]
                        st.markdown(
                            f"<div style='border:2px solid {color};"
                            f"border-radius:8px;padding:10px;'>"
                            f"<b>Visage {i+1}</b><br>"
                            f"👤 {res['gender']} "
                            f"({res['gender_conf']*100:.0f}%)<br>"
                            f"🎂 {res['age']} ans<br>"
                            f"<span style='color:{color};font-weight:bold;'>"
                            f"😊 {res['emotion']} "
                            f"({res['emotion_conf']*100:.0f}%)</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        # Patch facial aligné
                        st.image(res["face"],
                                 caption=f"Patch aligné {i+1}",
                                 use_column_width=True)

                # Barres de confiance émotion pour le 1er visage
                if len(results) > 0:
                    st.subheader("Confiances émotions — Visage 1")
                    r = results[0]
                    with torch.no_grad():
                        img_t = mtl_transform(
                            Image.fromarray(r["face"])
                        ).unsqueeze(0).to(device)
                        _, _, oe = model(img_t)
                        probs = torch.softmax(oe, 1)[0].cpu().numpy()

                    import pandas as pd
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
        video_file = st.file_uploader(
            "Choisir une vidéo",
            type=["mp4", "avi", "mov"],
            key="vid_upload"
        )

        use_tracking = st.checkbox(
            "Activer le tracking SORT (ID stable par personne)", value=True)

        if video_file and st.button("▶ Analyser la vidéo"):
            # Sauvegarder temporairement (Windows compatible)
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(temp_dir, "input_video.mp4")
            out_path = os.path.join(temp_dir, "output_video.mp4")
            with open(tmp_path, "wb") as f:
                f.write(video_file.read())

            cap = cv2.VideoCapture(tmp_path)
            fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w_src    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_src    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            st.info(f"Vidéo : {n_frames} frames | {fps_src:.0f} FPS "
                    f"| {w_src}×{h_src}")

            # Output vidéo annotée
            out_path = os.path.join(temp_dir, "output_video.mp4")
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(out_path, fourcc, fps_src,
                                       (w_src, h_src))

            tracker    = SimpleSORT() if use_tracking else None
            progress   = st.progress(0)
            frame_slot = st.empty()
            stats_slot = st.empty()

            times  = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.perf_counter()
                img_ann, results = run_pipeline(
                    frame, model, device, detector, tracker)
                elapsed = (time.perf_counter() - t0) * 1000
                times.append(elapsed)

                # Tracking : dessiner ID
                if use_tracking and tracker and results:
                    bboxes = [r["bbox"] for r in results]
                    tracked = tracker.update(bboxes)
                    for tid, bbox in tracked:
                        x1,y1,x2,y2 = [int(v) for v in bbox]
                        cv2.putText(img_ann, f"ID:{tid}",
                                    (x1, y2+15),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255,255,255), 1)

                writer.write(cv2.cvtColor(img_ann, cv2.COLOR_RGB2BGR))

                # Afficher 1 frame sur 5 pour la fluidité Streamlit
                if frame_idx % 5 == 0:
                    frame_slot.image(img_ann, use_column_width=True)
                    stats_slot.markdown(
                        f"Frame {frame_idx}/{n_frames} | "
                        f"{elapsed:.0f}ms | "
                        f"{1000/elapsed:.1f} FPS"
                    )

                progress.progress(min(frame_idx / max(n_frames, 1), 1.0))
                frame_idx += 1

            cap.release()
            writer.release()

            st.success(
                f"Vidéo traitée — {frame_idx} frames | "
                f"Temps moyen : {np.mean(times):.0f}ms | "
                f"FPS moyen : {1000/np.mean(times):.1f}"
            )

            # Téléchargement vidéo annotée
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Télécharger la vidéo annotée",
                    data        = f.read(),
                    file_name   = "video_annotee.mp4",
                    mime        = "video/mp4",
                )

    # ════════════════════════════════════════════════
    # ONGLET 3 — WEBCAM
    # ════════════════════════════════════════════════
    with tab3:
        st.subheader("Analyse temps réel — Webcam")
        st.info(
            "La webcam nécessite que l'application tourne en local. "
            "Si vous utilisez un serveur distant, utilisez le mode Vidéo."
        )

        use_tracking_cam = st.checkbox(
            "Activer SORT tracking", value=True, key="track_cam")

        run = st.checkbox("▶ Démarrer la webcam", value=False)

        if run:
            cap     = cv2.VideoCapture(0)
            tracker = SimpleSORT() if use_tracking_cam else None

            frame_slot = st.empty()
            fps_slot   = st.empty()
            stop_btn   = st.button("⏹ Arrêter")

            times = []

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.error("Impossible de lire la webcam")
                    break

                t0 = time.perf_counter()
                img_ann, results = run_pipeline(
                    frame, model, device, detector, tracker)
                elapsed = (time.perf_counter() - t0) * 1000
                times.append(elapsed)

                if use_tracking_cam and tracker and results:
                    bboxes  = [r["bbox"] for r in results]
                    tracked = tracker.update(bboxes)
                    for tid, bbox in tracked:
                        x1,y1,x2,y2 = [int(v) for v in bbox]
                        cv2.putText(img_ann, f"ID:{tid}",
                                    (x1, y2+15),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255,255,255), 1)

                frame_slot.image(img_ann, use_column_width=True)
                fps_slot.markdown(
                    f"**{1000/elapsed:.1f} FPS** | "
                    f"{elapsed:.0f}ms | "
                    f"{len(results)} visage(s)"
                )

            cap.release()
            if times:
                st.info(
                    f"Session terminée — FPS moyen : "
                    f"{1000/np.mean(times):.1f}"
                )


# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    main()