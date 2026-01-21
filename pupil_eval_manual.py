#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import mediapipe as mp
import numpy as np
import argparse
import math
import os

# ==== IDs MediaPipe pour les yeux / iris (comme dans ton code) ====
L_IDS = [33, 160, 158, 133, 153, 144]
R_IDS = [263, 387, 385, 362, 380, 373]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def _eye_roi_from_ids(lms, ids, w: int, h: int, scale: float = 1.6,
                      scale_x: float = None, scale_y: float = None) -> tuple[int, int, int, int]:
    if scale_x is None:
        scale_x = scale
    if scale_y is None:
        scale_y = scale * 2.0

    pts = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in ids], dtype=np.int32)
    x, y, w0, h0 = cv.boundingRect(pts)
    cx, cy = x + w0 // 2, y + h0 // 2

    sx = int(scale_x * w0 // 2)
    sy = int(scale_y * h0 // 2)

    x1, y1, x2, y2 = cx - sx, cy - sy, cx + sx, cy + sy
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return x1, y1, x2, y2


def get_iris_center(lms, iris_ids, w_img, h_img):
    pts = []
    for i in iris_ids:
        if i >= len(lms):
            continue
        pts.append((lms[i].x * w_img, lms[i].y * h_img))
    if not pts:
        return None
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    return int(c[0]), int(c[1])


# Variables globales pour le callback souris
_click = {"x": None, "y": None, "flag": False}


def _on_mouse(event, x, y, flags, userdata):
    global _click
    if event == cv.EVENT_LBUTTONDOWN:
        _click["x"] = x
        _click["y"] = y
        _click["flag"] = True


def main(video_path: str):
    if not os.path.isfile(video_path):
        raise SystemExit(f"Vidéo introuvable: {video_path}")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la vidéo: {video_path}")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )

    win_name = "Pupil Eval (clic = GT, s = skip, q = quit)"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.setMouseCallback(win_name, _on_mouse)

    frame_idx = 0
    errors = []

    print("\nInstructions :")
    print("  - Pour chaque frame, clique sur le CENTRE de la pupille GAUCHE dans la ROI verte.")
    print("  - Touche 's' : passer cette frame sans annotation")
    print("  - Touche 'q' : quitter")
    print("  - Une fois terminé, la moyenne \\bar e sera affichée.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        h_img, w_img = frame.shape[:2]

        # Détection FaceMesh
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not res.multi_face_landmarks:
            # Aucun visage → afficher et permettre de skip
            disp = frame.copy()
            cv.putText(disp, "Pas de visage detecte (s=skip, q=quit)",
                       (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imshow(win_name, disp)
            k = cv.waitKey(0) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                continue
            else:
                continue

        lms = res.multi_face_landmarks[0].landmark

        # ROI oeil gauche
        x1L, y1L, x2L, y2L = _eye_roi_from_ids(lms, L_IDS, w_img, h_img)
        if x2L <= x1L or y2L <= y1L:
            continue

        # Centre détecté de l'iris gauche
        iris_L = get_iris_center(lms, LEFT_IRIS, w_img, h_img)
        if iris_L is None:
            continue
        det_x_abs, det_y_abs = iris_L

        # Coordonnées détectées normalisées dans la ROI
        det_cx = (det_x_abs - x1L) / max(1, (x2L - x1L))
        det_cy = (det_y_abs - y1L) / max(1, (y2L - y1L))

        # Construire l'image d'affichage
        disp = frame.copy()
        # ROI
        cv.rectangle(disp, (x1L, y1L), (x2L, y2L), (0, 255, 0), 2)
        # Centre détecté
        cv.circle(disp, (det_x_abs, det_y_abs), 4, (0, 0, 255), -1)
        cv.putText(disp, f"Frame {frame_idx} - Clique centre pupille GAUCHE",
                   (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.putText(disp, "s = skip, q = quit",
                   (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Reset clic
        global _click
        _click["flag"] = False

        while True:
            cv.imshow(win_name, disp)
            k = cv.waitKey(20) & 0xFF

            if k == ord('q'):
                cap.release()
                cv.destroyAllWindows()
                face_mesh.close()
                # calcul stats avant de sortir
                if errors:
                    errs = np.array(errors)
                    print("\n=== Statistiques erreur pupille (video annotee) ===")
                    print(f"Nombre de frames annotees : {len(errs)}")
                    print(f"Erreur moyenne normalisee (bar e) = {errs.mean():.4f}")
                    print(f"Ecart-type = {errs.std(ddof=1):.4f}")
                    print("===============================================\n")
                else:
                    print("Aucune frame annotee, pas de stats.")
                return

            if k == ord('s'):  # skip frame
                break

            if _click["flag"]:
                # Clic utilisateur = reference
                click_x = _click["x"]
                click_y = _click["y"]

                # Vérifier que le clic est dans la ROI
                if x1L <= click_x <= x2L and y1L <= click_y <= y2L:
                    ref_cx = (click_x - x1L) / max(1, (x2L - x1L))
                    ref_cy = (click_y - y1L) / max(1, (y2L - y1L))

                    # Erreur e_t dans la ROI normalisée
                    e_t = math.sqrt((det_cx - ref_cx) ** 2 + (det_cy - ref_cy) ** 2)
                    errors.append(e_t)

                    # Feedback visuel
                    cv.circle(disp, (click_x, click_y), 4, (0, 255, 0), -1)
                    cv.putText(disp, f"e_t = {e_t:.4f}", (20, 90),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv.imshow(win_name, disp)
                    cv.waitKey(300)  # petite pause pour voir le résultat
                else:
                    # clic hors ROI
                    cv.putText(disp, "Clic hors ROI ! (re-essaye ou s/q)",
                               (20, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv.imshow(win_name, disp)
                    cv.waitKey(300)

                break  # passer a la frame suivante

    # Fin de la vidéo
    cap.release()
    cv.destroyAllWindows()
    face_mesh.close()

    if errors:
        errs = np.array(errors)
        print("\n=== Statistiques erreur pupille (video annotee) ===")
        print(f"Nombre de frames annotees : {len(errs)}")
        print(f"Erreur moyenne normalisee (bar e) = {errs.mean():.4f}")
        print(f"Ecart-type = {errs.std(ddof=1):.4f}")
        print("===============================================\n")
    else:
        print("Aucune frame annotee, pas de stats.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True,
                        help="chemin vers la video a evaluer (ex: data/pupil_eval/video.mov)")
    args = parser.parse_args()
    main(args.video)