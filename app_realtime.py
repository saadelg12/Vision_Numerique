#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capturer le flux webcam, afficher le FPS, tracer le maillage facial (FaceMesh) et calibrer la caméra

Objectif
--------
- Ouvrir un flux caméra et afficher les images en temps réel.
- Calculer un FPS lissé pour estimer la latence du pipeline.
- Détecter un visage, tracer un maillage (landmarks) stable, afficher les contours en option.
- Délimiter des régions d'intérêt autour des yeux (ROI) pour préparer l'extraction du regard.
- Collecter des images d'échiquier puis calibrer la caméra (intrinsèques K + distorsion) et évaluer l'erreur de reprojection.

Lien avec le cours
------------------
- Constituer la première brique d'un système de vision par acquisition vidéo.
- Récupérer des points image 2D (landmarks) et, si besoin, convertir en pixels.
- Considérer ces points comme des « points image » du modèle sténopé (~ p ~ K [R|t] P).
- Utiliser un motif plan (échiquier) pour fournir des couples 3D↔2D et calibrer la caméra (intrinsèques + distorsion).
- Préparer la pose de tête (PnP) et le suivi du regard avec des paramètres caméras corrects.

Utilisation
-----------
# Flux simple (capture + FPS)
python app_realtime.py --mode live --camera 0

# Maillage facial (FaceMesh + FPS)
python app_realtime.py --mode facemesh --camera 0 [--contours] [--eyes]

# Collecte d'images d'échiquier (appuyer sur 's' quand les coins sont verts)
python app_realtime.py --mode calib_collect --camera 0 --nx 7 --ny 7 --square 1.0 --save_dir calib

# Calibration (lire les images de --save_dir, calculer K/dist et intrinsics.json)
python app_realtime.py --mode calibrate --nx 7 --ny 7 --square 1.0 --save_dir calib

# Undistort (validation visuelle avec intrinsics.json)
python app_realtime.py --mode undistort --camera 0 --save_dir calib

# Pose de tête (PnP) avec intrinsics.json
python app_realtime.py --mode headpose --camera 0 --save_dir calib

Options communes
----------------
--width 640 --height 360   # fixer une résolution pour gagner en FPS
q                          # quitter la fenêtre

Options calib_collect / calibrate
---------------------------------
--nx, --ny                 # nombre de coins intérieurs (horiz, vert)
--square                   # taille de case (m). Si inconnue (écran iPhone), mettre 1.0
--save_dir                 # dossier des images collectées (entrée) et du JSON (sortie)

Prérequis
---------
pip install opencv-python mediapipe numpy
"""

import sys
import os
import time
import argparse
import cv2 as cv
import mediapipe as mp
import numpy as np
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import deque


# ============================================================================
# SYSTÈME D'ÉTATS DE TRANSITION POUR LE REGARD
# ============================================================================

class GazeState(Enum):
    """Énumération des états de regard possibles."""
    CENTRE = "CENTRE"
    HAUT = "HAUT"
    BAS = "BAS"
    GAUCHE = "GAUCHE"
    DROITE = "DROITE"
    HAUT_GAUCHE = "HAUT-GAUCHE"
    HAUT_DROITE = "HAUT-DROITE"
    BAS_GAUCHE = "BAS-GAUCHE"
    BAS_DROITE = "BAS-DROITE"

    @classmethod
    def from_string(cls, s: str) -> 'GazeState':
        """Convertir une chaîne en GazeState."""
        mapping = {
            "CENTRE": cls.CENTRE,
            "HAUT": cls.HAUT,
            "BAS": cls.BAS,
            "GAUCHE": cls.GAUCHE,
            "DROITE": cls.DROITE,
            "HAUT-GAUCHE": cls.HAUT_GAUCHE,
            "HAUT-DROITE": cls.HAUT_DROITE,
            "BAS-GAUCHE": cls.BAS_GAUCHE,
            "BAS-DROITE": cls.BAS_DROITE,
        }
        return mapping.get(s, cls.CENTRE)

    def to_string(self) -> str:
        """Convertir GazeState en chaîne pour compatibilité JSON."""
        return self.value


@dataclass
class StateTransition:
    """
    Représente une transition entre deux états de regard.

    Attributes:
        from_state: État source
        to_state: État destination
        movement_vector: Vecteur de mouvement de référence (dx, dy, dpitch)
        min_distance: Distance minimale requise pour déclencher la transition
        confidence_threshold: Seuil de confiance pour valider la transition
        hysteresis: Facteur d'hystérésis pour éviter les oscillations
    """
    from_state: GazeState
    to_state: GazeState
    movement_vector: Tuple[float, float, float]  # (dx, dy, dpitch)
    min_distance: float = 0.02
    confidence_threshold: float = 0.7
    hysteresis: float = 1.2

    def compute_match_score(self, observed_movement: Tuple[float, float, float],
                           scale_y: float = 2.0, scale_pitch: float = 0.10) -> float:
        """
        Calcule un score de correspondance entre le mouvement observé et le vecteur de référence.

        Returns:
            Score entre 0 et 1 (1 = correspondance parfaite)
        """
        dx_ref, dy_ref, dpitch_ref = self.movement_vector
        dx_obs, dy_obs, dpitch_obs = observed_movement

        # Distance euclidienne pondérée
        dist_sq = (
            (dx_obs - dx_ref) ** 2 +
            (scale_y * (dy_obs - dy_ref)) ** 2 +
            (scale_pitch * (dpitch_obs - dpitch_ref)) ** 2
        )

        # Convertir en score de similarité (0-1)
        # Plus la distance est petite, plus le score est élevé
        max_dist = 1.0  # Distance de normalisation
        score = max(0.0, 1.0 - (dist_sq ** 0.5) / max_dist)
        return score

    def is_valid(self, observed_movement: Tuple[float, float, float],
                 is_returning: bool = False) -> bool:
        """
        Vérifie si la transition est valide pour le mouvement observé.

        Args:
            observed_movement: Mouvement observé (dx, dy, dpitch)
            is_returning: True si on revient à l'état précédent (hystérésis réduite)
        """
        dx, dy, _ = observed_movement
        movement_norm = (dx**2 + dy**2) ** 0.5

        # Appliquer l'hystérésis
        threshold = self.min_distance if is_returning else self.min_distance * self.hysteresis

        if movement_norm < threshold:
            return False

        score = self.compute_match_score(observed_movement)
        return score >= self.confidence_threshold


class GazeStateMachine:
    """
    Machine à états finie pour gérer les transitions de regard.

    Gère l'état courant, l'historique, et décide des transitions
    en fonction des mouvements observés et des règles définies.
    """

    def __init__(self, initial_state: GazeState = GazeState.CENTRE,
                 history_size: int = 10,
                 stability_frames: int = 3):
        """
        Args:
            initial_state: État initial
            history_size: Taille de l'historique des états
            stability_frames: Nombre de frames consécutives pour confirmer un changement
        """
        self.current_state = initial_state
        self.previous_state = initial_state
        self.anchor_features: Optional[List[float]] = None

        # Historique pour filtrage temporel
        self.state_history: deque = deque(maxlen=history_size)
        self.state_history.append(initial_state)

        # Compteur de stabilité
        self.stability_frames = stability_frames
        self.candidate_state: Optional[GazeState] = None
        self.candidate_count = 0

        # Matrice de transitions
        self.transitions: Dict[Tuple[GazeState, GazeState], StateTransition] = {}

        # Statistiques
        self.transition_count = 0
        self.last_transition_time = time.time()

    def build_transition_matrix(self, positions_data: Dict[str, dict],
                                scale_y: float = 2.0,
                                scale_pitch: float = 0.10):
        """
        Construit la matrice de transitions à partir des données de calibration.

        Args:
            positions_data: Données de pupil_positions_test.json
            scale_y: Facteur d'échelle pour les mouvements verticaux
            scale_pitch: Facteur d'échelle pour le pitch
        """
        states = [GazeState.from_string(s) for s in positions_data.keys()]

        def _avg_xy_pitch(pos_data):
            cxL = pos_data["left"]["x"]
            cyL = pos_data["left"]["y"]
            cxR = pos_data["right"]["x"]
            cyR = pos_data["right"]["y"]
            pitch = pos_data.get("pitch", 0.0)
            cx = (cxL + cxR) / 2.0
            cy = (cyL + cyR) / 2.0
            return cx, cy, pitch

        # Créer toutes les transitions possibles
        for from_state in states:
            fx, fy, fpitch = _avg_xy_pitch(positions_data[from_state.to_string()])

            for to_state in states:
                tx, ty, tpitch = _avg_xy_pitch(positions_data[to_state.to_string()])

                # Vecteur de mouvement
                dx = tx - fx
                dy = ty - fy
                dpitch = tpitch - fpitch

                # Paramètres adaptatifs selon le type de transition
                min_dist = self._compute_adaptive_threshold(from_state, to_state)
                confidence = self._compute_confidence_threshold(from_state, to_state)
                hysteresis = self._compute_hysteresis(from_state, to_state)

                transition = StateTransition(
                    from_state=from_state,
                    to_state=to_state,
                    movement_vector=(dx, dy, dpitch),
                    min_distance=min_dist,
                    confidence_threshold=confidence,
                    hysteresis=hysteresis
                )

                self.transitions[(from_state, to_state)] = transition

    def _compute_adaptive_threshold(self, from_state: GazeState, to_state: GazeState) -> float:
        """Calcule un seuil adaptatif selon le type de transition."""
        # Rester dans le même état = seuil très bas
        if from_state == to_state:
            return 0.005

        # Transition vers le centre = seuil plus permissif
        if to_state == GazeState.CENTRE:
            return 0.015

        # Transitions diagonales = seuil plus élevé
        if from_state in [GazeState.CENTRE] and to_state in [
            GazeState.HAUT_GAUCHE, GazeState.HAUT_DROITE,
            GazeState.BAS_GAUCHE, GazeState.BAS_DROITE
        ]:
            return 0.03

        # Défaut
        return 0.02

    def _compute_confidence_threshold(self, from_state: GazeState, to_state: GazeState) -> float:
        """Calcule le seuil de confiance selon le type de transition."""
        if from_state == to_state:
            return 0.5
        if to_state == GazeState.CENTRE:
            return 0.6
        return 0.7

    def _compute_hysteresis(self, from_state: GazeState, to_state: GazeState) -> float:
        """Calcule le facteur d'hystérésis."""
        # Retour au centre = moins d'hystérésis
        if to_state == GazeState.CENTRE:
            return 1.1
        # Transitions diagonales = plus d'hystérésis
        if to_state in [GazeState.HAUT_GAUCHE, GazeState.HAUT_DROITE,
                       GazeState.BAS_GAUCHE, GazeState.BAS_DROITE]:
            return 1.3
        return 1.2

    def update(self, current_features: List[float],
               neutral_radius: float = 0.01) -> Tuple[GazeState, bool]:
        """
        Met à jour la machine à états avec de nouvelles features.

        Args:
            current_features: Features actuelles [cxL, cyL, cxR, cyR, yaw, pitch]
            neutral_radius: Rayon de la zone neutre (pas de mouvement)

        Returns:
            (état_courant, transition_effectuée)
        """
        if self.anchor_features is None:
            self.anchor_features = current_features
            return self.current_state, False

        # Calculer le mouvement depuis l'ancre
        movement = self._compute_movement(current_features, self.anchor_features)
        dx_avg, dy_avg, dpitch = movement
        movement_norm = (dx_avg**2 + dy_avg**2) ** 0.5

        # Zone neutre : pas de mouvement significatif
        if movement_norm < neutral_radius:
            self.candidate_state = None
            self.candidate_count = 0
            return self.current_state, False

        # Trouver la meilleure transition
        best_state = self._find_best_transition(movement)

        # Filtrage temporel : nécessite plusieurs frames consécutives
        if best_state == self.candidate_state:
            self.candidate_count += 1
        else:
            self.candidate_state = best_state
            self.candidate_count = 1

        # Confirmer le changement d'état si stable
        if self.candidate_count >= self.stability_frames and best_state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = best_state
            self.anchor_features = current_features
            self.state_history.append(best_state)
            self.transition_count += 1
            self.last_transition_time = time.time()
            return self.current_state, True

        return self.current_state, False

    def _compute_movement(self, current_f: List[float], anchor_f: List[float]) -> Tuple[float, float, float]:
        """Calcule le vecteur de mouvement entre deux ensembles de features."""
        cxL, cyL, cxR, cyR, yaw, pitch = current_f
        axL, ayL, axR, ayR, ayaw, apitch = anchor_f

        dx_L = cxL - axL
        dy_L = cyL - ayL
        dx_R = cxR - axR
        dy_R = cyR - ayR

        dx_avg = (dx_L + dx_R) / 2.0
        dy_avg = (dy_L + dy_R) / 2.0
        dpitch = pitch - apitch

        return dx_avg, dy_avg, dpitch

    def _find_best_transition(self, movement: Tuple[float, float, float]) -> GazeState:
        """
        Trouve la meilleure transition pour le mouvement donné.

        Args:
            movement: Vecteur de mouvement (dx, dy, dpitch)

        Returns:
            État destination le plus probable
        """
        best_score = -float('inf')
        best_state = self.current_state

        # Évaluer toutes les transitions possibles depuis l'état courant
        for to_state in GazeState:
            key = (self.current_state, to_state)
            if key not in self.transitions:
                continue

            transition = self.transitions[key]
            is_returning = (to_state == self.previous_state)

            # Vérifier la validité de base
            if not transition.is_valid(movement, is_returning):
                continue

            # Calculer le score
            score = transition.compute_match_score(movement)

            # Bonus pour rester dans l'état courant (stabilité)
            if to_state == self.current_state:
                score += 0.1

            # Bonus pour revenir à l'état précédent
            if is_returning:
                score += 0.05

            if score > best_score:
                best_score = score
                best_state = to_state

        return best_state

    def reset(self, new_state: GazeState = GazeState.CENTRE):
        """Réinitialise la machine à états."""
        self.current_state = new_state
        self.previous_state = new_state
        self.anchor_features = None
        self.state_history.clear()
        self.state_history.append(new_state)
        self.candidate_state = None
        self.candidate_count = 0

    def get_state_duration(self) -> float:
        """Retourne la durée depuis la dernière transition (en secondes)."""
        return time.time() - self.last_transition_time

    def get_state_history(self, n: int = 5) -> List[GazeState]:
        """Retourne les n derniers états."""
        return list(self.state_history)[-n:]

    def get_statistics(self) -> Dict:
        """Retourne des statistiques sur les transitions."""
        return {
            "current_state": self.current_state.to_string(),
            "previous_state": self.previous_state.to_string(),
            "transition_count": self.transition_count,
            "state_duration": self.get_state_duration(),
            "history": [s.to_string() for s in self.get_state_history()],
        }


def _open_camera(index: int, width: int | None, height: int | None) -> cv.VideoCapture:
    """Ouvrir la caméra, fixer éventuellement la résolution et retourner l'objet VideoCapture."""
    cap = cv.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la caméra (essayer --camera 1 ou autre index).")
    if width:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, float(width))
    if height:
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap


def _fps_update(last_ts: float, fps: float) -> tuple[float, float]:
    """Calculer un FPS lissé (90 % ancien + 10 % instantané) et retourner (nouveau_timestamp, fps)."""
    now = time.time()
    if now > last_ts:
        inst = 1.0 / (now - last_ts)
        fps = 0.9 * fps + 0.1 * inst
    return now, fps


def _put_fps(frame, fps: float, y: int = 30):
    """Afficher le FPS en overlay dans l'image au point (10, y)."""
    cv.putText(frame, f"FPS: {fps:5.1f}", (10, y),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)


class GazeMapper:
    """Régression linéaire (ridge) pour mapper features -> écran."""

    def __init__(self, lam=1e-2):
        self.lam = float(lam)
        self.Wx = None
        self.Wy = None

    def fit(self, F, T):
        X = np.asarray(F, float)
        Y = np.asarray(T, float)
        d = X.shape[1]
        A = X.T @ X + self.lam * np.eye(d)
        bx = X.T @ Y[:, 0]
        by = X.T @ Y[:, 1]
        self.Wx = np.linalg.solve(A, bx)
        self.Wy = np.linalg.solve(A, by)

    def predict(self, f):
        f = np.asarray(f, float)
        x = float(f @ self.Wx)
        y = float(f @ self.Wy)
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        return x, y

    def to_dict(self):
        return {"lam": self.lam, "Wx": self.Wx.tolist(), "Wy": self.Wy.tolist()}

    @staticmethod
    def from_dict(d):
        import numpy as np
        m = GazeMapper(d.get("lam", 1e-2))
        m.Wx = np.asarray(d["Wx"], float)
        m.Wy = np.asarray(d["Wy"], float)
        return m


def _median_vec(samples):
    return np.median(np.asarray(samples, float), axis=0).tolist()


def _norm_in_roi(cx, cy, x1, y1, x2, y2):
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return cx / w, cy / h


def _open_canvas(win_name: str, fullscreen: bool = True, fallback=(1280, 720)):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    if fullscreen:
        cv.setWindowProperty(win_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    else:
        cv.resizeWindow(win_name, fallback[0], fallback[1])
    cv.imshow(win_name, 255 * np.ones((fallback[1], fallback[0], 3), dtype=np.uint8))
    cv.waitKey(30)
    try:
        x, y, w, h = cv.getWindowImageRect(win_name)
    except Exception:
        w, h = fallback
    return w, h


def _phi_quadratic(f):
    """Expansion quadratique: [1, f, f^2, termes croisés] pour une régression plus flexible."""
    f = np.asarray(f, float)
    feats = [1.0]
    feats.extend(f.tolist())
    feats.extend((f * f).tolist())
    for i in range(len(f)):
        for j in range(i+1, len(f)):
            feats.append(float(f[i] * f[j]))
    return feats


def _phi_cubic(f):
    """Expansion cubique pour meilleure précision périphérique."""
    f = np.asarray(f, float)
    feats = [1.0]
    feats.extend(f.tolist())
    feats.extend((f * f).tolist())
    feats.extend((f * f * f).tolist())

    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            feats.append(float(f[i] * f[j]))

    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            for k in range(j + 1, len(f)):
                feats.append(float(f[i] * f[j] * f[k]))

    return feats


class EMA2D:
    """Lissage exponentiel simple pour (x,y) en runtime."""

    def __init__(self, alpha=0.25):
        self.a = float(alpha)
        self.x = None
        self.y = None

    def update(self, x, y):
        if self.x is None:
            self.x, self.y = float(x), float(y)
        else:
            a = self.a
            self.x = (1-a)*self.x + a*float(x)
            self.y = (1-a)*self.y + a*float(y)
        return self.x, self.y


class OneEuro1D:
    """Filtre One-Euro 1D pour lissage adaptatif."""

    def __init__(self, min_cutoff=1.0, beta=0.005, dcutoff=1.0, freq=60.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.freq = float(freq)
        self.prev_x = None
        self.prev_dx = 0.0
        self.prev_time = None

    def _alpha(self, cutoff):
        import math
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / max(1e-6, self.freq)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.prev_x is None:
            self.prev_x = x
        dx = (x - self.prev_x) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.prev_dx
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.prev_x
        self.prev_x, self.prev_dx = x_hat, dx_hat
        return x_hat


class OneEuro2D:
    """Filtre One-Euro 2D pour lissage adaptatif."""

    def __init__(self, min_cutoff=1.0, beta=0.005, dcutoff=1.0, freq=60.0):
        self.fx = OneEuro1D(min_cutoff, beta, dcutoff, freq)
        self.fy = OneEuro1D(min_cutoff, beta, dcutoff, freq)

    def update(self, x, y):
        return self.fx(x), self.fy(y)


def extract_gaze_features(frame, lms, w_img, h_img, reject_blink=True, ear_thresh=0.20):
    """
    Retourne (base_f, ok) où base_f = [cxLn, cyLn, cxRn, cyRn, yaw, pitch, ear]
    Utilise les iris MediaPipe pour plus de précision.
    EAR (Eye Aspect Ratio) est ajouté pour améliorer la détection verticale.
    """
    # Calculer l'EAR (Eye Aspect Ratio) pour détecter les paupières
    try:
        ear = compute_EAR(lms, w_img, h_img, side="both")
    except Exception:
        ear = 0.25  # Valeur par défaut

    if reject_blink:
        if ear < ear_thresh:
            return None, False

    x1L, y1L, x2L, y2L = _eye_roi_from_ids(lms, L_IDS, w_img, h_img)
    x1R, y1R, x2R, y2R = _eye_roi_from_ids(lms, R_IDS, w_img, h_img)

    feat_ok = False
    cxLn = cyLn = cxRn = cyRn = 0.5

    center_L = get_iris_center(lms, LEFT_IRIS, w_img, h_img)
    if center_L and (x2L > x1L) and (y2L > y1L):
        cx_abs, cy_abs = center_L
        cxL = cx_abs - x1L
        cyL = cy_abs - y1L
        wL, hL = (x2L - x1L), (y2L - y1L)
        if 4 <= cxL <= wL - 4 and 4 <= cyL <= hL - 4:
            cxLn = cxL / max(1, wL)
            cyLn = 1.0 - (cyL / max(1, hL))  # Inverser Y : haut=1, bas=0
            feat_ok = True

    center_R = get_iris_center(lms, RIGHT_IRIS, w_img, h_img)
    if center_R and (x2R > x1R) and (y2R > y1R):
        cx_abs, cy_abs = center_R
        cxR = cx_abs - x1R
        cyR = cy_abs - y1R
        wR, hR = (x2R - x1R), (y2R - y1R)
        if 4 <= cxR <= wR - 4 and 4 <= cyR <= hR - 4:
            cxRn = cxR / max(1, wR)
            cyRn = 1.0 - (cyR / max(1, hR))  # Inverser Y : haut=1, bas=0
            feat_ok = True

    if not feat_ok:
        return None, False

    try:
        yaw, pitch = get_headpose_angles(lms, w_img, h_img)
    except Exception:
        exL = lms[L_IDS[0]]
        exR = lms[R_IDS[0]]
        yaw = (exR.x - exL.x)
        pitch = (lms[1].y - (exL.y + exR.y) * 0.5)

    base_f = [cxLn, cyLn, cxRn, cyRn, float(yaw), float(pitch), float(ear)]
    return base_f, True


def _clamp(v, a=0.0, b=1.0):
    return a if v < a else b if v > b else v


def lips_open_ratio(lms, w_img, h_img):
    """
    Renvoie un ratio d'ouverture de bouche normalisé ~[0,1].
    FaceMesh: lèvres internes (13 haut, 14 bas), commissures (61, 291).
    """
    import math
    u = lms[13]
    d = lms[14]
    lc = lms[61]
    rc = lms[291]
    u = (u.x * w_img, u.y * h_img)
    d = (d.x * w_img, d.y * h_img)
    lc = (lc.x * w_img, lc.y * h_img)
    rc = (rc.x * w_img, rc.y * h_img)
    mouth_h = math.dist(u, d)
    mouth_w = math.dist(lc, rc) + 1e-6
    r = mouth_h / mouth_w
    return _clamp((r - 0.10) / (0.30))


class AvatarSmoother:
    """Lissage pour paramètres avatar (One-Euro + petites bornes)."""

    def __init__(self, freq=60.0):
        self.gaze = OneEuro2D(min_cutoff=1.2, beta=0.02, dcutoff=1.0, freq=freq)
        self.yaw = OneEuro1D(min_cutoff=1.0, beta=0.02, dcutoff=1.0, freq=freq)
        self.pitch = OneEuro1D(min_cutoff=1.0, beta=0.02, dcutoff=1.0, freq=freq)
        self.eyeL = OneEuro1D(min_cutoff=1.5, beta=0.01, dcutoff=1.0, freq=freq)
        self.eyeR = OneEuro1D(min_cutoff=1.5, beta=0.01, dcutoff=1.0, freq=freq)
        self.mouth = OneEuro1D(min_cutoff=1.5, beta=0.01, dcutoff=1.0, freq=freq)

    def update(self, xn, yn, yaw, pitch, eyeL, eyeR, mouth):
        xn, yn = self.gaze.update(_clamp(xn), _clamp(yn))
        yaw = self.yaw(yaw)
        pitch = self.pitch(pitch)
        eyeL = self.eyeL(_clamp(eyeL))
        eyeR = self.eyeR(_clamp(eyeR))
        mouth = self.mouth(_clamp(mouth))
        return xn, yn, yaw, pitch, eyeL, eyeR, mouth


def draw_avatar(canvas, xn, yn, yaw, pitch, eyeL, eyeR, mouth, landmarks=None, w_img=None, h_img=None,
                eyebrow_ratio=0.0, expression="NEUTRE"):
    """
    Avatar 3D maillé réaliste utilisant les vrais landmarks MediaPipe avec rendu amélioré.

    Paramètres :
    - canvas : image de fond
    - xn, yn : regard normalisé [0..1]
    - yaw, pitch : rotation de tête
    - eyeL, eyeR : ouverture des yeux [0..1]
    - mouth : ouverture bouche [0..1]
    - landmarks : liste des landmarks MediaPipe (468 points)
    - w_img, h_img : dimensions de l'image source (pour scaling)
    - eyebrow_ratio : position des sourcils (pour animation)
    - expression : expression détectée
    """
    import cv2 as cv
    import numpy as np
    import mediapipe as mp

    H, W = canvas.shape[:2]
    # Fond dégradé subtil au lieu de noir uni
    for y in range(H):
        intensity = int(15 + (y / H) * 10)
        canvas[y, :] = (intensity, intensity, intensity + 5)

    if landmarks is None or w_img is None or h_img is None:
        cx, cy = W // 2, H // 2
        cv.putText(canvas, "NO FACE DETECTED", (W//2 - 150, H//2),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 200), 2, cv.LINE_AA)
        return

    def lm_to_canvas(lm):
        """Convertit un landmark MediaPipe en coordonnées canvas"""
        x = int(lm.x * W)
        y = int(lm.y * H)
        return (x, y)

    mp_face_mesh = mp.solutions.face_mesh
    FACEMESH_TESSELATION = mp_face_mesh.FACEMESH_TESSELATION
    FACEMESH_CONTOURS = mp_face_mesh.FACEMESH_CONTOURS

    # ---- Rendu avec couleurs de peau réalistes ----
    # Couleur de peau de base (teinte chair: BGR)
    SKIN_BASE_COLOR = np.array([120, 160, 200])  # Peau claire (ajustable)

    for connection in FACEMESH_TESSELATION:
        pt1 = lm_to_canvas(landmarks[connection[0]])
        pt2 = lm_to_canvas(landmarks[connection[1]])

        z1 = landmarks[connection[0]].z
        z2 = landmarks[connection[1]].z
        avg_z = (z1 + z2) / 2

        # Profondeur influence l'intensité (ombrage)
        depth_factor = max(0.4, min(1.2, 1.0 - avg_z * 2.0))

        # Appliquer le facteur de profondeur à la couleur de peau
        skin_color = tuple(int(c * depth_factor) for c in SKIN_BASE_COLOR)

        cv.line(canvas, pt1, pt2, skin_color, 1, cv.LINE_AA)

    # Contours plus marqués avec couleur chair foncée
    for connection in FACEMESH_CONTOURS:
        pt1 = lm_to_canvas(landmarks[connection[0]])
        pt2 = lm_to_canvas(landmarks[connection[1]])
        cv.line(canvas, pt1, pt2, (80, 120, 150), 2, cv.LINE_AA)

    LEFT_EYE = [33, 160, 158, 133, 153, 144, 145, 22, 23, 25, 154, 31, 228, 229, 230, 231]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380, 381, 252, 253, 255, 359, 261, 448, 449, 450, 451]
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    def draw_enhanced_eye(eye_landmarks, iris_landmarks, openness, is_left=True):
        """Dessine un oeil avec iris coloré et pupille réaliste"""
        eye_points = [lm_to_canvas(landmarks[i]) for i in eye_landmarks if i < len(landmarks)]

        # ---- Dessiner le blanc de l'œil (sclère) ----
        if len(eye_points) > 0 and openness > 0.3:
            eye_pts = np.array(eye_points, np.int32)
            cv.fillPoly(canvas, [eye_pts], (220, 220, 240))  # Blanc légèrement bleuté

        # ---- Dessiner l'iris et la pupille (uniquement si œil suffisamment ouvert) ----
        if iris_landmarks[0] < len(landmarks) and len(eye_points) > 0 and openness > 0.4:
            iris_pts = [lm_to_canvas(landmarks[i]) for i in iris_landmarks]
            if len(iris_pts) > 0:
                iris_center = np.mean(iris_pts, axis=0).astype(int)

                # Déplacement de l'iris selon le regard
                gx = (xn - 0.5) * 2.0
                gy = (yn - 0.5) * 2.0
                iris_center[0] += int(gx * 12)
                iris_center[1] += int(gy * 12)

                # Rayon de l'iris et de la pupille
                iris_r = 12
                pupil_r = int(iris_r * 0.4)

                # Couleur de l'iris (marron réaliste)
                iris_color = (50, 90, 120)  # Marron/noisette (BGR)

                # Dessiner l'iris avec dégradé
                for r in range(iris_r, iris_r - 4, -1):
                    alpha = (iris_r - r) / 4.0
                    color_blend = tuple(int(c * (1 - alpha * 0.3)) for c in iris_color)
                    cv.circle(canvas, tuple(iris_center), r, color_blend, -1, cv.LINE_AA)

                # Dessiner la pupille (noire)
                cv.circle(canvas, tuple(iris_center), pupil_r, (0, 0, 0), -1, cv.LINE_AA)

                # Reflet lumineux dans la pupille (petit point blanc)
                reflect_offset = (int(pupil_r * 0.3), int(-pupil_r * 0.3))
                reflect_pos = (iris_center[0] + reflect_offset[0], iris_center[1] + reflect_offset[1])
                cv.circle(canvas, reflect_pos, 2, (255, 255, 255), -1, cv.LINE_AA)

        # ---- Animation de fermeture des yeux (paupière) ----
        if openness < 0.9 and len(eye_points) > 0:
            overlay = canvas.copy()
            eye_pts = np.array(eye_points, np.int32)

            # Calculer le niveau de fermeture
            close_amount = 1.0 - openness

            # Assombrir progressivement l'œil
            alpha = min(close_amount * 1.5, 1.0)
            cv.fillPoly(overlay, [eye_pts], (80, 110, 130))  # Couleur paupière
            cv.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    draw_enhanced_eye(LEFT_EYE, LEFT_IRIS, eyeL, is_left=True)
    draw_enhanced_eye(RIGHT_EYE, RIGHT_IRIS, eyeR, is_left=False)

    # ---- Dessiner les sourcils animés ----
    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300, 285, 295]

    def draw_animated_eyebrow(eyebrow_landmarks, is_left=True):
        """Dessine un sourcil animé selon l'expression"""
        brow_pts = [lm_to_canvas(landmarks[i]) for i in eyebrow_landmarks if i < len(landmarks)]
        if len(brow_pts) > 0:
            brow_array = np.array(brow_pts, np.int32)

            # Pas de décalage vertical (sourcils statiques)
            offset_y = 0

            # Ajuster position selon expression
            brow_array[:, 1] += offset_y

            # Couleur sourcil (marron foncé)
            eyebrow_color = (30, 50, 70)

            # Dessiner le sourcil avec plusieurs lignes pour épaisseur
            for thickness in [5, 3, 1]:
                alpha_val = 1.0 - (thickness / 6.0)
                color = tuple(int(c * alpha_val) for c in eyebrow_color)
                cv.polylines(canvas, [brow_array], False, color, thickness, cv.LINE_AA)

    draw_animated_eyebrow(LEFT_EYEBROW, is_left=True)
    draw_animated_eyebrow(RIGHT_EYEBROW, is_left=False)

    # ---- Dessiner la bouche avec séparation naturelle ----
    # Lèvre supérieure (contour extérieur + ligne médiane)
    UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    UPPER_LIP_INNER = [61, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 291]

    # Lèvre inférieure (contour extérieur + ligne médiane)
    LOWER_LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    LOWER_LIP_INNER = [61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291]

    upper_lip_outer = [lm_to_canvas(landmarks[i]) for i in UPPER_LIP_OUTER if i < len(landmarks)]
    upper_lip_inner = [lm_to_canvas(landmarks[i]) for i in UPPER_LIP_INNER if i < len(landmarks)]
    lower_lip_outer = [lm_to_canvas(landmarks[i]) for i in LOWER_LIP_OUTER if i < len(landmarks)]
    lower_lip_inner = [lm_to_canvas(landmarks[i]) for i in LOWER_LIP_INNER if i < len(landmarks)]

    if len(upper_lip_outer) > 0 and len(lower_lip_outer) > 0:
        # Couleur des lèvres selon expression
        base_upper_color = (80, 100, 150)  # Lèvre supérieure (plus foncée)
        base_lower_color = (95, 115, 165)  # Lèvre inférieure (plus claire)

        if expression == "SOURIRE":
            upper_color = (90, 120, 180)
            lower_color = (105, 135, 195)
        elif expression == "TRISTESSE":
            upper_color = (70, 85, 130)
            lower_color = (80, 95, 140)
        else:
            upper_color = base_upper_color
            lower_color = base_lower_color

        # Dessiner lèvre supérieure (forme fermée)
        if len(upper_lip_inner) > 0:
            upper_shape = np.array(upper_lip_outer + upper_lip_inner[::-1], np.int32)
            cv.fillPoly(canvas, [upper_shape], upper_color)

        # Dessiner lèvre inférieure (forme fermée)
        if len(lower_lip_inner) > 0:
            lower_shape = np.array(lower_lip_outer + lower_lip_inner[::-1], np.int32)
            cv.fillPoly(canvas, [lower_shape], lower_color)

        # *** LIGNE DE SÉPARATION ENTRE LES LÈVRES (importante !) ***
        # Utiliser les points de la ligne médiane commune
        separation_line = [lm_to_canvas(landmarks[i]) for i in [61, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 291] if i < len(landmarks)]
        if len(separation_line) > 0:
            cv.polylines(canvas, [np.array(separation_line, np.int32)], False, (50, 60, 100), 2, cv.LINE_AA)

        # Contours extérieurs des lèvres
        cv.polylines(canvas, [np.array(upper_lip_outer, np.int32)], False, (50, 65, 110), 2, cv.LINE_AA)
        cv.polylines(canvas, [np.array(lower_lip_outer, np.int32)], False, (50, 65, 110), 2, cv.LINE_AA)

        # Ligne verticale centrale (si bouche ouverte)
        if mouth > 0.2:
            mouth_center_pts = [lm_to_canvas(landmarks[i]) for i in [13, 14] if i < len(landmarks)]
            if len(mouth_center_pts) == 2:
                cv.line(canvas, mouth_center_pts[0], mouth_center_pts[1],
                       (30, 40, 60), 3, cv.LINE_AA)

    KEY_POINTS = [1, 10, 152, 33, 263, 61, 291]

    for idx in KEY_POINTS:
        if idx < len(landmarks):
            pt = lm_to_canvas(landmarks[idx])
            cv.circle(canvas, pt, 4, (0, 255, 200), -1, cv.LINE_AA)
            cv.circle(canvas, pt, 7, (0, 200, 150), 1, cv.LINE_AA)

    info_x = 20
    info_y = H - 140
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color_text = (0, 255, 200)

    cv.putText(canvas, f"GAZE: [{xn:.2f}, {yn:.2f}]", (info_x, info_y),
               font, font_scale, color_text, 1, cv.LINE_AA)
    cv.putText(canvas, f"HEAD: YAW={yaw:+.2f} PITCH={pitch:+.2f}", (info_x, info_y + 20),
               font, font_scale, color_text, 1, cv.LINE_AA)
    cv.putText(canvas, f"EYES: L={eyeL:.2f} R={eyeR:.2f}", (info_x, info_y + 40),
               font, font_scale, color_text, 1, cv.LINE_AA)
    cv.putText(canvas, f"MOUTH: {mouth:.2f}", (info_x, info_y + 60),
               font, font_scale, color_text, 1, cv.LINE_AA)
    cv.putText(canvas, f"LANDMARKS: {len(landmarks)}", (info_x, info_y + 80),
               font, font_scale, (100, 100, 255), 1, cv.LINE_AA)


L_IDS = [33, 160, 158, 133, 153, 144]
R_IDS = [263, 387, 385, 362, 380, 373]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def _eye_roi_from_ids(lms, ids, w: int, h: int, scale: float = 1.6, scale_x: float = None, scale_y: float = None) -> tuple[int, int, int, int]:
    """
    Calculer une ROI rectangulaire centrée sur l'oeil, avec marges (scale_x, scale_y).
    Si scale_x/scale_y non spécifiés, utilise scale pour les deux.
    scale_y plus grand permet de capturer plus de mouvement vertical.
    """
    if scale_x is None:
        scale_x = scale
    if scale_y is None:
        scale_y = scale * 2.0  # Doubler la hauteur par défaut pour plus de mouvement vertical

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
    """Obtenir le centre de l'iris depuis les landmarks MediaPipe."""
    if iris_ids[0] >= len(lms):
        return None
    iris_points = np.array([
        (lms[i].x * w_img, lms[i].y * h_img)
        for i in iris_ids if i < len(lms)
    ])
    if len(iris_points) == 0:
        return None
    center = np.mean(iris_points, axis=0)
    return int(center[0]), int(center[1])


def run_live(cam_index: int, width: int | None, height: int | None) -> None:
    """Valider la chaîne d'acquisition (caméra -> fenêtre), calculer/afficher le FPS et gérer la sortie clavier."""
    cap = _open_camera(cam_index, width, height)
    last, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        last, fps = _fps_update(last, fps)
        _put_fps(frame, fps)

        cv.imshow("Capture", frame)
        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


import cv2 as cv
import time
import csv
from pathlib import Path


def run_facemesh(
    cam_index: int,
    width: int | None,
    height: int | None,
    show_contours: bool = False,
    show_eyes: bool = False,
    save_dir: str = "."
) -> None:
    """
    Détecter un visage, tracer la tessellation des landmarks et, en option,
    les contours et les ROI des yeux, et enregistrer les coordonnées des yeux dans un fichier CSV.
    """
    if mp is None:
        raise SystemExit("MediaPipe n'est pas installé. Exécuter : pip install mediapipe")

    cap = _open_camera(cam_index, width, height)
    frame_id = 0

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    csv_path = save_path / "coordonnees_yeux.csv"

    with open(csv_path, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["frame_id", "oeil", "x", "y"])

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

        draw = mp.solutions.drawing_utils
        mesh_style = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        contour_style = (mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                         if show_contours else None)

        last, fps = time.time(), 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]

                draw.draw_landmarks(
                    image=frame,
                    landmark_list=face,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mesh_style
                )

                if show_contours and contour_style is not None:
                    draw.draw_landmarks(
                        image=frame,
                        landmark_list=face,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=contour_style
                    )

                if show_eyes:
                    h_img, w_img = frame.shape[:2]
                    lms = face.landmark

                    x1, y1, x2, y2 = _eye_roi_from_ids(lms, L_IDS, w_img, h_img)
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    center_L = get_iris_center(lms, LEFT_IRIS, w_img, h_img)
                    if center_L:
                        cv.circle(frame, center_L, 3, (0, 0, 255), -1)
                        frame_id += 1
                        print(f"Oeil gauche: {center_L}")
                        csvwriter.writerow([frame_id, "gauche", center_L[0], center_L[1]])

                    x1, y1, x2, y2 = _eye_roi_from_ids(lms, R_IDS, w_img, h_img)
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    center_R = get_iris_center(lms, RIGHT_IRIS, w_img, h_img)
                    if center_R:
                        cv.circle(frame, center_R, 3, (0, 0, 255), -1)
                        frame_id += 1
                        print(f"Oeil droit: {center_R}")
                        csvwriter.writerow([frame_id, "droit", center_R[0], center_R[1]])

            last, fps = _fps_update(last, fps)
            _put_fps(frame, fps)

            cv.imshow("Maillage facial", frame)
            if (cv.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()
    print(f"CSV sauvegardé ici : {csv_path}")


def run_pupil(cam_index: int, width: int | None, height: int | None, save_dir: str = ".") -> None:
    """
    Detecter la pupille dans chaque oeil, afficher le centre sur la video,
    calculer un point de regard moyen (gaze) et sauvegarder les coordonnées dans un CSV.
    """
    if mp is None:
        raise SystemExit("MediaPipe n'est pas installé. Exécuter : pip install mediapipe")

    cap = _open_camera(cam_index, width, height)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )

    from pathlib import Path
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    csv_path = save_path / "coordonnees_yeux_pupil.csv"

    import csv
    csvfile = open(csv_path, mode='w', newline='')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["frame_id", "oeil", "x", "y"])

    last, fps = time.time(), 0.0
    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h_img, w_img = frame.shape[:2]

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = face_mesh.process(rgb)
            rgb.flags.writeable = True

            l_center = None
            r_center = None
            l_rect = None
            r_rect = None

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                lms = face.landmark

                x1, y1, x2, y2 = _eye_roi_from_ids(lms, L_IDS, w_img, h_img)
                l_rect = (x1, y1, x2, y2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                center_L = get_iris_center(lms, LEFT_IRIS, w_img, h_img)
                if center_L:
                    l_center = (center_L[0] - x1, center_L[1] - y1)
                    cv.circle(frame, center_L, 3, (0, 255, 0), -1)
                    frame_id += 1
                    csvwriter.writerow([frame_id, "gauche", center_L[0], center_L[1]])

                x1, y1, x2, y2 = _eye_roi_from_ids(lms, R_IDS, w_img, h_img)
                r_rect = (x1, y1, x2, y2)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                center_R = get_iris_center(lms, RIGHT_IRIS, w_img, h_img)
                if center_R:
                    r_center = (center_R[0] - x1, center_R[1] - y1)
                    cv.circle(frame, center_R, 3, (0, 255, 0), -1)
                    frame_id += 1
                    csvwriter.writerow([frame_id, "droit", center_R[0], center_R[1]])

                gaze = compute_gaze_vector(l_center, r_center, l_rect, r_rect)
                if gaze:
                    gx, gy = gaze
                    cv.circle(frame, (gx, gy), 5, (255, 0, 255), -1)
                    cv.putText(frame, f"Gaze: ({gx},{gy})", (10, 150),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv.LINE_AA)

            last, fps = _fps_update(last, fps)
            _put_fps(frame, fps)

            cv.imshow("Pupil + Gaze detection (press q to quit)", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        csvfile.close()
        cap.release()
        cv.destroyAllWindows()
        face_mesh.close()
        print(f"[FIN] CSV sauvegardé ici : {csv_path}")


def compute_gaze_vector(l_eye_center: tuple[int, int] | None, r_eye_center: tuple[int, int] | None,
                        l_eye_rect: tuple[int, int, int, int] | None, r_eye_rect: tuple[int, int, int, int] | None):
    """
    Calculer un vecteur de regard 2D approximatif à partir des centres des pupilles.
    Retourne (gx, gy) en pixels dans l'image, ou None si pas détecté.
    """
    gaze_points = []

    if l_eye_center and l_eye_rect:
        x1, y1, _, _ = l_eye_rect
        gaze_points.append((x1 + l_eye_center[0], y1 + l_eye_center[1]))

    if r_eye_center and r_eye_rect:
        x1, y1, _, _ = r_eye_rect
        gaze_points.append((x1 + r_eye_center[0], y1 + r_eye_center[1]))

    if not gaze_points:
        return None

    gx = int(sum(p[0] for p in gaze_points) / len(gaze_points))
    gy = int(sum(p[1] for p in gaze_points) / len(gaze_points))
    return gx, gy


def run_calib_collect(
    cam_index: int,
    width: int | None,
    height: int | None,
    nx: int,
    ny: int,
    square_m: float,
    save_dir: str,
    prefix: str,
    max_images: int
) -> None:
    """Détecter un échiquier, afficher les coins et enregistrer des vues valides en appyant sur 's'."""
    os.makedirs(save_dir, exist_ok=True)

    cap = _open_camera(cam_index, width, height)
    last, fps = time.time(), 0.0
    saved = 0
    pattern_size = (nx, ny)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        found, corners = cv.findChessboardCorners(
            gray, pattern_size,
            flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
        )

        if found:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame, pattern_size, corners, found)
            cv.putText(frame, "ECHIQUIER OK - Appuyer sur 's' pour sauver", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
        else:
            cv.putText(frame, "Chercher echiquier (placer a bonne distance, bonne lumiere)", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv.LINE_AA)

        last, fps = _fps_update(last, fps)
        _put_fps(frame, fps)

        cv.putText(frame, f"Saved: {saved}/{max_images} | pattern {nx}x{ny} | square={square_m}m",
                   (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, f"Dir: {save_dir}  Prefix: {prefix}", (10, 115),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)

        cv.imshow("Calibration - Collecte echiquier", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and found:
            fname = os.path.join(save_dir, f"{prefix}{saved:03d}.png")
            cv.imwrite(fname, frame)
            saved += 1
            print(f"[OK] Sauvegarde {fname}")
            if saved >= max_images:
                print("[INFO] Quota atteint, fin de la collecte.")
                break

    cap.release()
    cv.destroyAllWindows()
    print(f"[FIN] Images sauvegardées dans: {os.path.abspath(save_dir)} — total: {saved}")


def run_calibrate(
    nx: int,
    ny: int,
    square_m: float,
    save_dir: str
) -> None:
    """Lire les images d'échiquier dans save_dir, calibrer la caméra et écrire calib/intrinsics.json."""
    import glob
    import json

    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(save_dir, ext)))
    files = sorted(files)
    if len(files) == 0:
        raise SystemExit(f"Aucune image trouvée dans {save_dir} (png/jpg). Lancer d'abord calib_collect.")

    pattern_size = (nx, ny)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square_m

    objpoints = []
    imgpoints = []
    image_size = None
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    used = 0
    for f in files:
        img = cv.imread(f)
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv.findChessboardCorners(
            gray, pattern_size,
            flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
        )
        if not found:
            print(f"[WARN] Motif non trouvé dans {os.path.basename(f)} — ignoré.")
            continue

        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used += 1

    if used < 6:
        raise SystemExit(f"Trop peu d'images valides ({used}) pour calibrer. Viser >= 10-15.")

    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    print(f"[INFO] RMS OpenCV (global) = {ret:.4f} px")

    tot_err = 0.0
    tot_pts = 0
    per_view = []
    for i, (objp_i, imgp_i, rvec, tvec) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs)):
        proj, _ = cv.projectPoints(objp_i, rvec, tvec, K, dist)
        err = cv.norm(imgp_i, proj, cv.NORM_L2) / len(proj)
        per_view.append(float(err))
        tot_err += err * len(proj)
        tot_pts += len(proj)

    mean_err = tot_err / tot_pts
    print(f"[INFO] Erreur moyenne de reprojection = {mean_err:.4f} px")
    print(f"[INFO] Vues utilisées = {used} / {len(files)} ; image_size = {image_size}")

    out_json = {
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "rms": float(ret),
        "mean_reproj_px": float(mean_err),
        "per_view_reproj_px": per_view,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "pattern": {"nx": int(nx), "ny": int(ny), "square_m": float(square_m)},
        "n_images_used": int(used),
    }
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "intrinsics.json")
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"[OK] intrinsics.json -> {os.path.abspath(out_path)}")
    print("[TIP] Utiliser ensuite un mode undistort (à venir) ou pnp pour valider visuellement.")


def _load_intrinsics(save_dir: str):
    """Charger calib/intrinsics.json et retourner (K, dist, image_size)."""
    import json
    path = os.path.join(save_dir, "intrinsics.json")
    if not os.path.isfile(path):
        raise SystemExit(f"Intrinsics manquant: {path}. Lancer --mode calibrate d'abord.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
    img_size = tuple(data.get("image_size", [0, 0]))
    return K, dist, img_size


def run_undistort(cam_index: int, width: int | None, height: int | None, save_dir: str) -> None:
    """Afficher le flux original et le flux corrigé (undistort) côte à côte pour valider K/dist."""
    K, dist, _ = _load_intrinsics(save_dir)

    cap = _open_camera(cam_index, width, height)
    last, fps = time.time(), 0.0

    newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (int(cap.get(3)), int(cap.get(4))), 1)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        und = cv.undistort(frame, K, dist, None, newK)

        h = min(frame.shape[0], und.shape[0])
        left = cv.resize(frame, (int(frame.shape[1]*h/frame.shape[0]), h))
        right = cv.resize(und, (int(und.shape[1]*h/und.shape[0]), h))
        combo = np.hstack([left, right])

        last, fps = _fps_update(last, fps)
        _put_fps(combo, fps, y=30)

        cv.putText(combo, "Original", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(combo, "Undistort", (left.shape[1]+10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("Undistort - Validation visuelle", combo)
        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def eye_aspect_ratio(landmarks, ids):
    """Calcule l'Eye Aspect Ratio pour un oeil donné."""
    import numpy as np
    p = np.array([[landmarks[i].x, landmarks[i].y] for i in ids])
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    ear = (A + B) / (2.0 * max(C, 1e-6))
    return ear


def compute_EAR(lms, w_img=None, h_img=None, side: str = "both") -> float:
    """
    Calcule l'EAR (Eye Aspect Ratio) à partir des landmarks MediaPipe.
    - side: "left", "right" ou "both" (moyenne des deux)
    - w_img/h_img sont acceptés pour compat signature mais non utilisés (coords normalisées).
    """
    def _ear(ids):
        return eye_aspect_ratio(lms, ids)

    if side == "left":
        return float(_ear(L_IDS))
    elif side == "right":
        return float(_ear(R_IDS))
    elif side == "both":
        return 0.5 * (float(_ear(L_IDS)) + float(_ear(R_IDS)))
    else:
        raise ValueError("side must be 'left', 'right', or 'both'")


def run_blink_click(cam_index: int, width: int | None, height: int | None) -> None:
    """Détecte les clignements et fait un clic à chaque blink (anti-rebond) avec affichage EAR."""
    import pyautogui
    if mp is None:
        raise SystemExit("MediaPipe n'est pas installé. Exécuter : pip install mediapipe")

    cap = _open_camera(cam_index, width, height)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    )

    EAR_THRESH = 0.40
    EAR_CONSEC_FRAMES = 2

    blink_counter = 0
    blink_detected = False
    click_frames = 0

    last, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h_img, w_img = frame.shape[:2]

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = face_mesh.process(rgb)
        rgb.flags.writeable = True

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark

            l_ear = eye_aspect_ratio(lms, L_IDS)
            r_ear = eye_aspect_ratio(lms, R_IDS)
            ear = (l_ear + r_ear) / 2.0

            cv.putText(frame, f"L_EAR: {l_ear:.2f} R_EAR: {r_ear:.2f} Avg: {ear:.2f}",
                       (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if ear < EAR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EAR_CONSEC_FRAMES:
                    blink_detected = True
                blink_counter = 0

            if blink_detected:
                pyautogui.click()
                blink_detected = False
                click_frames = 5

        if click_frames > 0:
            cv.putText(frame, "CLICK!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            click_frames -= 1

        last, fps = _fps_update(last, fps)
        _put_fps(frame, fps)

        cv.imshow("Blink Click (press q to quit)", frame)
        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    face_mesh.close()


_FM = {
    "nose_tip": 1,
    "chin": 199,
    "eye_l": 33,
    "eye_r": 263,
    "mouth_l": 61,
    "mouth_r": 291,
}
_MODEL3D = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)


def _euler_from_rvec(rvec: np.ndarray) -> tuple[float, float, float]:
    """Convertir rvec (Rodrigues) en angles Euler (yaw, pitch, roll) en degrés (conv yaw-Z, pitch-Y, roll-X)."""
    R, _ = cv.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])
    singular = sy < 1e-6
    if not singular:
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:
        yaw = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0
    return float(yaw), float(pitch), float(roll)


def get_headpose_angles(lms, w_img: int, h_img: int):
    """
    Retourne (yaw, pitch) en degrés à partir des landmarks MediaPipe FaceMesh.
    Utilise solvePnP sur 6 points stables : nez, menton, coins externes des yeux,
    coins de bouche. Hypothèse simple: focale ~ largeur d'image, centre = image center.
    """
    import numpy as np
    import cv2 as cv

    f = float(w_img)
    cx, cy = w_img * 0.5, h_img * 0.5
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)

    pts2d = np.array([
        [lms[_FM["nose_tip"]].x * w_img, lms[_FM["nose_tip"]].y * h_img],
        [lms[_FM["chin"]].x * w_img, lms[_FM["chin"]].y * h_img],
        [lms[_FM["eye_l"]].x * w_img, lms[_FM["eye_l"]].y * h_img],
        [lms[_FM["eye_r"]].x * w_img, lms[_FM["eye_r"]].y * h_img],
        [lms[_FM["mouth_l"]].x * w_img, lms[_FM["mouth_l"]].y * h_img],
        [lms[_FM["mouth_r"]].x * w_img, lms[_FM["mouth_r"]].y * h_img],
    ], dtype=np.float64)

    ok, rvec, tvec = cv.solvePnP(
        _MODEL3D, pts2d, K, dist, flags=cv.SOLVEPNP_ITERATIVE
    )
    if not ok:
        exL = lms[_FM["eye_l"]]
        exR = lms[_FM["eye_r"]]
        yaw = (exR.x - exL.x)
        pitch = (lms[1].y - (exL.y + exR.y) * 0.5)
        return float(yaw), float(pitch)

    yaw, pitch, _roll = _euler_from_rvec(rvec)
    return float(yaw), float(pitch)


def get_headpose_anglesas(lms, w_img: int, h_img: int):
    return get_headpose_angles(lms, w_img, h_img)


def run_headpose(cam_index: int, width: int | None, height: int | None, save_dir: str) -> None:
    """Estimer la pose de tête en temps réel (solvePnP) et afficher yaw/pitch/roll + axes + erreur reprojection.
    À la fin, affiche aussi des statistiques (moyenne / écart-type / latence)."""
    if mp is None:
        raise SystemExit("MediaPipe n'est pas installé. Exécuter : pip install mediapipe")

    K, dist, _ = _load_intrinsics(save_dir)

    cap = _open_camera(cam_index, width, height)
    mp_face_mesh = mp.solutions.face_mesh
    fm = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    last, fps = time.time(), 0.0

    axes3D = np.float64([[100, 0, 0],
                         [0, 100, 0],
                         [0, 0, 100]]).reshape(-1, 3)

    # =======================
    # LISTES POUR LES STATS
    # =======================
    yaw_list: list[float] = []
    pitch_list: list[float] = []
    err_list: list[float] = []
    latency_ms_list: list[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = fm.process(rgb)
        rgb.flags.writeable = True

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            pts2d = np.array([
                [lms[_FM["nose_tip"]].x * w, lms[_FM["nose_tip"]].y * h],
                [lms[_FM["chin"]].x * w,     lms[_FM["chin"]].y * h],
                [lms[_FM["eye_l"]].x * w,    lms[_FM["eye_l"]].y * h],
                [lms[_FM["eye_r"]].x * w,    lms[_FM["eye_r"]].y * h],
                [lms[_FM["mouth_l"]].x * w,  lms[_FM["mouth_l"]].y * h],
                [lms[_FM["mouth_r"]].x * w,  lms[_FM["mouth_r"]].y * h],
            ], dtype=np.float64)

            # On mesure la latence pour solvePnP + projectPoints
            t0 = time.time()
            okpnp, rvec, tvec = cv.solvePnP(
                _MODEL3D, pts2d, K, dist, flags=cv.SOLVEPNP_ITERATIVE
            )
            if okpnp:
                proj, _ = cv.projectPoints(_MODEL3D, rvec, tvec, K, dist)
                t1 = time.time()
                latency_ms = (t1 - t0) * 1000.0  # en millisecondes

                yaw, pitch, roll = _euler_from_rvec(rvec)

                # erreur de reprojection moyenne sur les 6 points
                err = cv.norm(pts2d.reshape(-1, 1, 2), proj, cv.NORM_L2) / len(proj)

                # =======================
                # ON STOCKE POUR LES STATS
                # =======================
                yaw_list.append(yaw)
                pitch_list.append(pitch)
                err_list.append(err)
                latency_ms_list.append(latency_ms)

                # Dessin des axes 3D projetés
                nose = pts2d[0].reshape(1, 1, 2)
                axes2D, _ = cv.projectPoints(axes3D, rvec, tvec, K, dist)
                origin = (int(nose[0, 0, 0]), int(nose[0, 0, 1]))
                X = (int(axes2D[0, 0, 0]), int(axes2D[0, 0, 1]))
                Y = (int(axes2D[1, 0, 0]), int(axes2D[1, 0, 1]))
                Z = (int(axes2D[2, 0, 0]), int(axes2D[2, 0, 1]))
                cv.line(frame, origin, X, (0, 0, 255), 2)
                cv.line(frame, origin, Y, (0, 255, 0), 2)
                cv.line(frame, origin, Z, (255, 0, 0), 2)

                cv.putText(
                    frame,
                    f"yaw {yaw:+5.1f}  pitch {pitch:+5.1f}  roll {roll:+5.1f}  | reproj {err:.2f}px",
                    (10, 70),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv.LINE_AA
                )

                for (u, v) in pts2d:
                    cv.circle(frame, (int(u), int(v)), 3, (0, 255, 255), -1)

        last, fps = _fps_update(last, fps)
        _put_fps(frame, fps)
        cv.imshow("Head pose (PnP)", frame)
        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    # =======================
    # STATS FINALES POUR LE RAPPORT
    # =======================
    if len(yaw_list) > 0:
        yaw_arr = np.array(yaw_list)
        pitch_arr = np.array(pitch_list)
        err_arr = np.array(err_list)
        lat_arr = np.array(latency_ms_list)

        print("\n=== Statistiques head pose (tête immobile) ===")
        print(f"Nombre d'images utilisées : {len(yaw_arr)}")
        print(f"Yaw    : moyenne = {yaw_arr.mean():.2f} deg,  sigma = {yaw_arr.std(ddof=1):.2f} deg")
        print(f"Pitch  : moyenne = {pitch_arr.mean():.2f} deg,  sigma = {pitch_arr.std(ddof=1):.2f} deg")
        print(f"Err. reprojection : moyenne = {err_arr.mean():.3f} px,  max = {err_arr.max():.3f} px")
        print(f"Latence solvePnP+projectPoints : moyenne = {lat_arr.mean():.2f} ms,  max = {lat_arr.max():.2f} ms")
        print("=============================================\n")
    else:
        print("Pas de visage détecté, aucune statistique calculée.")

def run_gaze_calib(cam_index: int, width: int | None, height: int | None, save_dir: str, use_cubic: bool = False) -> None:
    """Calibrage 5x5 : collecte des features via extract_gaze_features,
    régression ridge (features quadratiques), sauvegarde du mapper + taille écran."""

    cap = _open_camera(cam_index, width, height)

    win = "Gaze Calib - cible"
    screen_w, screen_h = _open_canvas(win, fullscreen=True, fallback=(1280, 720))

    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ys = [0.1, 0.3, 0.5, 0.7, 0.9]
    targets = [(x, y) for y in ys for x in xs]

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.75, min_tracking_confidence=0.75
    )

    SETTLE = 0.35
    SAMPLE = 1.00

    F, T = [], []

    try:
        for (tx, ty) in targets:
            img = 255 * np.ones((screen_h, screen_w, 3), dtype=np.uint8)
            cx = int(tx * screen_w)
            cy = int(ty * screen_h)
            cv.circle(img, (cx, cy), 12, (0, 0, 0), -1)
            cv.imshow(win, img)
            cv.waitKey(int(SETTLE * 1000))

            t0 = time.time()
            samples = []
            while time.time() - t0 < SAMPLE:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = face_mesh.process(rgb)
                rgb.flags.writeable = True
                if not res.multi_face_landmarks:
                    continue

                lms = res.multi_face_landmarks[0].landmark
                h_img, w_img = frame.shape[:2]

                base_f, okf = extract_gaze_features(frame, lms, w_img, h_img,
                                                    reject_blink=True, ear_thresh=0.20)
                if not okf:
                    continue

                if use_cubic:
                    feats = _phi_cubic(base_f)
                else:
                    feats = _phi_quadratic(base_f)
                samples.append(feats)

                cv.circle(img, (cx, cy), 14, (0, 255, 0), 2)
                cv.imshow(win, img)
                cv.waitKey(1)

            if len(samples) >= 3:
                F.append(_median_vec(samples))
                T.append([tx, ty])

    finally:
        cap.release()
        cv.destroyWindow(win)

    if len(F) < 5:
        raise SystemExit("Calibrage insuffisant (moins de 5 points valides). Recommencer.")

    mapper = GazeMapper(lam=1e-2)
    mapper.fit(F, T)

    X = np.asarray(F, float)
    Y = np.asarray(T, float)

    def _fit_and_pred(Xtr, Ytr, Xte):
        gm = GazeMapper(lam=1e-2)
        gm.fit(Xtr, np.column_stack([Ytr[:, 0], Ytr[:, 1]]))
        return np.array([gm.predict(x) for x in Xte], float)

    errs = []
    for i in range(len(X)):
        msk = np.ones(len(X), dtype=bool)
        msk[i] = False
        pred_i = _fit_and_pred(X[msk], Y[msk], X[~msk])
        ex = (pred_i[0, 0] - Y[~msk][0, 0]) * screen_w
        ey = (pred_i[0, 1] - Y[~msk][0, 1]) * screen_h
        errs.append((ex**2 + ey**2) ** 0.5)
    if len(errs):
        import numpy as _np
        print(f"[VAL] LOPO RMSE ~ {float(_np.mean(errs)):.1f} px (median {float(_np.median(errs)):.1f} px)")

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "gaze_mapper.json")
    data = mapper.to_dict()
    data["screen_w"] = int(screen_w)
    data["screen_h"] = int(screen_h)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Mapper sauvegardé dans {path}")


def run_gaze_runtime(cam_index: int, width: int | None, height: int | None, save_dir: str, use_cubic: bool = False) -> None:
    """Runtime : charge le mapper, extrait les features via extract_gaze_features,
    prédit (x,y), lisse (OneEuro) et affiche un curseur + fenêtre de debug des pupilles."""

    path = os.path.join(save_dir, "gaze_mapper.json")
    if not os.path.isfile(path):
        raise SystemExit(f"Mapper introuvable: {path}. Lancer d'abord --mode gaze_calib")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapper = GazeMapper.from_dict(data)

    cap = _open_camera(cam_index, width, height)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.75, min_tracking_confidence=0.75
    )

    win = "Gaze Runtime"
    win_debug = "Debug Pupils"
    screen_w, screen_h = _open_canvas(win, fullscreen=True, fallback=(1280, 720))
    cv.namedWindow(win_debug, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_debug, 800, 400)

    scr_meta_w = data.get("screen_w")
    scr_meta_h = data.get("screen_h")
    if scr_meta_w and scr_meta_h and (scr_meta_w != screen_w or scr_meta_h != screen_h):
        print(f"[WARN] Calib sur {scr_meta_w}x{scr_meta_h}, runtime {screen_w}x{screen_h} -> recalibre ou ajuste la fenêtre.")

    smooth = OneEuro2D(min_cutoff=1.2, beta=0.02, dcutoff=1.0, freq=60.0)

    x_px, y_px = screen_w // 2, screen_h // 2
    last, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = face_mesh.process(rgb)
        rgb.flags.writeable = True

        debug_canvas = np.zeros((400, 800, 3), dtype=np.uint8)
        xn, yn = 0.5, 0.5
        base_f = None

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            h_img, w_img = frame.shape[:2]

            base_f, okf = extract_gaze_features(frame, lms, w_img, h_img,
                                                reject_blink=True, ear_thresh=0.20)
            if okf:
                if use_cubic:
                    f = _phi_cubic(base_f)
                else:
                    f = _phi_quadratic(base_f)
                xn, yn = mapper.predict(f)
                xn, yn = smooth.update(xn, yn)
                x_px = int(xn * screen_w)
                y_px = int(yn * screen_h)

            # Debug: Afficher les yeux avec pupilles
            x1L, y1L, x2L, y2L = _eye_roi_from_ids(lms, L_IDS, w_img, h_img)
            x1R, y1R, x2R, y2R = _eye_roi_from_ids(lms, R_IDS, w_img, h_img)

            eye_L = frame[y1L:y2L, x1L:x2L].copy() if y2L > y1L and x2L > x1L else None
            eye_R = frame[y1R:y2R, x1R:x2R].copy() if y2R > y1R and x2R > x1R else None

            # Iris centers
            center_L = get_iris_center(lms, LEFT_IRIS, w_img, h_img)
            center_R = get_iris_center(lms, RIGHT_IRIS, w_img, h_img)

            # Dessiner sur les ROIs des yeux
            if eye_L is not None and center_L:
                cx_rel = center_L[0] - x1L
                cy_rel = center_L[1] - y1L
                cv.circle(eye_L, (cx_rel, cy_rel), 5, (0, 255, 0), 2)
                cv.circle(eye_L, (cx_rel, cy_rel), 2, (0, 0, 255), -1)

                eye_L_resized = cv.resize(eye_L, (380, 180))
                debug_canvas[10:190, 10:390] = eye_L_resized

                if base_f:
                    cv.putText(debug_canvas, f"L: ({base_f[0]:.3f}, {base_f[1]:.3f})",
                              (10, 210), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if eye_R is not None and center_R:
                cx_rel = center_R[0] - x1R
                cy_rel = center_R[1] - y1R
                cv.circle(eye_R, (cx_rel, cy_rel), 5, (0, 255, 0), 2)
                cv.circle(eye_R, (cx_rel, cy_rel), 2, (0, 0, 255), -1)

                eye_R_resized = cv.resize(eye_R, (380, 180))
                debug_canvas[10:190, 410:790] = eye_R_resized

                if base_f:
                    cv.putText(debug_canvas, f"R: ({base_f[2]:.3f}, {base_f[3]:.3f})",
                              (410, 210), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Afficher infos de prédiction
        if base_f:
            cv.putText(debug_canvas, f"Yaw: {base_f[4]:.3f}  Pitch: {base_f[5]:.3f}",
                      (10, 250), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv.putText(debug_canvas, f"Gaze predict: ({xn:.3f}, {yn:.3f})",
                  (10, 290), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.putText(debug_canvas, f"Screen pos: ({x_px}, {y_px})",
                  (10, 330), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 255), 2)
        cv.putText(debug_canvas, f"FPS: {fps:.1f}",
                  (10, 370), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Afficher curseur
        img = 255 * np.ones((screen_h, screen_w, 3), dtype=np.uint8)
        cv.circle(img, (x_px, y_px), 8, (0, 0, 255), -1)
        last, fps = _fps_update(last, fps)
        cv.putText(img, f"FPS: {fps:5.1f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv.LINE_AA)

        cv.imshow(win, img)
        cv.imshow(win_debug, debug_canvas)

        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv.destroyWindow(win)
    cv.destroyWindow(win_debug)


# ═══════════════════════════════════════════════════════════════════════════════
# DÉTECTION D'EXPRESSIONS FACIALES EN TEMPS RÉEL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_smile_ratio(landmarks, w_img, h_img):
    """
    Calcule un ratio de sourire basé sur la position des coins de la bouche.
    Plus le ratio est élevé, plus la personne sourit.

    Retourne une valeur entre 0.0 (pas de sourire) et 1.0+ (grand sourire)
    """
    # Coins de la bouche
    LEFT_MOUTH_CORNER = 61   # Coin gauche
    RIGHT_MOUTH_CORNER = 291  # Coin droit
    # Points du milieu de la bouche (lèvres supérieure et inférieure)
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14

    try:
        # Positions des coins de la bouche
        left_corner = landmarks[LEFT_MOUTH_CORNER]
        right_corner = landmarks[RIGHT_MOUTH_CORNER]

        # Position du centre de la bouche
        mouth_top = landmarks[MOUTH_TOP]
        mouth_bottom = landmarks[MOUTH_BOTTOM]

        # Hauteur moyenne des coins (en y, écran: y petit = haut)
        corners_avg_y = (left_corner.y + right_corner.y) / 2.0

        # Hauteur du centre de la bouche
        center_y = (mouth_top.y + mouth_bottom.y) / 2.0

        # Largeur de la bouche
        mouth_width = abs(right_corner.x - left_corner.x)

        # Différence verticale (si coins sont plus hauts que centre = sourire)
        # center_y - corners_avg_y positif = sourire (coins remontent)
        vertical_diff = center_y - corners_avg_y

        # Normaliser par rapport à la largeur de la bouche
        if mouth_width > 0:
            smile_ratio = vertical_diff / mouth_width
            # Amplifier et clamper
            smile_ratio = max(0.0, min(1.0, smile_ratio * 8.0))
            return smile_ratio
        else:
            return 0.0

    except Exception:
        return 0.0


def compute_eyebrow_position(landmarks, w_img, h_img):
    """
    Calcule la position des sourcils par rapport aux yeux.
    Retourne un ratio positif si sourcils relevés, négatif si sourcils froncés.

    Retourne (eyebrow_ratio, is_furrowed):
    - eyebrow_ratio: float (négatif = froncés, positif = relevés)
    - is_furrowed: bool (True si sourcils froncés)
    """
    # Landmarks des sourcils
    LEFT_EYEBROW = [70, 63, 105, 66, 107]   # Sourcil gauche (milieu)
    RIGHT_EYEBROW = [336, 296, 334, 293, 300]  # Sourcil droit (milieu)

    # Landmarks du haut des yeux (paupière supérieure)
    LEFT_EYE_TOP = [159, 145, 158]
    RIGHT_EYE_TOP = [386, 374, 385]

    try:
        # Position moyenne des sourcils
        left_brow_y = sum(landmarks[i].y for i in LEFT_EYEBROW) / len(LEFT_EYEBROW)
        right_brow_y = sum(landmarks[i].y for i in RIGHT_EYEBROW) / len(RIGHT_EYEBROW)
        avg_brow_y = (left_brow_y + right_brow_y) / 2.0

        # Position moyenne du haut des yeux
        left_eye_top_y = sum(landmarks[i].y for i in LEFT_EYE_TOP) / len(LEFT_EYE_TOP)
        right_eye_top_y = sum(landmarks[i].y for i in RIGHT_EYE_TOP) / len(RIGHT_EYE_TOP)
        avg_eye_top_y = (left_eye_top_y + right_eye_top_y) / 2.0

        # Distance sourcil-œil (en y, rappel: y petit = haut)
        # Si avg_eye_top_y - avg_brow_y est grand = sourcils hauts (relevés)
        # Si petit ou négatif = sourcils bas (froncés)
        brow_eye_distance = avg_eye_top_y - avg_brow_y

        # Normaliser (distance typique ~0.03-0.05)
        # Valeur positive = sourcils normaux/relevés
        # Valeur négative ou très petite = sourcils froncés
        eyebrow_ratio = brow_eye_distance * 20.0  # Amplifier pour avoir des valeurs ~1.0

        # Détecter si sourcils froncés (seuil augmenté pour meilleure détection)
        is_furrowed = brow_eye_distance < 0.035  # Seuil ajusté (était 0.025)

        return eyebrow_ratio, is_furrowed

    except Exception:
        return 0.0, False


def compute_sadness_ratio(landmarks, w_img, h_img):
    """
    Calcule un ratio de tristesse basé sur les coins de bouche vers le bas.
    Inverse du sourire.

    Retourne une valeur entre 0.0 (pas triste) et 1.0+ (très triste)
    """
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14

    try:
        left_corner = landmarks[LEFT_MOUTH_CORNER]
        right_corner = landmarks[RIGHT_MOUTH_CORNER]
        mouth_top = landmarks[MOUTH_TOP]
        mouth_bottom = landmarks[MOUTH_BOTTOM]

        corners_avg_y = (left_corner.y + right_corner.y) / 2.0
        center_y = (mouth_top.y + mouth_bottom.y) / 2.0
        mouth_width = abs(right_corner.x - left_corner.x)

        # Différence verticale inverse du sourire
        # corners_avg_y - center_y positif = tristesse (coins descendent)
        vertical_diff = corners_avg_y - center_y

        if mouth_width > 0:
            sadness_ratio = vertical_diff / mouth_width
            sadness_ratio = max(0.0, min(1.0, sadness_ratio * 8.0))
            return sadness_ratio
        else:
            return 0.0

    except Exception:
        return 0.0


def compute_nose_wrinkle(landmarks, w_img, h_img):
    """
    Détecte le plissement du nez (caractéristique de la colère/dégoût).
    Mesure la distance entre le pont du nez et les côtés du nez.

    Retourne (wrinkle_ratio, is_wrinkled):
    - wrinkle_ratio: float (plus élevé = nez plus plissé)
    - is_wrinkled: bool (True si nez plissé)
    """
    # Landmarks du nez
    NOSE_BRIDGE = 168      # Pont du nez (haut)
    NOSE_TIP = 1           # Pointe du nez
    LEFT_NOSE_SIDE = 120   # Côté gauche du nez
    RIGHT_NOSE_SIDE = 349  # Côté droit du nez

    try:
        nose_bridge = landmarks[NOSE_BRIDGE]
        nose_tip = landmarks[NOSE_TIP]
        left_side = landmarks[LEFT_NOSE_SIDE]
        right_side = landmarks[RIGHT_NOSE_SIDE]

        # Distance verticale pont-pointe (référence)
        nose_length = abs(nose_tip.y - nose_bridge.y)

        if nose_length == 0:
            return 0.0, False

        # Distance horizontale entre les côtés du nez
        nose_width = abs(right_side.x - left_side.x)

        # Ratio largeur/longueur (augmente quand le nez se plisse)
        wrinkle_ratio = nose_width / nose_length

        # Détecter si nez plissé (ratio élevé)
        is_wrinkled = wrinkle_ratio > 1.8  # Seuil empirique

        return wrinkle_ratio, is_wrinkled

    except Exception:
        return 0.0, False


def detect_expression(landmarks, w_img, h_img):
    """
    Détecte l'expression faciale actuelle basée sur les landmarks.

    Retourne (expression_name, confidence) où:
    - expression_name: str ("NEUTRE", "SOURIRE", "CLIN_OEIL_GAUCHE", etc.)
    - confidence: float (0.0 à 1.0)
    """
    try:
        # --- Métriques principales ---
        ear_left = compute_EAR(landmarks, w_img, h_img, side="left")
        ear_right = compute_EAR(landmarks, w_img, h_img, side="right")
        smile_ratio = compute_smile_ratio(landmarks, w_img, h_img)
        sadness_ratio = compute_sadness_ratio(landmarks, w_img, h_img)
        nose_wrinkle_ratio, nose_wrinkled = compute_nose_wrinkle(landmarks, w_img, h_img)
        mouth_open = lips_open_ratio(landmarks, w_img, h_img)

        # --- Sourcils (froncés / relevés) ---
        eyebrow_ratio, is_furrowed = compute_eyebrow_position(landmarks, w_img, h_img)

        # Seuils de détection
        EYE_CLOSED_THRESHOLD = 0.18
        EYE_WIDE_THRESHOLD = 0.35
        SMILE_THRESHOLD = 0.3
        SADNESS_THRESHOLD = 0.25
        MOUTH_OPEN_THRESHOLD = 0.25
        MOUTH_SLIGHTLY_OPEN = 0.15  # Pour étonnement

        left_closed = ear_left < EYE_CLOSED_THRESHOLD
        right_closed = ear_right < EYE_CLOSED_THRESHOLD
        both_eyes_wide = ear_left > EYE_WIDE_THRESHOLD and ear_right > EYE_WIDE_THRESHOLD
        is_smiling = smile_ratio > SMILE_THRESHOLD
        is_sad = sadness_ratio > SADNESS_THRESHOLD
        mouth_is_open = mouth_open > MOUTH_OPEN_THRESHOLD
        mouth_slightly_open = mouth_open > MOUTH_SLIGHTLY_OPEN

        # ================= PRIORITÉS =================

        # 1. Clin d'œil droit (écran inversé)
        if left_closed and not right_closed and not mouth_is_open:
            confidence = 1.0 - ear_left
            return ("CLIN_OEIL_DROIT", min(1.0, confidence))

        # 2. Clin d'œil gauche (écran inversé)
        if right_closed and not left_closed and not mouth_is_open:
            confidence = 1.0 - ear_right
            return ("CLIN_OEIL_GAUCHE", min(1.0, confidence))

        # 3. COLÈRE = nez plissé OU sourcils froncés (et pas de sourire franc)
        #    → plus facile à obtenir que seulement le nez
        if (nose_wrinkled or is_furrowed) and not is_smiling:
            # Combiner les deux signaux pour la confiance
            conf_nose = max(0.0, (nose_wrinkle_ratio - 0.2) / 1.0)   # seuil assoupli (~1.2 au lieu de 1.8)
            conf_brow = 0.3 if is_furrowed else 0.0                  # sourcils froncés = fort indice
            confidence = min(1.0, max(conf_nose, conf_brow))
            return ("COLERE", confidence)

        # 4. Surprise (yeux grand ouverts + bouche très ouverte)
        if both_eyes_wide and mouth_is_open:
            confidence = (ear_left + ear_right) / 2.0 * mouth_open
            return ("SURPRISE", min(1.0, confidence))

        # 6. Sourire
        if is_smiling:
            confidence = smile_ratio
            return ("SOURIRE", min(1.0, confidence))

        # 7. Tristesse
        if is_sad and not is_smiling:
            confidence = sadness_ratio
            return ("TRISTESSE", min(1.0, confidence))

        # 8. Bouche ouverte (sans autre expression forte)
        if mouth_is_open and not both_eyes_wide:
            confidence = mouth_open
            return ("BOUCHE_OUVERTE", min(1.0, confidence))

        # 9. Yeux fermés (fatigue/sommeil)
        if left_closed & right_closed:
            confidence = 1.0 - (ear_left + ear_right) / 2.0
            return ("YEUX_FERMES", min(1.0, confidence))

        # 10. Neutre
        return ("NEUTRE", 1.0)

    except Exception:
        return ("ERREUR", 0.0)


def run_avatar(cam_index: int, width: int | None, height: int | None, save_dir: str) -> None:
    """
    Avatar 3D maillé: tête réaliste pilotée par regard, blink, bouche, yaw/pitch.
    Utilise les vrais landmarks MediaPipe pour un rendu 3D réaliste.

    DÉTECTION D'EXPRESSIONS EN TEMPS RÉEL:
    - L'avatar détecte automatiquement vos expressions faciales
    - Expressions détectées: SOURIRE, CLIN_OEIL_GAUCHE, CLIN_OEIL_DROIT,
      SURPRISE, BOUCHE_OUVERTE, YEUX_FERMES, NEUTRE
    - Appuyez sur 'q' pour quitter
    """

    path = os.path.join(save_dir, "gaze_mapper.json")
    if not os.path.isfile(path):
        print(f"[WARN] Mapper introuvable: {path}. Mode dégradé sans regard calibré.")
        mapper = None
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapper = GazeMapper.from_dict(data)

    cap = _open_camera(cam_index, width, height)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.75, min_tracking_confidence=0.75
    )

    win = "Avatar 3D"
    win_debug = "Debug Camera"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.resizeWindow(win, 960, 720)
    cv.namedWindow(win_debug, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_debug, 800, 600)

    smooth = AvatarSmoother(freq=60.0)
    last, fps = time.time(), 0.0

    xn_s, yn_s = 0.5, 0.5
    yaw_s = pitch_s = 0.0
    eyeL_s = eyeR_s = 1.0
    mouth_s = 0.0

    # ---- Détection d'expressions ----
    current_expression = "NEUTRE"
    expression_confidence = 0.0

    print("\n" + "="*60)
    print("DÉTECTION D'EXPRESSIONS FACIALES EN TEMPS RÉEL")
    print("="*60)
    print("L'avatar va détecter automatiquement vos expressions:")
    print("  - SOURIRE")
    print("  - TRISTESSE")
    print("  - COLÈRE (nez plissé)")
    print("  - CLIN D'OEIL GAUCHE / DROITE")
    print("  - SURPRISE (yeux + bouche très ouverts)")
    print("  - BOUCHE OUVERTE")
    print("  - YEUX FERMÉS")
    print("  - NEUTRE")
    print("\nAppuyez sur 'q' pour quitter")
    print("="*60 + "\n")

    # Stats avatar
    frame_times = []
    fps_values = []

    while True:
        t0 = time.time()  # début traitement frame
        ok, frame = cap.read()
        if not ok:
            break

        h_img, w_img = frame.shape[:2]

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = face_mesh.process(rgb)
        rgb.flags.writeable = True

        lms = None
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark

            if mapper is not None:
                base_f, okf = extract_gaze_features(frame, lms, w_img, h_img,
                                                    reject_blink=True, ear_thresh=0.20)
                if okf:
                    f = _phi_quadratic(base_f)
                    xn, yn = mapper.predict(f)
                else:
                    xn, yn = xn_s, yn_s
            else:
                xn, yn = 0.5, 0.5

            try:
                EAR_L = compute_EAR(lms, w_img, h_img, side="left")
                EAR_R = compute_EAR(lms, w_img, h_img, side="right")
            except Exception:
                EAR_L = EAR_R = 0.25

            openL = _clamp((EAR_L - 0.15) / 0.15)
            openR = _clamp((EAR_R - 0.15) / 0.15)

            mopen = lips_open_ratio(lms, w_img, h_img)

            if mapper is not None and okf:
                yaw, pitch = float(base_f[4]), float(base_f[5])
            else:
                yaw, pitch = yaw_s, pitch_s

            yaw = _clamp(yaw, -0.35, 0.35)
            pitch = _clamp(pitch, -0.35, 0.35)

            # Lisser les valeurs détectées
            xn_s, yn_s, yaw_s, pitch_s, eyeL_s, eyeR_s, mouth_s = smooth.update(
                xn, yn, yaw, pitch, openL, openR, mopen
            )

            # ---- Détecter l'expression faciale actuelle ----
            current_expression, expression_confidence = detect_expression(lms, w_img, h_img)

            # Calculer la position des sourcils pour l'animation
            eyebrow_ratio_val, _ = compute_eyebrow_position(lms, w_img, h_img)
        else:
            # Pas de landmarks détectés, utiliser valeurs par défaut
            eyebrow_ratio_val = 0.0

        canvas = 255 * np.ones((720, 960, 3), dtype=np.uint8)
        draw_avatar(canvas, xn_s, yn_s, yaw_s, pitch_s, eyeL_s, eyeR_s, mouth_s,
                   landmarks=lms, w_img=w_img, h_img=h_img,
                   eyebrow_ratio=eyebrow_ratio_val, expression=current_expression)

        last, fps = _fps_update(last, fps)
        cv.putText(canvas, f"FPS:{fps:4.1f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2, cv.LINE_AA)

        # ---- Afficher l'expression détectée ----
        # Couleur en fonction de l'expression
        expr_colors = {
            "NEUTRE": (150, 150, 150),
            "SOURIRE": (0, 255, 255),        # Jaune (cyan en BGR)
            "CLIN_OEIL_GAUCHE": (255, 0, 255),  # Magenta
            "CLIN_OEIL_DROIT": (255, 0, 255),   # Magenta
            "SURPRISE": (0, 165, 255),       # Orange
            "BOUCHE_OUVERTE": (0, 255, 0),   # Vert
            "YEUX_FERMES": (255, 0, 0),      # Bleu
            "TRISTESSE": (180, 120, 0),      # Bleu foncé
            "COLERE": (0, 0, 200),           # Rouge foncé
            "ERREUR": (0, 0, 255),           # Rouge
        }

        expr_color = expr_colors.get(current_expression, (255, 255, 255))

        # Afficher l'expression avec une barre de confiance
        cv.putText(canvas, f"EXPRESSION: {current_expression}", (10, 70),
                  cv.FONT_HERSHEY_SIMPLEX, 0.9, expr_color, 2, cv.LINE_AA)

        # Barre de confiance
        conf_width = int(expression_confidence * 200)
        cv.rectangle(canvas, (10, 85), (210, 100), (50, 50, 50), -1)  # Fond
        cv.rectangle(canvas, (10, 85), (10 + conf_width, 100), expr_color, -1)  # Barre
        cv.putText(canvas, f"{expression_confidence:.0%}", (220, 98),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, expr_color, 1, cv.LINE_AA)

        # ---- Fenêtre de debug caméra ----
        debug_frame = frame.copy()
        if lms is not None:
            # Dessiner les landmarks du visage
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_face_mesh_module = mp.solutions.face_mesh

            mp_drawing.draw_landmarks(
                image=debug_frame,
                landmark_list=res.multi_face_landmarks[0],
                connections=mp_face_mesh_module.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=debug_frame,
                landmark_list=res.multi_face_landmarks[0],
                connections=mp_face_mesh_module.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        # Afficher l'expression détectée sur le debug
        cv.putText(debug_frame, f"EXPRESSION: {current_expression}", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, expr_color, 2, cv.LINE_AA)
        cv.putText(debug_frame, f"Confidence: {expression_confidence:.0%}", (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow(win, canvas)
        cv.imshow(win_debug, debug_frame)

        # Mise à jour FPS et temps de traitement
        last, fps = _fps_update(last, fps)
        fps_values.append(fps)
        t1 = time.time()
        frame_times.append((t1 - t0) * 1000.0)  # en ms

        # ---- Gestion clavier ----
        key = cv.waitKey(1) & 0xFF

        # Vérifier si c'est 'q' pour quitter
        if key == ord('q'):
            break

    # ===== STATISTIQUES AVATAR =====
    if frame_times:
        ft = np.array(frame_times)
        fv = np.array(fps_values)
        print("\n=== Statistiques avatar (runtime) ===")
        print(f"Frames traitées           : {len(ft)}")
        print(f"Temps de rendu avatar     : mean = {ft.mean():.2f} ms,  std = {ft.std(ddof=1):.2f} ms")
        print(f"Fréquence d'affichage     : mean = {fv.mean():.1f} FPS")
        print(f"Latence chaîne complète   : ~{ft.mean():.2f} ms (capture -> rendu inclus)")
        print("=====================================\n")

    cap.release()
    cv.destroyWindow(win)


def run_test_pupil_positions(cam_index: int, width: int | None, height: int | None, save_dir: str) -> None:
    """
    Test interactif pour capturer les coordonnées moyennes des pupilles
    dans différentes positions de regard (centre, haut, bas, gauche, droite),
    avec beaucoup plus d'échantillons par position (répétitions + durée allongée).
    """
    cap = _open_camera(cam_index, width, height)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9
    )

    # Positions simplifiées : seulement les 5 directions cardinales
    # (pas de diagonales car difficiles à détecter avec précision)
    positions = [
        ("CENTRE", "Regarde droit devant toi"),
        ("HAUT", "Regarde en HAUT"),
        ("BAS", "Regarde en BAS"),
        ("GAUCHE", "Regarde à GAUCHE"),
        ("DROITE", "Regarde à DROITE"),
    ]

    # >>> PARAMÈTRES IMPORTANTS <<<
    n_repeats = 1          # nombre de passes par position
    capture_duration = 4.0 # durée de capture par passe (secondes)
    countdown_start = 2.0  # temps de "prépare-toi" avant chaque passe

    results = {}
    win = "Test Positions Pupilles"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.resizeWindow(win, 1000, 600)

    for position_name, instruction in positions:
        print(f"\n=== Position: {position_name} ===")
        print(f"Instruction: {instruction}")

        # On va accumuler toutes les répétitions ici
        all_samples_L = []
        all_samples_R = []
        all_samples_yaw = []
        all_samples_pitch = []
        all_samples_ear = []

        for rep in range(n_repeats):
            print(f"  > Répétition {rep+1}/{n_repeats}")

            samples_L = []
            samples_R = []
            samples_yaw = []
            samples_pitch = []
            samples_ear = []

            countdown = countdown_start
            start_capture = None

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                h_img, w_img = frame.shape[:2]
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = face_mesh.process(rgb)
                rgb.flags.writeable = True

                display = frame.copy()

                # Phase de préparation ("prépare-toi")
                if countdown > 0:
                    cv.putText(display, f"Prepare-toi: {countdown:.1f}", (50, 100),
                              cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 4, cv.LINE_AA)
                    cv.putText(display, instruction, (50, 200),
                              cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3, cv.LINE_AA)
                    countdown -= 1 / 30.0  # approx 30 FPS

                # Début de la capture
                elif start_capture is None:
                    start_capture = time.time()

                # Phase de capture
                elif time.time() - start_capture < capture_duration:
                    elapsed = time.time() - start_capture
                    remaining = capture_duration - elapsed

                    cv.putText(display, f"CAPTURE EN COURS: {remaining:.1f}s", (50, 100),
                              cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv.LINE_AA)
                    cv.putText(display, instruction, (50, 200),
                              cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3, cv.LINE_AA)

                    if res.multi_face_landmarks:
                        lms = res.multi_face_landmarks[0].landmark

                        x1L, y1L, x2L, y2L = _eye_roi_from_ids(lms, L_IDS, w_img, h_img)
                        x1R, y1R, x2R, y2R = _eye_roi_from_ids(lms, R_IDS, w_img, h_img)

                        center_L = get_iris_center(lms, LEFT_IRIS, w_img, h_img)
                        center_R = get_iris_center(lms, RIGHT_IRIS, w_img, h_img)

                        if center_L and (x2L > x1L) and (y2L > y1L):
                            cxL = (center_L[0] - x1L) / max(1, x2L - x1L)
                            cyL = (center_L[1] - y1L) / max(1, y2L - y1L)
                            samples_L.append((cxL, cyL))
                            cv.circle(display, center_L, 5, (0, 255, 0), 2)

                        if center_R and (x2R > x1R) and (y2R > y1R):
                            cxR = (center_R[0] - x1R) / max(1, x2R - x1R)
                            cyR = (center_R[1] - y1R) / max(1, y2R - y1R)
                            samples_R.append((cxR, cyR))
                            cv.circle(display, center_R, 5, (0, 255, 0), 2)

                        try:
                            yaw, pitch = get_headpose_angles(lms, w_img, h_img)
                            samples_yaw.append(yaw)
                            samples_pitch.append(pitch)
                        except:
                            pass

                        # Capturer l'EAR (Eye Aspect Ratio) pour les paupières
                        try:
                            ear = compute_EAR(lms, w_img, h_img, side="both")
                            samples_ear.append(ear)
                        except:
                            pass

                        cv.rectangle(display, (x1L, y1L), (x2L, y2L), (255, 0, 0), 2)
                        cv.rectangle(display, (x1R, y1R), (x2R, y2R), (255, 0, 0), 2)

                # Fin de la capture pour cette répétition
                else:
                    if len(samples_L) > 0 and len(samples_R) > 0:
                        all_samples_L.extend(samples_L)
                        all_samples_R.extend(samples_R)
                        all_samples_yaw.extend(samples_yaw)
                        all_samples_pitch.extend(samples_pitch)
                        all_samples_ear.extend(samples_ear)
                        print(f"    - Rép {rep+1}: {len(samples_L)} samples")
                    else:
                        print(f"    - Rép {rep+1}: ECHEC (pas assez de données)")
                    break  # sortie du while -> prochaine répétition

                cv.imshow(win, display)
                if (cv.waitKey(1) & 0xFF) == ord('q'):
                    cap.release()
                    cv.destroyWindow(win)
                    return

            time.sleep(0.5)  # petite pause entre répétitions

        # Après toutes les répétitions pour cette position
        if len(all_samples_L) > 0 and len(all_samples_R) > 0:
            avg_L = np.mean(all_samples_L, axis=0)
            avg_R = np.mean(all_samples_R, axis=0)
            avg_yaw = np.mean(all_samples_yaw) if all_samples_yaw else 0.0
            avg_pitch = np.mean(all_samples_pitch) if all_samples_pitch else 0.0
            avg_ear = np.mean(all_samples_ear) if all_samples_ear else 0.25

            results[position_name] = {
                "left": {"x": float(avg_L[0]), "y": float(avg_L[1])},
                "right": {"x": float(avg_R[0]), "y": float(avg_R[1])},
                "yaw": float(avg_yaw),
                "pitch": float(avg_pitch),
                "ear": float(avg_ear),  # Eye Aspect Ratio pour les paupières
                "n_samples": len(all_samples_L)
            }

            print(f"  => TOTAL {position_name}: {len(all_samples_L)} samples")
        else:
            print(f"  ECHEC GLOBAL: Pas assez de données pour {position_name}")

    cap.release()
    cv.destroyWindow(win)

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "pupil_positions_test.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Résultats sauvegardés dans: {output_path}")

    print("\n=== RESUME ===")
    for pos, data in results.items():
        print(f"{pos:15s} | L:({data['left']['x']:.3f},{data['left']['y']:.3f}) | "
              f"R:({data['right']['x']:.3f},{data['right']['y']:.3f}) | "
              f"Yaw:{data['yaw']:+.3f} Pitch:{data['pitch']:+.3f} | "
              f"samples={data['n_samples']}")


def run_gaze_directional(cam_index: int, width: int | None, height: int | None, save_dir: str) -> None:
    """
    Contrôle du curseur par mapping directionnel basé sur pupil_positions_test.json,
    utilisant une machine à états finie (FSM) pour gérer les transitions de regard.

    Nouveautés avec la FSM :
    - États formellement définis via l'enum GazeState
    - Transitions avec seuils adaptatifs et hystérésis
    - Filtrage temporel (plusieurs frames consécutives pour confirmer un changement)
    - Historique des états pour analyse et débogage
    - Calcul de scores de confiance pour chaque transition

    Principe :
    - On charge les positions moyennes du JSON
    - On initialise une GazeStateMachine avec les données de calibration
    - La machine construit automatiquement la matrice de transitions
    - À chaque frame, on met à jour la machine avec les features courantes
    - La FSM décide intelligemment des transitions d'états
    """

    # ---- Seuils de détection basés sur mouvement relatif des pupilles ----
    # NOTE: La détection verticale est plus difficile que l'horizontale car :
    # - Les mouvements verticaux de la pupille sont physiologiquement plus petits
    # - Les paupières limitent le mouvement vertical de l'œil
    # - Les variations y sont 3-4× plus faibles que les variations x
    # On amplifie donc les mouvements verticaux et on combine avec l'EAR des paupières

    print("[Détection] Utilisation de mouvements relatifs (9 directions: CENTRE + 4 cardinales + 4 diagonales)")
    print("[Geste] Fermer un œil → Curseur devient GRIS et se FIGE")

    # ═══════════════════════════════════════════════════════════════════════════════
    # EXPLICATION : Pourquoi 9 positions marchent avec calibration sur 5 positions ?
    # ═══════════════════════════════════════════════════════════════════════════════
    #
    # ✅ 9 POSITIONS (5 calibrées + 4 diagonales inférées) - FONCTIONNE :
    #
    # 1. CALIBRATION sur 5 positions cardinales :
    #    - CENTRE : définit le point de référence (centre_left_x, centre_left_y)
    #    - HAUT, BAS, GAUCHE, DROITE : définissent les extrêmes de mouvement
    #
    # 2. DÉTECTION à l'exécution :
    #    - On calcule dx = left_x - centre_left_x et dy = left_y - centre_left_y
    #    - Si dx > seuil → GAUCHE, si dx < -seuil → DROITE, sinon → CENTRE (horizontal)
    #    - Si dy > seuil → HAUT, si dy < -seuil → BAS, sinon → CENTRE (vertical)
    #
    # 3. LES DIAGONALES ÉMERGENT NATURELLEMENT :
    #    - Si HAUT + GAUCHE détectés → HAUT_GAUCHE (combinaison automatique)
    #    - Si HAUT + DROITE détectés → HAUT_DROITE
    #    - Si BAS + GAUCHE détectés → BAS_GAUCHE
    #    - Si BAS + DROITE détectés → BAS_DROITE
    #    → Pas besoin de calibrer les diagonales explicitement !
    #
    # 4. ANALOGIE avec un joystick :
    #    - On calibre le centre et les 4 directions cardinales
    #    - Les diagonales sont détectées en combinant horizontal + vertical
    #    - C'est exactement comme un joystick physique
    #
    # ═══════════════════════════════════════════════════════════════════════════════
    # EXPLICATION : Pourquoi 11 positions (avec centres intermédiaires) NE MARCHENT PAS ?
    # ═══════════════════════════════════════════════════════════════════════════════
    #
    # ❌ 11 POSITIONS (ajout de CENTRE_GAUCHE, CENTRE_DROITE, etc.) - NE FONCTIONNE PAS :
    #
    # 1. PROBLÈME DE ZONES AMBIGUËS :
    #    - Avec 11 positions, on essaie de diviser l'axe horizontal en 5 zones :
    #      DROITE | CENTRE_DROITE | CENTRE | CENTRE_GAUCHE | GAUCHE
    #    - Il faudrait 4 seuils différents pour séparer ces 5 zones
    #    - Sans calibration explicite de CENTRE_GAUCHE/DROITE, on ne sait pas où placer ces seuils
    #
    # 2. MANQUE DE CALIBRATION DES POSITIONS INTERMÉDIAIRES :
    #    - Exemple avec les données réelles (calib/pupil_positions_test.json) :
    #      * DROITE : left.x = 0.367
    #      * CENTRE : left.x = 0.476
    #      * GAUCHE : left.x = 0.574
    #    - Où devrait être CENTRE_GAUCHE ? À 0.525 (milieu entre CENTRE et GAUCHE) ?
    #    - PROBLÈME : Sans mesurer explicitement cette position, on devine !
    #    - Chaque personne a une amplitude oculaire différente
    #
    # 3. INSTABILITÉ DE DÉTECTION :
    #    - Les seuils deviennent trop serrés (écart de ~0.05 entre zones)
    #    - Les micro-tremblements oculaires naturels font osciller entre zones voisines
    #    - Le curseur devient nerveux et imprévisible
    #    - Exemple : regard au CENTRE oscille entre CENTRE et CENTRE_GAUCHE
    #
    # 4. PRÉCISION INSUFFISANTE DES YEUX :
    #    - Les mouvements oculaires ne sont pas assez précis pour maintenir
    #      une position stable dans une zone intermédiaire sans entraînement
    #    - Les yeux ont tendance à aller vers des positions "franches" (extrêmes)
    #    - Les positions intermédiaires sont des zones de transition, pas des cibles
    #
    # 5. SOLUTION : Garder uniquement les positions calibrées + diagonales inférées
    #    → 5 positions calibrées (CENTRE + 4 cardinales) suffisent
    #    → 4 diagonales émergent automatiquement par combinaison
    #    → Total : 9 positions détectées de manière robuste
    #
    # ═══════════════════════════════════════════════════════════════════════════════

    DELTA_X_SEUIL = 0.04   # Déplacement horizontal pour détecter GAUCHE/DROITE
    DELTA_Y_SEUIL = 0.005  # Déplacement vertical pour détecter HAUT/BAS (très réduit car mouvements verticaux faibles)
    AMPLIF_Y = 2.0         # Facteur d'amplification pour les mouvements verticaux

    # Paramètres de détection de fermeture d'œil
    EYE_CLOSED_THRESHOLD = 0.18  # Seuil EAR pour considérer qu'un œil est fermé

    # ---- Mapping état → position écran ----
    direction_map = {
        GazeState.CENTRE: (0.5, 0.5),
        GazeState.HAUT: (0.5, 0.1),
        GazeState.BAS: (0.5, 0.9),
        GazeState.GAUCHE: (0.1, 0.5),
        GazeState.DROITE: (0.9, 0.5),
        GazeState.HAUT_GAUCHE: (0.1, 0.1),
        GazeState.HAUT_DROITE: (0.9, 0.1),
        GazeState.BAS_GAUCHE: (0.1, 0.9),
        GazeState.BAS_DROITE: (0.9, 0.9),
    }

    cap = _open_camera(cam_index, width, height)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.75, min_tracking_confidence=0.75
    )

    win = "Gaze Directional"
    win_debug = "Debug Directional (Relative Movement)"
    screen_w, screen_h = _open_canvas(win, fullscreen=False, fallback=(1280, 720))
    cv.namedWindow(win_debug, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_debug, 800, 450)

    smooth = OneEuro2D(min_cutoff=1.2, beta=0.02, dcutoff=1.0, freq=60.0)

    x_px, y_px = screen_w // 2, screen_h // 2
    last, fps = time.time(), 0.0

    # Variable pour la couleur du curseur
    cursor_color = (0, 0, 255)  # Rouge par défaut (BGR)

    # ---- Calibration de la position de départ (CENTRE) ----
    print("\n=== Calibration de la position de départ ===")
    print("Regarde droit devant au CENTRE de l'écran...")
    print("Calibration dans 3 secondes...")

    calibration_start = None
    calibration_duration = 2.0
    center_samples = []
    center_left_x, center_left_y, center_ear = 0.0, 0.0, 0.25
    is_calibrated = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h_img, w_img = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = face_mesh.process(rgb)
        rgb.flags.writeable = True

        debug_canvas = np.zeros((450, 800, 3), dtype=np.uint8)

        # === Phase 1 : Calibration du CENTRE ===
        if not is_calibrated:
            if calibration_start is None:
                calibration_start = time.time()

            elapsed = time.time() - calibration_start
            if elapsed < calibration_duration:
                remaining = calibration_duration - elapsed

                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    base_f, okf = extract_gaze_features(frame, lms, w_img, h_img,
                                                        reject_blink=True, ear_thresh=0.20)
                    if okf:
                        # Capturer left_x, left_y et ear
                        ear_val = base_f[6] if len(base_f) > 6 else 0.25
                        center_samples.append((base_f[0], base_f[1], ear_val))

                cv.putText(debug_canvas, f"CALIBRATION: {remaining:.1f}s", (50, 100),
                          cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv.LINE_AA)
                cv.putText(debug_canvas, "Regarde au CENTRE !", (50, 200),
                          cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2, cv.LINE_AA)

            else:
                if len(center_samples) > 0:
                    # Calculer la moyenne des positions et de l'EAR
                    center_left_x = sum(s[0] for s in center_samples) / len(center_samples)
                    center_left_y = sum(s[1] for s in center_samples) / len(center_samples)
                    center_ear = sum(s[2] for s in center_samples) / len(center_samples)
                    is_calibrated = True
                    print(f"✅ Calibration terminée: centre = ({center_left_x:.3f}, {center_left_y:.3f}), EAR = {center_ear:.3f}")
                else:
                    print("❌ Échec calibration, pas assez de données")
                    break

        # === Phase 2 : Détection avec mouvements relatifs ===
        else:
            current_state = GazeState.CENTRE
            xn, yn = direction_map.get(current_state, (0.5, 0.5))
            left_x, left_y, ear = center_left_x, center_left_y, center_ear  # Valeurs par défaut = centre
            dx, dy, ear_delta = 0.0, 0.0, 0.0  # Deltas par défaut

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                base_f, okf = extract_gaze_features(frame, lms, w_img, h_img,
                                                    reject_blink=True, ear_thresh=0.20)
                if okf:
                    # Extraire left_x, left_y et ear des features [cxL, cyL, cxR, cyR, yaw, pitch, ear]
                    left_x = base_f[0]
                    left_y = base_f[1]
                    ear = base_f[6] if len(base_f) > 6 else 0.25

                    # === Détection de fermeture d'un œil ===
                    # Calculer l'EAR de chaque œil séparément
                    one_eye_closed = False
                    try:
                        ear_left = compute_EAR(lms, w_img, h_img, side="left")
                        ear_right = compute_EAR(lms, w_img, h_img, side="right")

                        # Vérifier si un seul œil est fermé
                        left_closed = ear_left < EYE_CLOSED_THRESHOLD
                        right_closed = ear_right < EYE_CLOSED_THRESHOLD

                        # Si un seul œil fermé (mais pas les deux) → curseur gris et figé
                        if (left_closed and not right_closed) or (right_closed and not left_closed):
                            one_eye_closed = True
                            cursor_color = (128, 128, 128)  # Gris (BGR)
                        else:
                            cursor_color = (0, 0, 255)  # Rouge normal (BGR)
                    except:
                        cursor_color = (0, 0, 255)  # Rouge par défaut en cas d'erreur

                    # Calculer les deltas par rapport à la position de départ
                    dx = left_x - center_left_x
                    dy = (left_y - center_left_y) * AMPLIF_Y  # Amplifier les mouvements verticaux

                    # Utiliser l'EAR pour améliorer la détection verticale
                    # EAR augmente quand on regarde en HAUT, diminue en BAS
                    ear_delta = (ear - center_ear) * 2.0  # Amplifier les variations d'EAR

                    # Détection par mouvements relatifs
                    # Composante horizontale (dx positif = pupille vers la droite = regarde à GAUCHE)
                    if dx > DELTA_X_SEUIL:
                        horizontal = "GAUCHE"
                    elif dx < -DELTA_X_SEUIL:
                        horizontal = "DROITE"
                    else:
                        horizontal = "CENTRE"

                    # Composante verticale : combiner dy (position pupille) et ear_delta (ouverture paupières)
                    # dy positif = pupille vers le bas = regarde en HAUT
                    # ear_delta positif = yeux plus ouverts = regarde en HAUT
                    vertical_score = dy + ear_delta  # Combiner les deux signaux

                    if vertical_score > DELTA_Y_SEUIL:
                        vertical = "HAUT"
                    elif vertical_score < -DELTA_Y_SEUIL:
                        vertical = "BAS"
                    else:
                        vertical = "CENTRE"

                    # Combiner pour obtenir la direction finale (9 directions)
                    if vertical == "CENTRE" and horizontal == "CENTRE":
                        current_state = GazeState.CENTRE
                    elif vertical == "HAUT" and horizontal == "CENTRE":
                        current_state = GazeState.HAUT
                    elif vertical == "BAS" and horizontal == "CENTRE":
                        current_state = GazeState.BAS
                    elif vertical == "CENTRE" and horizontal == "GAUCHE":
                        current_state = GazeState.GAUCHE
                    elif vertical == "CENTRE" and horizontal == "DROITE":
                        current_state = GazeState.DROITE
                    elif vertical == "HAUT" and horizontal == "GAUCHE":
                        current_state = GazeState.HAUT_GAUCHE
                    elif vertical == "HAUT" and horizontal == "DROITE":
                        current_state = GazeState.HAUT_DROITE
                    elif vertical == "BAS" and horizontal == "GAUCHE":
                        current_state = GazeState.BAS_GAUCHE
                    elif vertical == "BAS" and horizontal == "DROITE":
                        current_state = GazeState.BAS_DROITE
                    else:
                        current_state = GazeState.CENTRE

                    # Récupérer la position écran (seulement si aucun œil n'est fermé)
                    if not one_eye_closed:
                        xn, yn = direction_map.get(current_state, (0.5, 0.5))

                        # Lissage OneEuro
                        xn, yn = smooth.update(xn, yn)
                        x_px = int(xn * screen_w)
                        y_px = int(yn * screen_h)
                    # Sinon, x_px et y_px gardent leur dernière valeur (curseur figé)

                    # Debug : yeux
                    x1L, y1L, x2L, y2L = _eye_roi_from_ids(lms, L_IDS, w_img, h_img)
                    x1R, y1R, x2R, y2R = _eye_roi_from_ids(lms, R_IDS, w_img, h_img)

                    eye_L = frame[y1L:y2L, x1L:x2L].copy() if y2L > y1L and x2L > x1L else None
                    eye_R = frame[y1R:y2R, x1R:x2R].copy() if y2R > y1R and x2R > x1R else None

                    center_L = get_iris_center(lms, LEFT_IRIS, w_img, h_img)
                    center_R = get_iris_center(lms, RIGHT_IRIS, w_img, h_img)

                    if eye_L is not None and center_L:
                        cx_rel = center_L[0] - x1L
                        cy_rel = center_L[1] - y1L
                        cv.circle(eye_L, (cx_rel, cy_rel), 5, (0, 255, 0), 2)
                        eye_L_resized = cv.resize(eye_L, (380, 180))
                        debug_canvas[10:190, 10:390] = eye_L_resized

                    if eye_R is not None and center_R:
                        cx_rel = center_R[0] - x1R
                        cy_rel = center_R[1] - y1R
                        cv.circle(eye_R, (cx_rel, cy_rel), 5, (0, 255, 0), 2)
                        eye_R_resized = cv.resize(eye_R, (380, 180))
                        debug_canvas[10:190, 410:790] = eye_R_resized

            # Infos debug texte avec mouvements relatifs et EAR
            if cursor_color == (128, 128, 128):
                cursor_status = "GRIS (FIGÉ)"
                status_color = (100, 150, 255)
            else:
                cursor_status = "ROUGE (ACTIF)"
                status_color = (0, 255, 255)

            # Afficher l'état (gérer à la fois GazeState et strings)
            state_str = current_state if isinstance(current_state, str) else current_state.to_string()
            cv.putText(debug_canvas, f"State: {state_str} [Curseur: {cursor_status}]", (10, 230),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv.putText(debug_canvas, f"Pupil: ({left_x:.3f}, {left_y:.3f}) | Delta: ({dx:.3f}, {dy:.3f})",
                      (10, 260), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv.putText(debug_canvas, f"EAR: {ear:.3f} | EAR_delta: {ear_delta:.3f}",
                      (10, 290), cv.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 150), 2)
            cv.putText(debug_canvas, f"Cursor: ({xn:.3f}, {yn:.3f})",
                      (10, 320), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 255), 2)
            cv.putText(debug_canvas, f"FPS: {fps:.1f}", (10, 350),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Affichage curseur plein écran
        img = 255 * np.ones((screen_h, screen_w, 3), dtype=np.uint8)
        cv.circle(img, (x_px, y_px), 8, cursor_color, -1)
        last, fps = _fps_update(last, fps)
        cv.putText(img, f"FPS: {fps:5.1f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv.LINE_AA)

        cv.imshow(win, img)
        cv.imshow(win_debug, debug_canvas)

        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv.destroyWindow(win)
    cv.destroyWindow(win_debug)

def run_pupil_eval(cam_index: int, width: int | None, height: int | None, save_dir: str = ".") -> None:
    """
    Mode d'évaluation de la détection pupillaire :
    - mesure FPS de détection
    - temps moyen de post-traitement (ROI + features)
    - variabilité intra-ROI du centre de pupille
    - jitter avant/après filtrage One-Euro
    """
    if mp is None:
        raise SystemExit("MediaPipe n'est pas installé. Exécuter : pip install mediapipe")

    cap = _open_camera(cam_index, width, height)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )

    one_euro = OneEuro2D(min_cutoff=1.2, beta=0.02, dcutoff=1.0, freq=60.0)

    last, fps = time.time(), 0.0

    # --- Stats à accumuler ---
    t_start = time.perf_counter()
    n_frames_total = 0
    n_frames_valid = 0
    proc_times_ms: list[float] = []
    raw_centers: list[list[float]] = []   # [cx, cy] normalisés (moyenne des 2 yeux)
    filt_centers: list[list[float]] = []

    print("\n[INFO] Mode pupil_eval : garde la tête globalement immobile et regarde le centre.")
    print("      Appuie sur 'q' quand tu as quelques centaines de frames.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frames_total += 1

        h_img, w_img = frame.shape[:2]

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = face_mesh.process(rgb)
        rgb.flags.writeable = True

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark

            # --- Mesurer temps post-traitement ROI + features ---
            t0 = time.perf_counter()
            base_f, okf = extract_gaze_features(
                frame, lms, w_img, h_img,
                reject_blink=True, ear_thresh=0.20
            )
            t1 = time.perf_counter()

            if okf:
                proc_times_ms.append((t1 - t0) * 1000.0)

                # base_f = [cxLn, cyLn, cxRn, cyRn, yaw, pitch, ear]
                cxL, cyL, cxR, cyR = base_f[0], base_f[1], base_f[2], base_f[3]

                # centre moyen des 2 yeux dans la/les ROI (coord. normalisées)
                cx = 0.5 * (cxL + cxR)
                cy = 0.5 * (cyL + cyR)

                n_frames_valid += 1
                raw_centers.append([cx, cy])

                fx, fy = one_euro.update(cx, cy)
                filt_centers.append([fx, fy])

                # Visu rapide
                cv.putText(frame, f"Pupil norm: ({cx:.3f}, {cy:.3f})",
                           (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        last, fps = _fps_update(last, fps)
        _put_fps(frame, fps)

        cv.imshow("Pupil Eval (press q to quit)", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    face_mesh.close()

    # --- Calcul des métriques ---
    if n_frames_valid < 2:
        print("[WARN] Trop peu de frames valides pour calculer des statistiques.")
        return

    t_total = time.perf_counter() - t_start
    raw = np.asarray(raw_centers, dtype=np.float64)
    filt = np.asarray(filt_centers, dtype=np.float64)
    proc = np.asarray(proc_times_ms, dtype=np.float64)

    # Fréquences
    fps_capture = n_frames_total / t_total
    fps_detection = n_frames_valid / t_total

    # Temps moyen de post-traitement
    proc_mean = float(proc.mean())
    proc_max = float(proc.max())

    # Variabilité intra-ROI : RMS distance au centre moyen
    mean_raw = raw.mean(axis=0)
    diffs_from_mean = raw - mean_raw
    rms_intra_roi = float(np.sqrt((diffs_from_mean ** 2).sum(axis=1).mean()))

    # Jitter = RMS des déplacements inter-frame
    raw_diffs = np.diff(raw, axis=0)
    filt_diffs = np.diff(filt, axis=0)

    jitter_raw = float(np.sqrt((raw_diffs ** 2).sum(axis=1).mean()))
    jitter_filt = float(np.sqrt((filt_diffs ** 2).sum(axis=1).mean()))
    noise_reduction = (jitter_raw / jitter_filt) if jitter_filt > 1e-6 else float("inf")

    print("\n=== Statistiques détection oculaire (tête immobile) ===")
    print(f"Frames totales (capture)         : {n_frames_total}")
    print(f"Frames valides (features)        : {n_frames_valid}")
    print(f"Fréquence capture globale        : {fps_capture:.1f} FPS")
    print(f"Fréquence de détection (utile)   : {fps_detection:.1f} FPS")
    print(f"Post-traitement ROI+features     : moyenne = {proc_mean:.2f} ms, max = {proc_max:.2f} ms")
    print(f"Variabilité intra-ROI (normée)   : RMS = {rms_intra_roi:.4f}")
    print(f"Jitter pupille brut (normé)      : {jitter_raw:.4f}")
    print(f"Jitter pupille filtré One-Euro   : {jitter_filt:.4f}")
    print(f"Réduction de bruit (jitter_raw/jitter_filt) : facteur {noise_reduction:.2f}")
    print("=======================================================\n")

def _predict_direction_from_features(base_f, lms, w_img, h_img,
                                     center_left_x, center_left_y, center_ear,
                                     smooth, DELTA_X_SEUIL, DELTA_Y_SEUIL,
                                     AMPLIF_Y, EYE_CLOSED_THRESHOLD):
    """
    Renvoie (state, xn, yn, dx, dy, ear, ear_delta, one_eye_closed)
    en réutilisant exactement la logique de run_gaze_directional.
    """
    # Par défaut : CENTRE
    current_state = GazeState.CENTRE
    xn, yn = 0.5, 0.5
    dx = dy = ear_delta = 0.0
    one_eye_closed = False

    if base_f is None:
        return current_state, xn, yn, dx, dy, 0.25, 0.0, False

    # base_f = [cxLn, cyLn, cxRn, cyRn, yaw, pitch, ear]
    left_x = base_f[0]
    left_y = base_f[1]
    ear = base_f[6] if len(base_f) > 6 else 0.25

    # EAR par oeil pour détecter clignement asymétrique
    try:
        ear_left = compute_EAR(lms, w_img, h_img, side="left")
        ear_right = compute_EAR(lms, w_img, h_img, side="right")
        left_closed = ear_left < EYE_CLOSED_THRESHOLD
        right_closed = ear_right < EYE_CLOSED_THRESHOLD
        if (left_closed and not right_closed) or (right_closed and not left_closed):
            one_eye_closed = True
    except Exception:
        pass

    # Deltas relatifs par rapport au centre calibré
    dx = left_x - center_left_x
    dy = (left_y - center_left_y) * AMPLIF_Y
    ear_delta = (ear - center_ear) * 2.0

    # Composante horizontale
    if dx > DELTA_X_SEUIL:
        horizontal = "GAUCHE"
    elif dx < -DELTA_X_SEUIL:
        horizontal = "DROITE"
    else:
        horizontal = "CENTRE"

    # Composante verticale : combinaison dy + EAR
    vertical_score = dy + ear_delta
    if vertical_score > DELTA_Y_SEUIL:
        vertical = "HAUT"
    elif vertical_score < -DELTA_Y_SEUIL:
        vertical = "BAS"
    else:
        vertical = "CENTRE"

    # Combiner horizontal + vertical → 9 états
    if vertical == "CENTRE" and horizontal == "CENTRE":
        current_state = GazeState.CENTRE
    elif vertical == "HAUT" and horizontal == "CENTRE":
        current_state = GazeState.HAUT
    elif vertical == "BAS" and horizontal == "CENTRE":
        current_state = GazeState.BAS
    elif vertical == "CENTRE" and horizontal == "GAUCHE":
        current_state = GazeState.GAUCHE
    elif vertical == "CENTRE" and horizontal == "DROITE":
        current_state = GazeState.DROITE
    elif vertical == "HAUT" and horizontal == "GAUCHE":
        current_state = GazeState.HAUT_GAUCHE
    elif vertical == "HAUT" and horizontal == "DROITE":
        current_state = GazeState.HAUT_DROITE
    elif vertical == "BAS" and horizontal == "GAUCHE":
        current_state = GazeState.BAS_GAUCHE
    elif vertical == "BAS" and horizontal == "DROITE":
        current_state = GazeState.BAS_DROITE
    else:
        current_state = GazeState.CENTRE

    # Mapping état → coordonnée écran (comme dans run_gaze_directional)
    direction_map = {
        GazeState.CENTRE: (0.5, 0.5),
        GazeState.HAUT: (0.5, 0.1),
        GazeState.BAS: (0.5, 0.9),
        GazeState.GAUCHE: (0.1, 0.5),
        GazeState.DROITE: (0.9, 0.5),
        GazeState.HAUT_GAUCHE: (0.1, 0.1),
        GazeState.HAUT_DROITE: (0.9, 0.1),
        GazeState.BAS_GAUCHE: (0.1, 0.9),
        GazeState.BAS_DROITE: (0.9, 0.9),
    }

    xn, yn = direction_map.get(current_state, (0.5, 0.5))
    xn, yn = smooth.update(xn, yn)

    return current_state, xn, yn, dx, dy, ear, ear_delta, one_eye_closed


def parse_args():
    """Parser la ligne de commande et retourner l'espace de noms des options (Namespace)."""
    ap = argparse.ArgumentParser(description="Capturer webcam, afficher FPS, tracer maillage facial et calibrer la caméra")
    ap.add_argument("--mode",
    choices=["live", "facemesh", "calib_collect", "calibrate", "undistort",
             "headpose", "pupil", "pupil_eval", "blink_click",
             "gaze_calib", "gaze_runtime",
             "avatar", "test_positions", "gaze_directional"],
    required=True, help="mode d'exécution")
    ap.add_argument("--camera", type=int, default=0, help="index de la caméra (0 par défaut)")
    ap.add_argument("--contours", action="store_true",
                    help="afficher aussi les contours (sourcils, mâchoire, hairline)")
    ap.add_argument("--eyes", action="store_true",
                    help="dessiner des rectangles ROI autour des yeux")
    ap.add_argument("--nx", type=int, default=9, help="nb de coins intérieurs horizontaux (ex: 9)")
    ap.add_argument("--ny", type=int, default=6, help="nb de coins intérieurs verticaux (ex: 6)")
    ap.add_argument("--square", type=float, default=0.025, help="taille d'une case en mètres (ex: 0.025)")
    ap.add_argument("--save_dir", type=str, default="calib", help="dossier des images et sortie intrinsics.json")
    ap.add_argument("--prefix", type=str, default="img_", help="préfixe des fichiers images (calib_collect)")
    ap.add_argument("--max", type=int, default=30, help="nombre maximum d'images à collecter (calib_collect)")
    ap.add_argument("--width", type=int, default=None, help="largeur (px), ex: 640")
    ap.add_argument("--height", type=int, default=None, help="hauteur (px), ex: 360")
    ap.add_argument("--cubic", action="store_true",
        help="utiliser expansion cubique (au lieu de quadratique) pour le regard")
    
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "live":
        run_live(args.camera, args.width, args.height)

    elif args.mode == "facemesh":
        from pathlib import Path
        log_dir = Path("yeux")
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile_path = log_dir / "cd_yeux.log"

        run_facemesh(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
            show_contours=args.contours,
            show_eyes=args.eyes,
            save_dir=args.save_dir,
        )
        print(f"Log yeux sauvegardé dans : {logfile_path}")

    elif args.mode == "calib_collect":
        run_calib_collect(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
            nx=args.nx,
            ny=args.ny,
            square_m=args.square,
            save_dir=args.save_dir,
            prefix=args.prefix,
            max_images=args.max,
        )

    elif args.mode == "calibrate":
        run_calibrate(
            nx=args.nx,
            ny=args.ny,
            square_m=args.square,
            save_dir=args.save_dir,
        )

    elif args.mode == "undistort":
        run_undistort(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
            save_dir=args.save_dir,
        )

    elif args.mode == "headpose":
        run_headpose(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
            save_dir=args.save_dir,
        )

    elif args.mode == "pupil":
        run_pupil(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
            save_dir=args.save_dir,
        )

    elif args.mode == "blink_click":
        run_blink_click(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
        )

    elif args.mode == "gaze_calib":
        run_gaze_calib(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
            save_dir=args.save_dir,
            use_cubic=args.cubic,
        )

    elif args.mode == "gaze_runtime":
        run_gaze_runtime(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
            save_dir=args.save_dir,
            use_cubic=args.cubic,
        )

    elif args.mode == "avatar":
        run_avatar(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
            save_dir=args.save_dir,
        )

    elif args.mode == "test_positions":
        run_test_pupil_positions(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
            save_dir=args.save_dir,
        )

    elif args.mode == "gaze_directional":
        run_gaze_directional(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
            save_dir=args.save_dir,
        )

    elif args.mode == "pupil_eval":
        run_pupil_eval(
            cam_index=args.camera,
            width=args.width,
            height=args.height,
            save_dir=args.save_dir,
        )
