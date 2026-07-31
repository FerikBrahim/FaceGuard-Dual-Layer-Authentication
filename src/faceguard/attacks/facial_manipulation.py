from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from faceguard.io import ensure_rgb_uint8


@dataclass
class AttackImage:
    name: str
    category: str
    image: np.ndarray
    description: str


def _face_box(image: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(ensure_rgb_uint8(image), cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, 1.08, 5, minSize=(80, 80))
    if len(faces):
        x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        return int(x), int(y), int(width), int(height)
    height, width = gray.shape
    face_width, face_height = int(0.52 * width), int(0.68 * height)
    return (width - face_width) // 2, max(0, int(0.12 * height)), face_width, face_height


def _region(box, left, top, right, bottom, shape):
    x, y, width, height = box
    x1, y1 = x + int(left * width), y + int(top * height)
    x2, y2 = x + int(right * width), y + int(bottom * height)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(shape[1], x2), min(shape[0], y2)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    for channel in range(3):
        source_mean, source_std = cv2.meanStdDev(source_lab[:, :, channel])
        target_mean, target_std = cv2.meanStdDev(target_lab[:, :, channel])
        source_lab[:, :, channel] = (
            (source_lab[:, :, channel] - float(source_mean[0, 0]))
            * (max(float(target_std[0, 0]), 1e-6) / max(float(source_std[0, 0]), 1e-6))
            + float(target_mean[0, 0])
        )
    return cv2.cvtColor(np.clip(source_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def controlled_face_swap(target_rgb: np.ndarray, donor_rgb: np.ndarray) -> np.ndarray:
    target, donor = ensure_rgb_uint8(target_rgb).copy(), ensure_rgb_uint8(donor_rgb)
    tx, ty, tw, th = _face_box(target)
    dx, dy, dw, dh = _face_box(donor)
    donor_face = cv2.resize(donor[dy : dy + dh, dx : dx + dw], (tw, th), interpolation=cv2.INTER_CUBIC)
    target_face = target[ty : ty + th, tx : tx + tw]
    donor_face = _color_transfer(donor_face, target_face)
    mask = np.zeros((th, tw), dtype=np.uint8)
    cv2.ellipse(mask, (tw // 2, th // 2), (int(0.46 * tw), int(0.48 * th)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    try:
        result = cv2.seamlessClone(
            cv2.cvtColor(donor_face, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(target, cv2.COLOR_RGB2BGR),
            mask,
            (tx + tw // 2, ty + th // 2),
            cv2.NORMAL_CLONE,
        )
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    except cv2.error:
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        target[ty : ty + th, tx : tx + tw] = (alpha * donor_face + (1.0 - alpha) * target_face).astype(np.uint8)
        return target


def mouth_blur(image_rgb: np.ndarray) -> np.ndarray:
    image = ensure_rgb_uint8(image_rgb).copy()
    x, y, width, height = _region(_face_box(image), 0.24, 0.62, 0.76, 0.84, image.shape)
    roi = image[y : y + height, x : x + width]
    kernel = max(7, (min(width, height) // 3) | 1)
    image[y : y + height, x : x + width] = cv2.GaussianBlur(roi, (kernel, kernel), 0)
    return image


def eyes_occlusion(image_rgb: np.ndarray) -> np.ndarray:
    image = ensure_rgb_uint8(image_rgb).copy()
    x, y, width, height = _region(_face_box(image), 0.12, 0.24, 0.88, 0.45, image.shape)
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (20, 20, 20), -1)
    return cv2.addWeighted(overlay, 0.82, image, 0.18, 0)


def cheek_alteration(image_rgb: np.ndarray) -> np.ndarray:
    image = ensure_rgb_uint8(image_rgb).copy()
    box = _face_box(image)
    sx, sy, sw, sh = _region(box, 0.12, 0.46, 0.40, 0.70, image.shape)
    dx, dy, dw, dh = _region(box, 0.60, 0.46, 0.88, 0.70, image.shape)
    patch = cv2.resize(image[sy : sy + sh, sx : sx + sw], (dw, dh), interpolation=cv2.INTER_CUBIC)
    mask = np.zeros((dh, dw), dtype=np.uint8)
    cv2.ellipse(mask, (dw // 2, dh // 2), (max(1, dw // 2 - 1), max(1, dh // 2 - 1)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    destination = image[dy : dy + dh, dx : dx + dw]
    image[dy : dy + dh, dx : dx + dw] = (alpha * patch + (1.0 - alpha) * destination).astype(np.uint8)
    return image


def controlled_synthetic_manipulation(image_rgb: np.ndarray) -> np.ndarray:
    """Controlled smoothing/color/warp attack; it is not a trained deepfake model."""
    image = ensure_rgb_uint8(image_rgb).copy()
    x, y, width, height = _face_box(image)
    face = image[y : y + height, x : x + width].copy()
    smooth = cv2.bilateralFilter(face, 15, 85, 85)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.82
    hsv[:, :, 2] *= 1.07
    smooth = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    rows, cols = smooth.shape[:2]
    source = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    destination = np.float32(
        [[int(0.035 * cols), int(0.020 * rows)], [cols - 1 - int(0.040 * cols), int(0.035 * rows)], [int(0.020 * cols), rows - 1 - int(0.025 * rows)]]
    )
    warped = cv2.warpAffine(smooth, cv2.getAffineTransform(source, destination), (cols, rows), borderMode=cv2.BORDER_REFLECT)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (width // 2, height // 2), (int(0.47 * width), int(0.48 * height)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    image[y : y + height, x : x + width] = (alpha * warped + (1.0 - alpha) * face).astype(np.uint8)
    return image


def generate_facial_attack_suite(watermarked_rgb: np.ndarray, donor_rgb: np.ndarray) -> list[AttackImage]:
    target = cv2.resize(ensure_rgb_uint8(watermarked_rgb), (512, 512), interpolation=cv2.INTER_CUBIC)
    donor = cv2.resize(ensure_rgb_uint8(donor_rgb), (512, 512), interpolation=cv2.INTER_CUBIC)
    return [
        AttackImage("genuine_watermarked", "control", target.copy(), "Original watermarked query"),
        AttackImage("face_swap", "identity_manipulation", controlled_face_swap(target, donor), "Controlled donor face swap"),
        AttackImage("identity_replacement", "identity_manipulation", donor.copy(), "Complete donor identity replacement"),
        AttackImage("mouth_blur", "local_tampering", mouth_blur(target), "Mouth-region blurring"),
        AttackImage("eyes_occlusion", "local_tampering", eyes_occlusion(target), "Eye-region occlusion"),
        AttackImage("cheek_alteration", "local_tampering", cheek_alteration(target), "Cheek-region copy-move alteration"),
        AttackImage(
            "controlled_synthetic_manipulation",
            "controlled_synthetic_manipulation",
            controlled_synthetic_manipulation(target),
            "Controlled smoothing, color shift, and geometric warping; not a generative deepfake",
        ),
    ]
