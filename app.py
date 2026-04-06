
import io
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageStat, ImageDraw, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None

import streamlit as st

st.set_page_config(
    page_title="ÇOFSAT Fotoğraf Ön Değerlendirme",
    layout="wide",
    page_icon="📷",
)

PHI = 1.61803398875
MAX_ANALYSIS_SIZE = 1600

CULTURE = {
    "ad": "ÇOFSAT",
    "amac": "Fotoğrafı yalnızca beğeni nesnesi olmaktan çıkarıp, niyet, kadraj, anlatı ve görsel dil üzerinden okumak.",
    "temel_soru": "Bu fotoğraf neden var?",
    "okuma_sorulari": [
        "İlk bakışta beni durduran ne?",
        "Gözüm kadrajda nereye gidiyor?",
        "Fotoğraf sessiz mi, gergin mi, hareketli mi?",
        "Bu görüntü neyi gösteriyor ve neyi dışarıda bırakıyor?",
        "Bu fotoğraf bende nasıl bir his bırakıyor?",
    ],
    "rubric": {
        "ilk_etki": 0.08,
        "teknik_butunluk": 0.10,
        "kompozisyon": 0.12,
        "odak_ve_hiyerarsi": 0.10,
        "anlati_gucu": 0.12,
        "gorsel_dil": 0.08,
        "sadelik": 0.08,
        "niyet_tutarliligi": 0.08,
        "isik_yonu": 0.05,
        "derinlik_hissi": 0.05,
        "dikkat_dagitici_unsurlar": 0.05,
        "zamanlama": 0.03,
        "negatif_alan": 0.03,
        "duygusal_yogunluk": 0.05,
        "editoryal_deger": 0.03,
        "tekrar_bakma_istegi": 0.03,
    },
}

RUBRIC_LABELS = {
    "ilk_etki": "İlk Etki",
    "teknik_butunluk": "Teknik Bütünlük",
    "kompozisyon": "Kompozisyon",
    "odak_ve_hiyerarsi": "Odak ve Hiyerarşi",
    "anlati_gucu": "Anlatı Gücü",
    "gorsel_dil": "Görsel Dil",
    "sadelik": "Sadelik",
    "niyet_tutarliligi": "Niyet Tutarlılığı",
    "isik_yonu": "Işık Yönü",
    "derinlik_hissi": "Derinlik Hissi",
    "dikkat_dagitici_unsurlar": "Dikkat Dağıtan Unsurlar",
    "zamanlama": "Zamanlama",
    "negatif_alan": "Negatif Alan",
    "duygusal_yogunluk": "Duygusal Yoğunluk",
    "editoryal_deger": "Editoryal Değer",
    "tekrar_bakma_istegi": "Tekrar Bakma İsteği",
}

MODE_PROFILES = {
    "Sokak": {
        "description": "Anlık karşılaşmalar, zamanlama, şehir ritmi, insan ve çevre ilişkisi üzerinden okur.",
        "focus_hint": "Sokak fotoğrafında zamanlama, jest, katman ve sahne içi akış çok belirleyicidir.",
    },
    "Portre": {
        "description": "Yüz, ifade, beden dili, bakış ve özneyle kurulan bağ üzerinden okur.",
        "focus_hint": "Portrede duygusal temas, yüzün okunurluğu ve arka plan-özne ilişkisi belirleyicidir.",
    },
    "Belgesel": {
        "description": "Bağlam, tanıklık gücü, sahnenin doğruluğu ve anlatısal dürüstlük üzerinden okur.",
        "focus_hint": "Belgeselde sahnenin anlamı, bağlamı ve görsel dürüstlüğü teknik şatafattan daha değerlidir.",
    },
    "Soyut": {
        "description": "Biçim, ritim, yüzey, ton, boşluk ve görsel dil üzerinden okur.",
        "focus_hint": "Soyutta neyin gösterildiğinden çok, görsel dilin tutarlılığı ve ritmi önemlidir.",
    },
}

EDITOR_MODES = {
    "Yapıcı": {
        "summary_prefix": "Bu karede dikkat çekici bir yön var.",
        "improve_prefix": "Biraz daha güçlenmesi için şu alanlar öne çıkıyor:",
        "ending": "Buradaki amaç kusur bulmak değil, kareyi bir adım ileri taşımak.",
    },
    "Dürüst": {
        "summary_prefix": "Bu kare bazı yerlerde iyi çalışıyor, bazı yerlerde ise net kararlar istiyor.",
        "improve_prefix": "Gelişime açık başlıca alanlar şunlar:",
        "ending": "Burada asıl mesele neyin çalıştığını ve neyin çalışmadığını net görebilmek.",
    },
    "Sert": {
        "summary_prefix": "Bu kare potansiyel taşısa da bazı temel kararlar henüz yerine tam oturmamış görünüyor.",
        "improve_prefix": "En net kırılmalar şuralarda görünüyor:",
        "ending": "Bu tonu seçmek yargılamak için değil, fotoğrafı daha açık görmek içindir.",
    },
}


@dataclass
class ImageMetrics:
    width: int
    height: int
    aspect_ratio: float
    brightness_mean: float
    brightness_std: float
    contrast_std: float
    highlight_clip_ratio: float
    shadow_clip_ratio: float
    edge_density: float
    focus_score: float
    center_of_mass_x: float
    center_of_mass_y: float
    symmetry_score: float
    visual_noise_score: float
    thirds_alignment_score: float
    negative_space_score: float
    tonal_balance_score: float
    dynamic_tension_score: float
    left_brightness: float
    right_brightness: float
    top_brightness: float
    bottom_brightness: float


@dataclass
class CritiqueResult:
    total_score: float
    overall_level: str
    overall_tag: str
    rubric_scores: Dict[str, float]
    strengths: List[str]
    development_areas: List[str]
    editor_summary: str
    first_reading: str
    structural_reading: str
    editorial_result: str
    shooting_notes: List[str]
    editing_notes: List[str]
    reading_prompts: List[str]
    tags: List[str]
    key_strength: str
    key_issue: str
    one_move_improvement: str
    suggested_mode: str
    suggested_mode_reason: str
    metrics: Dict


def find_logo_file() -> Optional[str]:
    candidates = ["logo.png", "logo.jpg", "logo.jpeg", "2.jpeg", "2.jpg"]
    for name in candidates:
        if Path(name).exists():
            return name
    return None


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def tone_text(text: str, editor_mode: str) -> str:
    if editor_mode == "Yapıcı":
        return text
    if editor_mode == "Dürüst":
        return text.replace("biraz ", "").replace("nispeten ", "")
    if editor_mode == "Sert":
        return (
            text.replace("biraz ", "")
            .replace("nispeten ", "")
            .replace("olabilir", "gerekiyor")
        )
    return text


def normalize_array(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = arr.min()
    mx = arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def safe_resize(img: Image.Image, max_size: int = MAX_ANALYSIS_SIZE) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_size:
        return img
    scale = max_size / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def pil_to_gray_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"))


def estimate_focus_score(gray: np.ndarray) -> float:
    if cv2 is None:
        gy, gx = np.gradient(gray.astype(np.float32))
        return float(np.var(gx) + np.var(gy))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def estimate_edge_density(gray: np.ndarray) -> float:
    if cv2 is None:
        gy, gx = np.gradient(gray.astype(np.float32))
        mag = np.sqrt(gx**2 + gy**2)
        return float((mag > np.percentile(mag, 80)).mean())
    edges = cv2.Canny(gray, 100, 200)
    return float((edges > 0).mean())


def estimate_center_of_mass(gray: np.ndarray) -> Tuple[float, float]:
    inv = 255 - gray.astype(np.float32)
    total = inv.sum() + 1e-9
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((xs * inv).sum() / total) / w
    cy = float((ys * inv).sum() / total) / h
    return cx, cy


def estimate_symmetry(gray: np.ndarray) -> float:
    h, w = gray.shape
    left = gray[:, : w // 2]
    right = gray[:, w - left.shape[1]:]
    right = np.fliplr(right)
    diff = np.abs(left.astype(np.float32) - right.astype(np.float32)).mean() / 255.0
    return float(max(0.0, 1.0 - diff))


def estimate_visual_noise(gray: np.ndarray) -> float:
    small = np.array(Image.fromarray(gray).resize((64, 64)))
    gy, gx = np.gradient(small.astype(np.float32))
    local_activity = np.sqrt(gx**2 + gy**2)
    return float(np.clip(local_activity.mean() / 40.0, 0, 1))


def estimate_thirds_alignment(cx: float, cy: float) -> float:
    thirds_x = [1 / 3, 2 / 3]
    thirds_y = [1 / 3, 2 / 3]
    dx = min(abs(cx - p) for p in thirds_x)
    dy = min(abs(cy - p) for p in thirds_y)
    d = math.sqrt(dx * dx + dy * dy)
    return clamp01(1 - d / 0.35)


def estimate_negative_space(gray: np.ndarray) -> float:
    small = np.array(Image.fromarray(gray).resize((96, 96))).astype(np.float32)
    gy, gx = np.gradient(small)
    mag = np.sqrt(gx**2 + gy**2)
    low_activity = (mag < np.percentile(mag, 35)).mean()
    return float(low_activity)


def estimate_tonal_balance(mean_val: float, std_val: float, hi_clip: float, sh_clip: float) -> float:
    brightness_balance = 1 - abs(mean_val - 127) / 127
    clip_penalty = min(1.0, (hi_clip + sh_clip) * 8)
    contrast = clamp01(std_val / 70)
    return clamp01(0.35 * brightness_balance + 0.4 * contrast + 0.25 * (1 - clip_penalty))


def estimate_dynamic_tension(cx: float, cy: float, symmetry: float) -> float:
    offcenter = clamp01((abs(cx - 0.5) + abs(cy - 0.5)) / 0.3)
    asymmetry = 1 - symmetry
    return clamp01(0.55 * offcenter + 0.45 * asymmetry)


@st.cache_data(show_spinner=False)
def extract_metrics_cached(image_bytes: bytes) -> Dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = safe_resize(img)
    gray = pil_to_gray_np(img)
    stat = ImageStat.Stat(img.convert("L"))
    brightness_mean = float(stat.mean[0])
    brightness_std = float(stat.stddev[0])
    highlight_clip_ratio = float((gray >= 250).mean())
    shadow_clip_ratio = float((gray <= 5).mean())
    edge_density = estimate_edge_density(gray)
    focus_score = estimate_focus_score(gray)
    center_of_mass_x, center_of_mass_y = estimate_center_of_mass(gray)
    symmetry_score = estimate_symmetry(gray)
    visual_noise_score = estimate_visual_noise(gray)
    thirds_alignment_score = estimate_thirds_alignment(center_of_mass_x, center_of_mass_y)
    negative_space_score = estimate_negative_space(gray)
    tonal_balance_score = estimate_tonal_balance(
        brightness_mean, brightness_std, highlight_clip_ratio, shadow_clip_ratio
    )
    dynamic_tension_score = estimate_dynamic_tension(
        center_of_mass_x, center_of_mass_y, symmetry_score
    )

    h, w = gray.shape
    left_brightness = float(gray[:, : max(1, w // 2)].mean())
    right_brightness = float(gray[:, w // 2:].mean())
    top_brightness = float(gray[: max(1, h // 2), :].mean())
    bottom_brightness = float(gray[h // 2:, :].mean())

    return asdict(
        ImageMetrics(
            width=img.width,
            height=img.height,
            aspect_ratio=img.width / max(1, img.height),
            brightness_mean=brightness_mean,
            brightness_std=brightness_std,
            contrast_std=brightness_std,
            highlight_clip_ratio=highlight_clip_ratio,
            shadow_clip_ratio=shadow_clip_ratio,
            edge_density=edge_density,
            focus_score=focus_score,
            center_of_mass_x=center_of_mass_x,
            center_of_mass_y=center_of_mass_y,
            symmetry_score=symmetry_score,
            visual_noise_score=visual_noise_score,
            thirds_alignment_score=thirds_alignment_score,
            negative_space_score=negative_space_score,
            tonal_balance_score=tonal_balance_score,
            dynamic_tension_score=dynamic_tension_score,
            left_brightness=left_brightness,
            right_brightness=right_brightness,
            top_brightness=top_brightness,
            bottom_brightness=bottom_brightness,
        )
    )


def normalize_focus(metrics: ImageMetrics) -> float:
    return clamp01(1 - abs((math.log1p(metrics.focus_score) - 4.2) / 3.0))


def score_first_impact(metrics: ImageMetrics) -> float:
    return clamp01(
        0.35 * normalize_focus(metrics)
        + 0.35 * metrics.dynamic_tension_score
        + 0.30 * metrics.tonal_balance_score
    )


def score_technical(metrics: ImageMetrics) -> float:
    clip_penalty = min(1.0, (metrics.highlight_clip_ratio + metrics.shadow_clip_ratio) * 8)
    return clamp01(
        0.4 * metrics.tonal_balance_score
        + 0.35 * normalize_focus(metrics)
        + 0.25 * (1 - clip_penalty)
    )


def score_composition(metrics: ImageMetrics) -> float:
    return clamp01(
        0.34 * metrics.thirds_alignment_score
        + 0.22 * metrics.dynamic_tension_score
        + 0.22 * metrics.symmetry_score
        + 0.22 * (1 - abs(metrics.negative_space_score - 0.45))
    )


def score_hierarchy(metrics: ImageMetrics) -> float:
    focus = normalize_focus(metrics)
    edge_balance = clamp01(1 - abs(metrics.edge_density - 0.10) / 0.18)
    return clamp01(0.45 * focus + 0.35 * edge_balance + 0.20 * metrics.thirds_alignment_score)


def score_narrative(metrics: ImageMetrics) -> float:
    return clamp01(
        0.35 * score_hierarchy(metrics)
        + 0.35 * score_composition(metrics)
        + 0.30 * metrics.tonal_balance_score
    )


def score_abstraction(metrics: ImageMetrics) -> float:
    return clamp01(
        0.35 * metrics.symmetry_score
        + 0.30 * metrics.negative_space_score
        + 0.35 * metrics.tonal_balance_score
    )


def score_simplification(metrics: ImageMetrics) -> float:
    edge_penalty = clamp01((metrics.edge_density - 0.12) / 0.25)
    return clamp01(1 - (0.55 * edge_penalty + 0.45 * metrics.visual_noise_score))


def score_intention(metrics: ImageMetrics) -> float:
    return clamp01(
        0.30 * score_composition(metrics)
        + 0.25 * score_hierarchy(metrics)
        + 0.25 * score_simplification(metrics)
        + 0.20 * score_technical(metrics)
    )


def score_light_direction(metrics: ImageMetrics) -> float:
    horizontal_sep = abs(metrics.left_brightness - metrics.right_brightness) / 255.0
    vertical_sep = abs(metrics.top_brightness - metrics.bottom_brightness) / 255.0
    separation = max(horizontal_sep, vertical_sep)
    return clamp01(0.4 + separation * 1.1)


def score_depth(metrics: ImageMetrics) -> float:
    return clamp01(
        0.35 * metrics.dynamic_tension_score
        + 0.30 * (1 - metrics.symmetry_score)
        + 0.35 * clamp01(metrics.edge_density / 0.20)
    )


def score_distraction_control(metrics: ImageMetrics) -> float:
    distraction = clamp01(0.55 * metrics.visual_noise_score + 0.45 * clamp01(metrics.edge_density / 0.22))
    return clamp01(1 - distraction)


def score_timing(metrics: ImageMetrics, mode: str) -> float:
    base = clamp01(
        0.45 * metrics.dynamic_tension_score
        + 0.30 * normalize_focus(metrics)
        + 0.25 * metrics.tonal_balance_score
    )
    if mode == "Sokak":
        return clamp01(base * 1.10)
    if mode == "Belgesel":
        return clamp01(base * 1.05)
    return base


def score_negative_space_usage(metrics: ImageMetrics) -> float:
    return clamp01(1 - abs(metrics.negative_space_score - 0.45) / 0.35)


def score_emotional_intensity(metrics: ImageMetrics, mode: str) -> float:
    base = clamp01(
        0.40 * score_narrative(metrics)
        + 0.35 * metrics.tonal_balance_score
        + 0.25 * metrics.dynamic_tension_score
    )
    if mode == "Portre":
        return clamp01(base * 1.08)
    if mode == "Sokak":
        return clamp01(base * 1.05)
    return base


def score_editorial_value(metrics: ImageMetrics) -> float:
    return clamp01(
        0.30 * score_composition(metrics)
        + 0.25 * score_hierarchy(metrics)
        + 0.20 * score_simplification(metrics)
        + 0.25 * score_narrative(metrics)
    )


def score_revisit_desire(metrics: ImageMetrics, mode: str) -> float:
    base = clamp01(
        0.30 * score_first_impact(metrics)
        + 0.35 * score_narrative(metrics)
        + 0.20 * score_abstraction(metrics)
        + 0.15 * score_editorial_value(metrics)
    )
    if mode == "Soyut":
        return clamp01(base * 1.08)
    return base


def mode_adjustment(scores: Dict[str, float], mode: str) -> Dict[str, float]:
    adjusted = scores.copy()

    if mode == "Sokak":
        adjusted["anlati_gucu"] = clamp01(adjusted["anlati_gucu"] * 1.08)
        adjusted["ilk_etki"] = clamp01(adjusted["ilk_etki"] * 1.05)
        adjusted["zamanlama"] = clamp01(adjusted["zamanlama"] * 1.12)

    elif mode == "Portre":
        adjusted["odak_ve_hiyerarsi"] = clamp01(adjusted["odak_ve_hiyerarsi"] * 1.08)
        adjusted["anlati_gucu"] = clamp01(adjusted["anlati_gucu"] * 1.06)
        adjusted["duygusal_yogunluk"] = clamp01(adjusted["duygusal_yogunluk"] * 1.10)

    elif mode == "Belgesel":
        adjusted["niyet_tutarliligi"] = clamp01(adjusted["niyet_tutarliligi"] * 1.08)
        adjusted["anlati_gucu"] = clamp01(adjusted["anlati_gucu"] * 1.06)
        adjusted["editoryal_deger"] = clamp01(adjusted["editoryal_deger"] * 1.08)

    elif mode == "Soyut":
        adjusted["gorsel_dil"] = clamp01(adjusted["gorsel_dil"] * 1.12)
        adjusted["kompozisyon"] = clamp01(adjusted["kompozisyon"] * 1.06)
        adjusted["negatif_alan"] = clamp01(adjusted["negatif_alan"] * 1.08)
        adjusted["tekrar_bakma_istegi"] = clamp01(adjusted["tekrar_bakma_istegi"] * 1.08)

    return adjusted


def build_rubric_scores(metrics: ImageMetrics, mode: str) -> Dict[str, float]:
    base_scores = {
        "ilk_etki": score_first_impact(metrics),
        "teknik_butunluk": score_technical(metrics),
        "kompozisyon": score_composition(metrics),
        "odak_ve_hiyerarsi": score_hierarchy(metrics),
        "anlati_gucu": score_narrative(metrics),
        "gorsel_dil": score_abstraction(metrics),
        "sadelik": score_simplification(metrics),
        "niyet_tutarliligi": score_intention(metrics),
        "isik_yonu": score_light_direction(metrics),
        "derinlik_hissi": score_depth(metrics),
        "dikkat_dagitici_unsurlar": score_distraction_control(metrics),
        "zamanlama": score_timing(metrics, mode),
        "negatif_alan": score_negative_space_usage(metrics),
        "duygusal_yogunluk": score_emotional_intensity(metrics, mode),
        "editoryal_deger": score_editorial_value(metrics),
        "tekrar_bakma_istegi": score_revisit_desire(metrics, mode),
    }
    return mode_adjustment(base_scores, mode)


def weighted_total(scores: Dict[str, float]) -> float:
    total = sum(CULTURE["rubric"][key] * val for key, val in scores.items())
    return round(total * 100, 1)


def score_band(score_100: float) -> str:
    if score_100 < 45:
        return "Gelişmeye Açık"
    if score_100 < 65:
        return "Orta"
    if score_100 < 80:
        return "Güçlü"
    return "Çok Güçlü"


def suggest_mode(metrics: ImageMetrics, scores_100: Dict[str, float]) -> Tuple[str, str]:
    candidates = {
        "Sokak": (
            0.32 * scores_100["zamanlama"]
            + 0.24 * scores_100["anlati_gucu"]
            + 0.18 * scores_100["derinlik_hissi"]
            + 0.12 * scores_100["dikkat_dagitici_unsurlar"]
            + 0.14 * scores_100["tekrar_bakma_istegi"]
        ),
        "Portre": (
            0.30 * scores_100["odak_ve_hiyerarsi"]
            + 0.28 * scores_100["duygusal_yogunluk"]
            + 0.18 * scores_100["isik_yonu"]
            + 0.12 * scores_100["anlati_gucu"]
            + 0.12 * scores_100["sadelik"]
        ),
        "Belgesel": (
            0.30 * scores_100["niyet_tutarliligi"]
            + 0.26 * scores_100["editoryal_deger"]
            + 0.20 * scores_100["anlati_gucu"]
            + 0.12 * scores_100["zamanlama"]
            + 0.12 * scores_100["tekrar_bakma_istegi"]
        ),
        "Soyut": (
            0.34 * scores_100["gorsel_dil"]
            + 0.24 * scores_100["kompozisyon"]
            + 0.18 * scores_100["negatif_alan"]
            + 0.12 * scores_100["tekrar_bakma_istegi"]
            + 0.12 * scores_100["isik_yonu"]
        ),
    }

    best_mode = max(candidates.items(), key=lambda x: x[1])[0]

    if best_mode == "Sokak":
        reason = "Zamanlama, anlatı ve sahne akışı birlikte çalıştığı için bu kare sokak okumasına daha yakın görünüyor."
    elif best_mode == "Portre":
        reason = "Odak, duygusal yoğunluk ve ışığın özneyi taşıma biçimi bu kareyi portreye yaklaştırıyor."
    elif best_mode == "Belgesel":
        reason = "Niyet, editoryal ağırlık ve bağlam hissi bu karede belgesel tarafı öne çıkarıyor."
    else:
        reason = "Biçim, boşluk ve görsel dil ilişkisi bu kareyi soyut okumaya daha yakın kılıyor."

    return best_mode, reason


def overall_tag_from_scores(scores_100: Dict[str, float], mode: str) -> str:
    if mode == "Sokak":
        if scores_100["anlati_gucu"] >= 75:
            return "Sahne duygusu güçlü"
        if scores_100["zamanlama"] >= 75:
            return "Zamanlama iyi"
        return "Sokak potansiyeli yüksek"
    if mode == "Portre":
        if scores_100["duygusal_yogunluk"] >= 75:
            return "Duygusal temas güçlü"
        if scores_100["odak_ve_hiyerarsi"] >= 75:
            return "Yüz ve ifade okunuyor"
        return "Portre bağı kuruluyor"
    if mode == "Belgesel":
        if scores_100["editoryal_deger"] >= 75:
            return "Belgesel değeri güçlü"
        if scores_100["niyet_tutarliligi"] >= 75:
            return "Tanıklık hissi güçlü"
        return "Bağlam taşıyor"
    if mode == "Soyut":
        if scores_100["gorsel_dil"] >= 75:
            return "Görsel dili güçlü"
        if scores_100["tekrar_bakma_istegi"] >= 75:
            return "Tekrar bakma isteği yaratıyor"
        return "Soyut potansiyel taşıyor"
    return "Potansiyeli olan kare"


def pick_strengths(scores_100: Dict[str, float], editor_mode: str) -> List[str]:
    ordered = sorted(scores_100.items(), key=lambda x: x[1], reverse=True)
    mapping = {
        "ilk_etki": "Fotoğraf ilk bakışta kendine alan açabiliyor; izleyiciyi tamamen dışarıda bırakmıyor.",
        "teknik_butunluk": "Işık, ton ve netlik tamamen rastlantısal görünmüyor; teknik yapı anlatıyı belli ölçüde taşıyabiliyor.",
        "kompozisyon": "Kadrajın iskeleti dağılmıyor; çizgi, denge ve ritim birbirini destekliyor.",
        "odak_ve_hiyerarsi": "Gözün tutunacağı yer büyük ölçüde belli; fotoğraf neye bakmamızı istediğini söyleyebiliyor.",
        "anlati_gucu": "Kare yalnızca bir görüntü değil; küçük de olsa bir duygu ve anlatı alanı açıyor.",
        "gorsel_dil": "Fotoğraf biçimsel olarak kendi dilini kurmaya çalışıyor ve bu çaba hissediliyor.",
        "sadelik": "Gereksiz yük geri çekilmiş; fotoğraf nefes alabiliyor.",
        "niyet_tutarliligi": "Kare tesadüfi görünmüyor; arkasında belli bir niyet ve karar hissi var.",
        "isik_yonu": "Işığın sahne içindeki yönü ve dağılımı fotoğrafın okunmasına destek veriyor.",
        "derinlik_hissi": "Kare düz kalmıyor; katman ya da hacim hissi izleyiciyi içeri çekiyor.",
        "dikkat_dagitici_unsurlar": "Dikkat dağıtan unsurlar büyük ölçüde kontrol altında tutulmuş.",
        "zamanlama": "Fotoğrafın zamanlaması karenin etkisini belirgin biçimde güçlendiriyor.",
        "negatif_alan": "Boşluk kullanımı yalnızca boş bırakmak değil; anlatıyı taşıyan bir yapı kuruyor.",
        "duygusal_yogunluk": "Fotoğrafın duygusal yoğunluğu yapay durmadan hissedilebiliyor.",
        "editoryal_deger": "Bu kare yalnızca hoş görünmüyor; editoryal olarak da ağırlık taşıyor.",
        "tekrar_bakma_istegi": "Kare bir kez bakılıp geçilmiyor; izleyiciyi yeniden dönmeye davet ediyor.",
    }
    return [tone_text(mapping[k], editor_mode) for k, _ in ordered[:4]]


def pick_development_areas(scores_100: Dict[str, float], editor_mode: str) -> List[str]:
    ordered = sorted(scores_100.items(), key=lambda x: x[1])
    mapping = {
        "ilk_etki": "İlk bakışta izleyiciyi durduracak kadar güçlü bir giriş henüz tam kurulamıyor.",
        "teknik_butunluk": "Ton, ışık veya netlik düzeyi anlamı desteklemek yerine yer yer onu zayıflatıyor.",
        "kompozisyon": "Kadrajın grafik yapısı daha bilinçli kurulursa fotoğraf çok daha güçlü okunabilir.",
        "odak_ve_hiyerarsi": "Ana ağırlık merkezi daha netleşirse göz fotoğraf içinde daha kararlı dolaşır.",
        "anlati_gucu": "Fotoğrafın his ve anlatı tarafı var ama henüz tam açılmamış; daha net bir yön istiyor.",
        "gorsel_dil": "Biçimsel tercihlerin ortak bir dile dönüşmesi için birkaç kararın netleşmesi gerekiyor.",
        "sadelik": "Kadrajdan bazı şeyleri çıkarmak ya da geri itmek fotoğrafın anlatısını rahatlatabilir.",
        "niyet_tutarliligi": "Fotoğrafın neden var olduğu biraz daha görünür hale gelirse etkisi belirginleşir.",
        "isik_yonu": "Işığın ana özneyi taşıma biçimi daha bilinçli kurulursa kare daha kararlı okunur.",
        "derinlik_hissi": "Karede derinlik duygusu biraz daha güçlenirse fotoğraf daha hacimli görünür.",
        "dikkat_dagitici_unsurlar": "Bazı bölgeler gereğinden fazla dikkat çekiyor; bunların görsel ağırlığını azaltmak iyi olabilir.",
        "zamanlama": "Zamanlama bir kademe daha isabetli olursa sahnenin etkisi belirgin biçimde büyür.",
        "negatif_alan": "Boşluk kullanımı biraz daha kontrollü olursa fotoğraf daha rafine görünür.",
        "duygusal_yogunluk": "Duygusal etki var ama izleyiciye daha doğrudan geçmesi için biraz daha açıklık istiyor.",
        "editoryal_deger": "Kare iyi bir fikir taşıyor; fakat editoryal ağırlık için birkaç kararın daha netleşmesi gerekiyor.",
        "tekrar_bakma_istegi": "Fotoğraf ilk bakışta kendini gösteriyor ama ikinci bakış için daha fazla katman isteyebilir.",
    }
    return [tone_text(mapping[k], editor_mode) for k, _ in ordered[:4]]


def build_key_strength(scores_100: Dict[str, float], editor_mode: str) -> str:
    key = max(scores_100.items(), key=lambda x: x[1])[0]
    mapping = {
        "ilk_etki": "Bu karenin en güçlü tarafı ilk temas gücü; izleyiciyi dışarıda bırakmıyor.",
        "teknik_butunluk": "Bu karenin en güçlü tarafı teknik bütünlüğü; ton ve netlik yapıyı taşıyor.",
        "kompozisyon": "Bu karenin en güçlü tarafı kompozisyonu; iskelet dağılmıyor.",
        "odak_ve_hiyerarsi": "Bu karenin en güçlü tarafı odak yapısı; gözün tutunacağı yer belli.",
        "anlati_gucu": "Bu karenin en güçlü tarafı anlatı hissi; yalnızca göstermiyor, bir şey hissettiriyor.",
        "gorsel_dil": "Bu karenin en güçlü tarafı görsel dili; biçimsel kararlar ortak bir yapı kuruyor.",
        "sadelik": "Bu karenin en güçlü tarafı sadeliği; gereksiz yük geri çekilmiş.",
        "niyet_tutarliligi": "Bu karenin en güçlü tarafı niyet hissi; neden var olduğu seziliyor.",
        "isik_yonu": "Bu karenin en güçlü tarafı ışık kullanımı; ışık sahneye anlam katıyor.",
        "derinlik_hissi": "Bu karenin en güçlü tarafı derinlik duygusu; kare hacim kazanıyor.",
        "dikkat_dagitici_unsurlar": "Bu karenin en güçlü tarafı dikkat kontrolü; göz kolay dağılmıyor.",
        "zamanlama": "Bu karenin en güçlü tarafı zamanlaması; an etkisi taşıyor.",
        "negatif_alan": "Bu karenin en güçlü tarafı boşluk kullanımı; alan anlatıya hizmet ediyor.",
        "duygusal_yogunluk": "Bu karenin en güçlü tarafı duygusal yoğunluğu; izleyiciyle temas kuruyor.",
        "editoryal_deger": "Bu karenin en güçlü tarafı editoryal ağırlığı; seçki içinde yer açabilecek bir yapı taşıyor.",
        "tekrar_bakma_istegi": "Bu karenin en güçlü tarafı tekrar bakma isteği yaratması; ilk bakıştan sonra da çalışıyor.",
    }
    return tone_text(mapping[key], editor_mode)


def build_key_issue(scores_100: Dict[str, float], editor_mode: str) -> str:
    key = min(scores_100.items(), key=lambda x: x[1])[0]
    mapping = {
        "ilk_etki": "Bu karede en belirgin sorun ilk temasın zayıf kalması; izleyiciyi durdurma gücü tam açılmıyor.",
        "teknik_butunluk": "Bu karede en belirgin sorun teknik bütünlüğün yer yer dağılması; ton veya netlik anlatıyı zorlayabiliyor.",
        "kompozisyon": "Bu karede en belirgin sorun kompozisyonun yeterince sıkı kurulmaması.",
        "odak_ve_hiyerarsi": "Bu karede en belirgin sorun odak ve hiyerarşinin netleşmemesi.",
        "anlati_gucu": "Bu karede en belirgin sorun anlatının tam açılmaması; his var ama yeterince taşınmıyor.",
        "gorsel_dil": "Bu karede en belirgin sorun görsel dilin henüz tam bütünleşmemesi.",
        "sadelik": "Bu karede en belirgin sorun gereksiz görsel yükün kadraj içinde kalması.",
        "niyet_tutarliligi": "Bu karede en belirgin sorun niyetin yeterince görünür hale gelmemesi.",
        "isik_yonu": "Bu karede en belirgin sorun ışığın ana anlatıyı yeterince taşımaması.",
        "derinlik_hissi": "Bu karede en belirgin sorun derinlik duygusunun zayıf kalması.",
        "dikkat_dagitici_unsurlar": "Bu karede en belirgin sorun dikkat dağıtan bölgelerin fazla baskın olması.",
        "zamanlama": "Bu karede en belirgin sorun zamanlamanın bir kademe daha iyi olabilecek hissi vermesi.",
        "negatif_alan": "Bu karede en belirgin sorun boşluk kullanımının tam kararında oturmaması.",
        "duygusal_yogunluk": "Bu karede en belirgin sorun duygusal etkinin yeterince yoğunlaşmaması.",
        "editoryal_deger": "Bu karede en belirgin sorun editoryal ağırlığın henüz tam oluşmaması.",
        "tekrar_bakma_istegi": "Bu karede en belirgin sorun ikinci bakış için yeterli katman üretmemesi.",
    }
    return tone_text(mapping[key], editor_mode)


def build_one_move_improvement(scores_100: Dict[str, float], editor_mode: str) -> str:
    key = min(scores_100.items(), key=lambda x: x[1])[0]
    mapping = {
        "ilk_etki": "Tek hamlede en büyük iyileşme, giriş etkisini güçlendirecek daha net bir vurgu kurmak olur.",
        "teknik_butunluk": "Tek hamlede en büyük iyileşme, ton ve netliği biraz daha rafine temizlemek olur.",
        "kompozisyon": "Tek hamlede en büyük iyileşme, kadraj iskeletini daha kararlı kurmak olur.",
        "odak_ve_hiyerarsi": "Tek hamlede en büyük iyileşme, ana öznenin görsel ağırlığını netleştirmek olur.",
        "anlati_gucu": "Tek hamlede en büyük iyileşme, fotoğrafın asıl hissetmek istediği şeyi daha görünür kılmak olur.",
        "gorsel_dil": "Tek hamlede en büyük iyileşme, biçimsel kararları daha tek bir dil altında toplamak olur.",
        "sadelik": "Tek hamlede en büyük iyileşme, dikkat dağıtan unsurları kadrajdan çıkarmak ya da geri itmek olur.",
                "niyet_tutarliligi": "Tek hamlede en büyük iyileşme, fotoğrafın neden var olduğunu daha açık hissettirmek olur.",
        "isik_yonu": "Tek hamlede en büyük iyileşme, ışığın ana özneyi taşıma biçimini güçlendirmek olur.",
        "derinlik_hissi": "Tek hamlede en büyük iyileşme, ön-orta-arka plan ilişkisini belirginleştirmek olur.",
        "dikkat_dagitici_unsurlar": "Tek hamlede en büyük iyileşme, fazla dikkat çeken yan alanları bastırmak olur.",
        "zamanlama": "Tek hamlede en büyük iyileşme, daha doğru anı seçmek olur.",
        "negatif_alan": "Tek hamlede en büyük iyileşme, boşluğu daha bilinçli bir anlatı aracına çevirmek olur.",
        "duygusal_yogunluk": "Tek hamlede en büyük iyileşme, duygusal odağı daha görünür hale getirmek olur.",
        "editoryal_deger": "Tek hamlede en büyük iyileşme, karenin editoryal ağırlığını artıracak daha net bir seçim yapmak olur.",
        "tekrar_bakma_istegi": "Tek hamlede en büyük iyileşme, ilk bakıştan sonra da çalışacak ikinci katmanı güçlendirmek olur.",
    }
    return tone_text(mapping[key], editor_mode)


def build_reading_prompts(scores_100: Dict[str, float]) -> List[str]:
    prompts = {
        "ilk_etki": "Bu kare ilk anda beni gerçekten durduruyor mu?",
        "odak_ve_hiyerarsi": "Gözüm nereye gidiyor ve orada kalmak için yeterli sebep buluyor mu?",
        "anlati_gucu": "Bu görüntü bana ne hissettiriyor?",
        "isik_yonu": "Işık burada yalnızca aydınlatıyor mu, yoksa anlam da kuruyor mu?",
        "derinlik_hissi": "Kare düz mü kalıyor, yoksa içine girilecek bir alan açıyor mu?",
        "dikkat_dagitici_unsurlar": "Ana öznenin önüne geçen gereksiz bir ağırlık var mı?",
        "zamanlama": "Bu an yarım saniye önce ya da sonra çekilse etkisi değişir miydi?",
        "negatif_alan": "Boşluk burada gerçekten işe yarıyor mu?",
        "duygusal_yogunluk": "Fotoğraf duyguyu hissettiriyor mu, yoksa sadece işaret mi ediyor?",
        "editoryal_deger": "Bu kare bir seçkide kendine yer açabilir mi?",
        "tekrar_bakma_istegi": "Fotoğrafa ikinci kez dönmek için yeterli neden var mı?",
    }
    ordered = sorted(scores_100.items(), key=lambda x: x[1])
    return [prompts[k] for k, _ in ordered[:4] if k in prompts]


def build_shooting_notes(metrics: ImageMetrics, scores_100: Dict[str, float], mode: str, editor_mode: str) -> List[str]:
    notes = []
    if scores_100["odak_ve_hiyerarsi"] < 60:
        notes.append("Çekim anında ana öznenin görsel ağırlığını daha net kurmak kareyi belirgin biçimde güçlendirebilir.")
    if scores_100["sadelik"] < 60:
        notes.append("Kadrajı sadeleştirip dikkat dağıtan alanları azaltmak fotoğrafın nefesini açar.")
    if scores_100["kompozisyon"] < 60:
        notes.append("Boşluklar ve çizgiler daha bilinçli kullanılırsa göz akışı çok daha güçlü hale gelir.")
    if scores_100["zamanlama"] < 60:
        notes.append("Anı biraz daha sabırla beklemek ya da yarım saniye daha erken davranmak sahnenin etkisini yükseltebilir.")
    if scores_100["derinlik_hissi"] < 60:
        notes.append("Ön plan, orta plan ve arka plan ilişkisini daha bilinçli kurmak kareye hacim kazandırabilir.")
    if mode == "Portre":
        notes.append("Portrede özneyle kurulan küçük bir rahatlık hissi teknik her şeyden daha güçlü çalışabilir.")
    if mode == "Belgesel":
        notes.append("Bağlamı biraz daha görünür bırakmak fotoğrafın tanıklık gücünü artırabilir.")

    unique = []
    for n in notes:
        t = tone_text(n, editor_mode)
        if t not in unique:
            unique.append(t)
    return unique[:5] if unique else ["Bu kare küçük kararlarla daha da güçlenebilir."]


def build_editing_notes(metrics: ImageMetrics, scores_100: Dict[str, float], editor_mode: str) -> List[str]:
    notes = []
    if metrics.highlight_clip_ratio > 0.03:
        notes.append("Highlight bölgelerini biraz geri çekmek, fotoğrafın dikkat dengesini daha kontrollü hale getirir.")
    if metrics.shadow_clip_ratio > 0.08:
        notes.append("Siyahları derinleştirirken bilgi kaybını azaltmak kareye daha rafine bir ton dengesi kazandırır.")
    if scores_100["odak_ve_hiyerarsi"] < 60:
        notes.append("Ana özne çevresinde lokal kontrast veya parlaklık desteği vermek gözün tutunmasını kolaylaştırır.")
    if scores_100["dikkat_dagitici_unsurlar"] < 60:
        notes.append("Dikkat dağıtan bölgeleri ton olarak biraz geri itmek anlatıyı öne çıkarabilir.")
    if scores_100["duygusal_yogunluk"] < 60:
        notes.append("Ton geçişlerini biraz daha yumuşatmak ya da belirginleştirmek duygusal etkiyi güçlendirebilir.")

    unique = []
    for n in notes:
        t = tone_text(n, editor_mode)
        if t not in unique:
            unique.append(t)
    return unique[:5] if unique else ["Bu kare için aşırı filtre yerine küçük ve bilinçli ton düzenlemeleri en iyi sonucu verir."]


def build_first_reading(scores_100: Dict[str, float], editor_mode: str) -> str:
    if scores_100["ilk_etki"] >= 75:
        text = "İlk anda bu kare kendine alan açabiliyor. İzleyiciyi tamamen dışarıda bırakmayan, dikkat toplamayı bilen bir giriş kuruyor."
    elif scores_100["ilk_etki"] >= 60:
        text = "İlk anda fotoğrafın bir niyeti hissediliyor. Etki var; fakat daha güçlü bir ilk temasla kare çok daha akılda kalıcı olabilir."
    else:
        text = "Fotoğraf hemen açılan bir etki kurmakta biraz zorlanıyor. Bu kötü olduğu anlamına gelmez; yalnızca ilk temasını belirginleştirmesi gerektiğini gösterir."
    return tone_text(text, editor_mode)


def build_structural_reading(scores_100: Dict[str, float], editor_mode: str) -> str:
    pieces = []
    if scores_100["kompozisyon"] >= 70:
        pieces.append("kadrajın iskeleti toparlanmış")
    else:
        pieces.append("kadrajın iskeleti daha disiplin istiyor")
    if scores_100["odak_ve_hiyerarsi"] >= 70:
        pieces.append("gözün tutunacağı alan büyük ölçüde belli")
    else:
        pieces.append("odak ve hiyerarşi daha netleşirse okuma rahatlar")
    if scores_100["derinlik_hissi"] >= 65:
        pieces.append("derinlik duygusu kareyi destekliyor")
    else:
        pieces.append("derinlik hissi biraz daha güçlenirse fotoğraf hacim kazanır")
    if scores_100["dikkat_dagitici_unsurlar"] < 60:
        pieces.append("bazı bölgeler ana anlatının önüne geçiyor")
    return tone_text("Yapısal olarak bakıldığında " + ", ".join(pieces) + ".", editor_mode)


def build_editorial_result(total: float, editor_mode: str) -> str:
    if total >= 80:
        text = "Genel sonuç olarak kare yalnızca doğru kararlar içermiyor; aynı zamanda kendi dilini hissettirebiliyor. Editöryel olarak güçlü bir zemini var."
    elif total >= 65:
        text = "Genel sonuç olarak fotoğraf güçlü bir potansiyel taşıyor. Doğru yerlere küçük dokunuşlar gelirse etkisi belirgin biçimde artar."
    elif total >= 45:
        text = "Genel sonuç olarak karede iyi bir niyet var. Bazı kararlar tam yerine oturmamış olsa da üzerinde düşünülmüş bir yön hissediliyor."
    else:
        text = "Genel sonuç olarak bu kare henüz tam açılmamış görünüyor. Burada önemli olan eksik değil; hangi kararların fotoğrafı ileri taşıyacağını görebilmektir."
    return tone_text(text + " " + EDITOR_MODES[editor_mode]["ending"], editor_mode)


def build_editor_summary(total: float, strengths: List[str], dev_areas: List[str], mode: str, editor_mode: str) -> str:
    level = score_band(total)
    prefix = EDITOR_MODES[editor_mode]["summary_prefix"]
    improve = EDITOR_MODES[editor_mode]["improve_prefix"]
    text = (
        f"{prefix} Bu **{mode.lower()}** kare şu an için **{level}** bir potansiyel gösteriyor. "
        f"En güçlü taraflarından biri şu: {strengths[0].lower()} "
        f"{improve} {dev_areas[0].lower()}"
    )
    return tone_text(text, editor_mode)


def build_tags(scores_100: Dict[str, float], total: float, mode: str) -> List[str]:
    tags = [mode]
    if scores_100["anlati_gucu"] >= 75:
        tags.append("Güçlü duygu")
    if scores_100["kompozisyon"] >= 75:
        tags.append("Sağlam kompozisyon")
    if scores_100["isik_yonu"] >= 75:
        tags.append("Işık iyi kullanılmış")
    if scores_100["editoryal_deger"] >= 75:
        tags.append("Editoryal değer")
    if scores_100["tekrar_bakma_istegi"] >= 75:
        tags.append("Tekrar bakma isteği")
    if total >= 80:
        tags.append("Çok güçlü")
    elif total >= 65:
        tags.append("Yüksek potansiyel")
    else:
        tags.append("Geliştirilebilir")
    unique = []
    for tag in tags:
        if tag not in unique:
            unique.append(tag)
    return unique[:6]


def critique_image(image_bytes: bytes, mode: str, editor_mode: str) -> CritiqueResult:
    metrics = ImageMetrics(**extract_metrics_cached(image_bytes))
    scores = build_rubric_scores(metrics, mode)
    total = weighted_total(scores)
    scores_100 = {k: round(v * 100, 1) for k, v in scores.items()}

    strengths = pick_strengths(scores_100, editor_mode)
    dev_areas = pick_development_areas(scores_100, editor_mode)
    suggested_mode, suggested_mode_reason = suggest_mode(metrics, scores_100)

    return CritiqueResult(
        total_score=total,
        overall_level=score_band(total),
        overall_tag=overall_tag_from_scores(scores_100, mode),
        rubric_scores=scores_100,
        strengths=strengths,
        development_areas=dev_areas,
        editor_summary=build_editor_summary(total, strengths, dev_areas, mode, editor_mode),
        first_reading=build_first_reading(scores_100, editor_mode),
        structural_reading=build_structural_reading(scores_100, editor_mode),
        editorial_result=build_editorial_result(total, editor_mode),
        shooting_notes=build_shooting_notes(metrics, scores_100, mode, editor_mode),
        editing_notes=build_editing_notes(metrics, scores_100, editor_mode),
        reading_prompts=build_reading_prompts(scores_100),
        tags=build_tags(scores_100, total, mode),
        key_strength=build_key_strength(scores_100, editor_mode),
        key_issue=build_key_issue(scores_100, editor_mode),
        one_move_improvement=build_one_move_improvement(scores_100, editor_mode),
        suggested_mode=suggested_mode,
        suggested_mode_reason=suggested_mode_reason,
        metrics=asdict(metrics),
    )


@st.cache_data(show_spinner=False)
def get_resized_rgb(image_bytes: bytes) -> Image.Image:
    return safe_resize(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


def build_attention_map(img: Image.Image) -> np.ndarray:
    gray = np.array(img.convert("L")).astype(np.float32)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])

    grad = np.sqrt(gx**2 + gy**2)
    inv = 255.0 - gray

    grad_n = normalize_array(grad)
    inv_n = normalize_array(inv)
    center_bias = build_center_bias(gray.shape[1], gray.shape[0])

    raw = (0.58 * grad_n + 0.22 * inv_n + 0.20 * center_bias).astype(np.float32)

    if cv2 is not None:
        raw = cv2.GaussianBlur(raw, (0, 0), 9)
    else:
        raw = np.array(Image.fromarray((raw * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=6))) / 255.0

    return normalize_array(raw)


def build_center_bias(w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w / 2.0) / max(w, 1)
    yy = (yy - h / 2.0) / max(h, 1)
    dist = np.sqrt(xx**2 + yy**2)
    return normalize_array(1.0 - dist)


def top_regions(attention: np.ndarray, n: int = 3, window: int = 50) -> List[Tuple[int, int]]:
    h, w = attention.shape
    work = attention.copy()
    pts: List[Tuple[int, int]] = []

    for _ in range(n):
        idx = np.argmax(work)
        y, x = np.unravel_index(idx, work.shape)
        pts.append((int(x), int(y)))

        x1 = max(0, x - window)
        x2 = min(w, x + window)
        y1 = max(0, y - window)
        y2 = min(h, y + window)
        work[y1:y2, x1:x2] = 0

    return pts


def distraction_regions(attention: np.ndarray, main_points: List[Tuple[int, int]], n: int = 2) -> List[Tuple[int, int]]:
    h, w = attention.shape
    work = attention.copy()

    for x, y in main_points:
        x1 = max(0, x - 60)
        x2 = min(w, x + 60)
        y1 = max(0, y - 60)
        y2 = min(h, y + 60)
        work[y1:y2, x1:x2] = 0

    return top_regions(work, n=n, window=45)


def draw_arrow(draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], color=(255, 214, 102, 220), width=4) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_len = 14
    p1 = (
        end[0] - arrow_len * math.cos(angle - math.pi / 6),
        end[1] - arrow_len * math.sin(angle - math.pi / 6),
    )
    p2 = (
        end[0] - arrow_len * math.cos(angle + math.pi / 6),
        end[1] - arrow_len * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, p1, p2], fill=color)


def draw_analysis_overlay(
    img: Image.Image,
    main_points: List[Tuple[int, int]],
    distraction_points: List[Tuple[int, int]],
) -> Image.Image:
    out = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if len(main_points) >= 2:
        for a, b in zip(main_points[:-1], main_points[1:]):
            draw_arrow(draw, a, b)

    for i, (x, y) in enumerate(main_points):
        r = 24 if i == 0 else 18
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 214, 102, 255), width=4)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 214, 102, 255))

    for x, y in distraction_points:
        r = 24
        draw.rounded_rectangle((x - r, y - r, x + r, y + r), outline=(255, 99, 99, 235), width=4, radius=6)

    return Image.alpha_composite(out, overlay).convert("RGB")


def build_heatmap_image(img: Image.Image, attention: np.ndarray) -> Image.Image:
    base = img.copy().convert("RGBA")
    alpha = (normalize_array(attention) * 190).astype(np.uint8)

    heat = np.zeros((attention.shape[0], attention.shape[1], 4), dtype=np.uint8)
    heat[..., 0] = 255
    heat[..., 1] = 176
    heat[..., 2] = 66
    heat[..., 3] = alpha

    heat_img = Image.fromarray(heat, mode="RGBA")
    return Image.alpha_composite(base, heat_img).convert("RGB")


def phi_grid_positions(w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(w / PHI)
    x2 = w - x1
    y1 = int(h / PHI)
    y2 = h - y1
    return x1, x2, y1, y2


def draw_phi_grid(img: Image.Image) -> Image.Image:
    out = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = out.size
    x1, x2, y1, y2 = phi_grid_positions(w, h)

    for x in (x1, x2):
        draw.line((x, 0, x, h), fill=(112, 203, 255, 235), width=3)
    for y in (y1, y2):
        draw.line((0, y, w, y), fill=(112, 203, 255, 235), width=3)

    for x in (x1, x2):
        for y in (y1, y2):
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(112, 203, 255, 235))

    return Image.alpha_composite(out, overlay).convert("RGB")


def draw_golden_diagonals(img: Image.Image) -> Image.Image:
    out = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = out.size
    x1, x2, y1, y2 = phi_grid_positions(w, h)

    lines = [
        (0, 0, w, h),
        (w, 0, 0, h),
        (x1, 0, w, y2),
        (0, y1, x2, h),
        (x2, 0, 0, y2),
        (w, y1, x1, h),
    ]
    for i, line in enumerate(lines):
        draw.line(line, fill=(255, 154, 82, 230 if i < 2 else 180), width=3 if i < 2 else 2)

    return Image.alpha_composite(out, overlay).convert("RGB")


def draw_golden_spiral(img: Image.Image) -> Image.Image:
    out = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = out.size

    x, y = 0, 0
    rw, rh = w, h
    direction = 0
    boxes = []

    for _ in range(9):
        boxes.append((x, y, x + rw, y + rh))
        if rw < 24 or rh < 24:
            break
        if direction == 0:
            new_rw = int(rw / PHI)
            x = x + (rw - new_rw)
            rw = new_rw
        elif direction == 1:
            new_rh = int(rh / PHI)
            y = y + (rh - new_rh)
            rh = new_rh
        elif direction == 2:
            rw = int(rw / PHI)
        else:
            rh = int(rh / PHI)
        direction = (direction + 1) % 4

    for i, box in enumerate(boxes[:-1]):
        x1, y1, x2, y2 = box
        phase = i % 4
        if phase == 0:
            start, end = 90, 180
        elif phase == 1:
            start, end = 180, 270
        elif phase == 2:
            start, end = 270, 360
        else:
            start, end = 0, 90
        draw.arc((x1, y1, x2, y2), start=start, end=end, fill=(159, 241, 103, 235), width=4)

    return Image.alpha_composite(out, overlay).convert("RGB")


def distance_to_nearest(points: List[Tuple[int, int]], targets: List[Tuple[int, int]]) -> float:
    if not points or not targets:
        return 1e9
    return min(
        math.dist((px, py), (tx, ty))
        for px, py in points
        for tx, ty in targets
    )


def describe_golden_ratio_fit(main_points: List[Tuple[int, int]], w: int, h: int) -> Tuple[str, str]:
    x1, x2, y1, y2 = phi_grid_positions(w, h)
    intersections = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    center = [(w / 2, h / 2)]

    d_intersections = distance_to_nearest(main_points, intersections)
    d_center = distance_to_nearest(main_points, center)

    if d_intersections < min(w, h) * 0.10:
        return (
            "Altın oran ızgarası",
            "Ana dikkat noktaları altın oran kesişimlerine oldukça yakın. Izgara yerleşimi bu kareyi iyi açıklıyor.",
        )
    if d_center < min(w, h) * 0.08:
        return (
            "Altın spiral",
            "Dikkat ağırlığı merkeze toplanıp sonra çevreye açılıyor. Spiral okuması bu karede daha doğal çalışıyor.",
        )
    return (
        "Altın diyagonaller",
        "Göz akışı çapraz ilerliyor. Diyagonal gerilim ve hareket hissi bu karede daha belirgin görünüyor.",
    )


def make_download_button(img: Image.Image, label: str, file_name: str, key: str) -> None:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=file_name,
        mime="image/png",
        key=key,
        use_container_width=True,
    )


def score_color(score: float) -> str:
    if score >= 80:
        return "#5BE49B"
    if score >= 65:
        return "#F4D35E"
    return "#FF7B7B"


def render_score_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-title">{title}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pill_row(tags: List[str]) -> None:
    html = "".join([f"<span class='tag-pill'>{tag}</span>" for tag in tags])
    st.markdown(html, unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1020;
            --card: rgba(255,255,255,0.06);
            --card-border: rgba(255,255,255,0.12);
            --text: #eef2ff;
            --muted: #aab4d1;
            --accent: #f8d66d;
            --accent-2: #68d5ff;
            --danger: #ff7f7f;
            --ok: #61e6a5;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(104,213,255,0.10), transparent 22%),
                radial-gradient(circle at top right, rgba(248,214,109,0.12), transparent 20%),
                linear-gradient(180deg, #0b1020 0%, #10172d 100%);
        }
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
        h1, h2, h3, h4, h5, h6, p, li, span, label, div {color: var(--text);}
        .hero {
            background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 1.4rem 1.4rem 1.25rem 1.4rem;
            box-shadow: 0 24px 60px rgba(0,0,0,0.20);
            backdrop-filter: blur(10px);
        }
        .hero h1 {
            font-size: 2rem;
            margin: 0 0 .4rem 0;
            line-height: 1.1;
        }
        .hero p {
            color: var(--muted);
            font-size: 1rem;
            margin-bottom: .9rem;
        }
        .hero-badges {display:flex; flex-wrap:wrap; gap:.55rem;}
        .hero-badge, .ghost-badge {
            display:inline-flex;
            align-items:center;
            gap:.4rem;
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 999px;
            padding: .35rem .7rem;
            font-size: .86rem;
            color: #f3f6ff;
            background: rgba(255,255,255,0.05);
        }
        .ghost-badge {background: rgba(255,255,255,0.035); color: var(--muted);}
        .sidebar-card, .panel-card, .summary-card, .score-box, .chart-card, .upload-panel {
            background: rgba(255,255,255,0.055);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 18px 45px rgba(0,0,0,.18);
            backdrop-filter: blur(8px);
        }
        .score-box {padding: 1.1rem 1rem;}
        .section-title {
            font-size: 1.06rem;
            font-weight: 700;
            margin: 0 0 .65rem 0;
        }
        .mini-note, .muted-note, .stat-caption {
            color: var(--muted);
            font-size: .93rem;
            line-height: 1.5;
        }
        .tag-pill {
            display:inline-block;
            margin: 0 .4rem .5rem 0;
            padding:.38rem .72rem;
            border-radius:999px;
            font-size:.83rem;
            border:1px solid rgba(255,255,255,.12);
            background: rgba(255,255,255,.06);
        }
        .stat-grid {
            display:grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .8rem;
        }
        .stat-card {
            background: rgba(255,255,255,.05);
            border:1px solid rgba(255,255,255,.10);
            border-radius:18px;
            padding: .95rem;
            min-height: 108px;
        }
        .stat-title {font-size: .82rem; color: var(--muted); margin-bottom:.35rem;}
        .stat-value {font-size: 1.45rem; font-weight: 800; line-height: 1.1; margin-bottom: .3rem;}
        .editor-title {font-weight: 700; margin-bottom: .4rem;}
        .rubric-row {
            margin-bottom: .75rem;
        }
        .rubric-head {
            display:flex; justify-content:space-between; gap:.7rem;
            font-size:.92rem; margin-bottom:.3rem;
        }
        .rubric-track {
            height: 10px; border-radius:999px; overflow:hidden;
            background: rgba(255,255,255,0.08);
        }
        .rubric-fill {
            height: 100%;
            border-radius:999px;
            background: linear-gradient(90deg, rgba(104,213,255,0.95), rgba(248,214,109,0.95));
        }
        .upload-panel {
            text-align:center;
            padding: 1rem;
            margin-top: 1rem;
        }
        .upload-title {font-size: 1.2rem; font-weight: 700; margin-bottom: .25rem;}
        .upload-hint {color: var(--muted); font-size: .95rem; margin-bottom: .8rem;}
        .stFileUploader > div > div {
            border: 2px dashed rgba(255,255,255,0.18) !important;
            border-radius: 20px !important;
            background: rgba(255,255,255,0.03) !important;
            padding: 1rem !important;
        }
        .stFileUploader small {color: var(--muted) !important;}
        .panel-title {font-size: 1.02rem; font-weight: 700; margin-bottom: .55rem;}
        .scheme-name {
            display:inline-block; padding:.35rem .6rem; border-radius:999px;
            border:1px solid rgba(104,213,255,.25); background: rgba(104,213,255,.09);
            margin-bottom:.5rem; font-size:.88rem;
        }
        .footer-note {color: var(--muted); font-size: .9rem; text-align:center; margin-top: 2rem;}
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.05);
            border:1px solid rgba(255,255,255,.10);
            border-radius:18px;
            padding: .75rem;
        }
        div[data-testid="stMetric"] label {color: var(--muted)!important;}
        .stTabs [data-baseweb="tab-list"] {
            gap: .5rem;
            background: rgba(255,255,255,.04);
            padding: .35rem;
            border-radius: 16px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: .55rem .9rem;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(255,255,255,.08) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_rubric_scores(scores_100: Dict[str, float]) -> None:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Rubric puanları</div>", unsafe_allow_html=True)
    for key, value in scores_100.items():
        st.markdown(
            f"""
            <div class="rubric-row">
                <div class="rubric-head">
                    <span>{RUBRIC_LABELS.get(key, key)}</span>
                    <span>{value:.1f}</span>
                </div>
                <div class="rubric-track">
                    <div class="rubric-fill" style="width:{value}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_bullets(title: str, items: List[str], icon: str = "•") -> None:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    for item in items:
        st.markdown(f"{icon} {item}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar(selected_mode: str, selected_editor_mode: str) -> None:
    with st.sidebar:
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown("### ÇOFSAT Kontrol Alanı")
        st.markdown(
            f"<div class='mini-note'>{MODE_PROFILES[selected_mode]['description']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**Fotoğraf türü**")
        st.markdown(f"<div class='ghost-badge'>{selected_mode}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='mini-note'>{MODE_PROFILES[selected_mode]['focus_hint']}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Editör tonu**")
        st.markdown(f"<div class='ghost-badge'>{selected_editor_mode}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='mini-note'>{EDITOR_MODES[selected_editor_mode]['ending']}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Manifesto sorusu**")
        st.info(CULTURE["temel_soru"])

        st.markdown("**Kısa kullanım**")
        st.markdown(
            "- Fotoğrafı yükle\n- Tür ve tonu seç\n- Heatmap, göz akışı ve altın oran katmanlarını incele\n- Çekim ve düzenleme notlarını uygula"
        )
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_css()

    logo_file = find_logo_file()

    with st.sidebar:
        selected_mode = st.selectbox("Tür", list(MODE_PROFILES.keys()), index=0)
        selected_editor_mode = st.selectbox("Ton", list(EDITOR_MODES.keys()), index=0)

    render_sidebar(selected_mode, selected_editor_mode)

    header_left, header_right = st.columns([0.14, 0.86])
    with header_left:
        if logo_file:
            st.image(logo_file, use_container_width=True)

    with header_right:
        st.markdown(
            f"""
            <div class="hero">
                <h1>ÇOFSAT Fotoğraf Ön Değerlendirme</h1>
                <p>
                    Fotoğrafı yalnızca göstermek için değil, okumak için ele alan;
                    daha profesyonel, daha hızlı ve daha net sonuç veren yeni arayüz.
                </p>
                <div class="hero-badges">
                    <span class="hero-badge">Aktif tür: {selected_mode}</span>
                    <span class="hero-badge">Ton: {selected_editor_mode}</span>
                    <span class="ghost-badge">Heatmap + Göz Akışı + Altın Oran</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="upload-panel">
            <div class="upload-title">Fotoğraf yükle</div>
            <div class="upload-hint">Beyaz yükleme alanına tıklayın ya da dosyayı sürükleyip bırakın.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        label="Yükle",
        type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
        help="JPG, JPEG, PNG, WEBP, TIF ve TIFF desteklenir.",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="panel-card">
                <div class="section-title">Hazır olduğunda analiz burada görünecek</div>
                <div class="mini-note">
                    Sonuç ekranında genel skor, editör özeti, heatmap, göz akışı, dikkat dağıtan alanlar,
                    altın oran şemaları, otomatik tür önerisi ve çekim/düzenleme notları yer alır.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    image_bytes = uploaded_file.getvalue()

    with st.spinner("Fotoğraf okunuyor, metrikler çıkarılıyor ve görsel katmanlar hazırlanıyor..."):
        image = get_resized_rgb(image_bytes)
        result = critique_image(image_bytes, selected_mode, selected_editor_mode)

        attention = build_attention_map(image)
        main_points = top_regions(attention, n=3, window=max(35, min(image.size) // 12))
        distraction_points = distraction_regions(attention, main_points, n=2)

        overlay_img = draw_analysis_overlay(image, main_points, distraction_points)
        heatmap_img = build_heatmap_image(image, attention)

        phi_grid_img = draw_phi_grid(image)
        diagonal_img = draw_golden_diagonals(image)
        spiral_img = draw_golden_spiral(image)

        best_scheme, scheme_reason = describe_golden_ratio_fit(main_points, image.size[0], image.size[1])

    top_left, top_right = st.columns([1.08, 0.92], gap="large")

    with top_left:
        st.image(image, caption=f"Yüklenen fotoğraf · {uploaded_file.name}", use_container_width=True)

    with top_right:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Genel sonuç</div>", unsafe_allow_html=True)
        render_score_card("Skor", f"{result.total_score:.1f}/100", result.overall_level)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Genel Seviye", result.overall_level, result.overall_tag)
        with c2:
            st.metric("Önerilen Tür", result.suggested_mode)
        with c3:
            st.metric("Aktif Ton", selected_editor_mode)
        st.progress(result.total_score / 100.0)
        st.info(result.editor_summary)
        render_pill_row(result.tags)
        st.markdown("</div>", unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3, gap="large")
    with s1:
        st.markdown(
            f"<div class='summary-card'><div class='editor-title'>Ana güçlü taraf</div><div class='mini-note'>{result.key_strength}</div></div>",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"<div class='summary-card'><div class='editor-title'>Ana sorun</div><div class='mini-note'>{result.key_issue}</div></div>",
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"<div class='summary-card'><div class='editor-title'>Tek hamlede iyileştirme</div><div class='mini-note'>{result.one_move_improvement}</div></div>",
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Görsel Katmanlar", "Editör Okuması", "Rubric", "Altın Oran", "Teknik"]
    )

    with tab1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div class='panel-title'>Odak noktası · Dikkat dağıtan alan · Göz akışı</div>", unsafe_allow_html=True)
            st.image(overlay_img, use_container_width=True)
            st.caption("Sarı daireler güçlü odak bölgeleri, kırmızı kutular dikkat dağıtabilecek alanlar, sarı oklar muhtemel göz akışını gösterir.")
            make_download_button(overlay_img, "Overlay indir", "cofsat_overlay.png", "dl_overlay")
        with c2:
            st.markdown("<div class='panel-title'>Heatmap</div>", unsafe_allow_html=True)
            st.image(heatmap_img, use_container_width=True)
            st.caption("Turuncu yoğun alanlar görsel enerjinin ve dikkatin toplandığı bölgeleri temsil eder.")
            make_download_button(heatmap_img, "Heatmap indir", "cofsat_heatmap.png", "dl_heatmap")

    with tab2:
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            render_bullets("Güçlü yönler", result.strengths, "✅")
            render_bullets("Gelişim alanları", result.development_areas, "⚠️")
        with c2:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>İlk okuma</div>", unsafe_allow_html=True)
            st.write(result.first_reading)
            st.markdown("<div class='section-title'>Yapısal okuma</div>", unsafe_allow_html=True)
            st.write(result.structural_reading)
            st.markdown("<div class='section-title'>Editöryel sonuç</div>", unsafe_allow_html=True)
            st.write(result.editorial_result)
            st.markdown("</div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2, gap="large")
        with c3:
            render_bullets("Çekim notları", result.shooting_notes, "📷")
        with c4:
            render_bullets("Düzenleme notları", result.editing_notes, "🎛️")

        render_bullets("Kendine sor", result.reading_prompts, "❓")

    with tab3:
        left, right = st.columns([0.7, 0.3], gap="large")
        with left:
            render_rubric_scores(result.rubric_scores)
        with right:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Kısa özet</div>", unsafe_allow_html=True)
            ordered = sorted(result.rubric_scores.items(), key=lambda x: x[1], reverse=True)
            for key, value in ordered[:5]:
                color = score_color(value)
                st.markdown(
                    f"<div class='ghost-badge' style='display:flex; justify-content:space-between; border-color:{color}; margin-bottom:.45rem; width:100%;'><span>{RUBRIC_LABELS[key]}</span><span>{value:.1f}</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                "<div class='mini-note'>En üstteki alanlar şu an fotoğrafın en çok çalışan taraflarını gösteriyor.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown(
            f"""
            <div class="panel-card">
                <div class="section-title">En uygun şema</div>
                <div class="scheme-name">{best_scheme}</div>
                <div class="mini-note">{scheme_reason}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        g1, g2, g3 = st.columns(3, gap="large")
        with g1:
            st.markdown("<div class='panel-title'>Altın oran ızgarası</div>", unsafe_allow_html=True)
            st.image(phi_grid_img, use_container_width=True)
            make_download_button(phi_grid_img, "Izgarayı indir", "cofsat_phi_grid.png", "dl_phi")
        with g2:
            st.markdown("<div class='panel-title'>Altın diyagonaller</div>", unsafe_allow_html=True)
            st.image(diagonal_img, use_container_width=True)
            make_download_button(diagonal_img, "Diyagonalleri indir", "cofsat_diagonal.png", "dl_diag")
        with g3:
            st.markdown("<div class='panel-title'>Altın spiral</div>", unsafe_allow_html=True)
            st.image(spiral_img, use_container_width=True)
            make_download_button(spiral_img, "Spirali indir", "cofsat_spiral.png", "dl_spiral")

    with tab5:
        metrics = result.metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Boyut", f"{metrics['width']} × {metrics['height']}")
            st.metric("Parlaklık ort.", f"{metrics['brightness_mean']:.1f}")
            st.metric("Kontrast std", f"{metrics['contrast_std']:.1f}")
        with m2:
            st.metric("Edge yoğunluğu", f"{metrics['edge_density']:.3f}")
            st.metric("Highlight clip", f"%{metrics['highlight_clip_ratio']*100:.2f}")
            st.metric("Shadow clip", f"%{metrics['shadow_clip_ratio']*100:.2f}")
        with m3:
            st.metric("Simetri", f"{metrics['symmetry_score']*100:.1f}")
            st.metric("Negatif alan", f"{metrics['negative_space_score']*100:.1f}")
            st.metric("Dinamik gerilim", f"{metrics['dynamic_tension_score']*100:.1f}")

        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Ham metrikler</div>", unsafe_allow_html=True)
        st.json(metrics)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer-note">
            ÇOFSAT ön değerlendirme sistemi, hızlı karar vermek için görsel okumayı destekler.
            Son yargı yine fotoğrafçının niyeti, bağlamı ve seçki içindeki yeriyle verilir.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
