import io
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageStat

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

# ============================================================
# ÇOFSAT Fotoğraf Ön Değerlendirme
# ============================================================

CULTURE = {
    "ad": "ÇOFSAT",
    "amac": "Fotoğrafı yüzeysel beğeni nesnesi olmaktan çıkarıp, düşünsel, estetik ve anlatı yapısıyla okumak.",
    "temel_soru": "Bu fotoğraf neden var?",
    "okuma_sorulari": [
        "İlk bakışta beni durduran ne?",
        "Gözüm kadrajda nereye gidiyor?",
        "Fotoğraf sessiz mi, gergin mi, hareketli mi?",
        "Bu görüntü neyi gösteriyor ve neyi dışarıda bırakıyor?",
        "Bu fotoğraf izleyicide ne uyandırıyor?",
    ],
    "ilkeler": [
        "Görmek yetmez; görüntü düşünceye dönüşmelidir.",
        "Eleştiri kişiye değil fotoğrafa yönelir.",
        "Niyet, tesadüften değerlidir.",
        "Kadraj sade ve bilinçli olmalıdır.",
        "Işık ve ton, gösteriş için değil anlam için kullanılmalıdır.",
        "Görsel gürültü anlatıyı zayıflatır.",
        "Fotoğraf hızlı tüketilen içerik değil, okunacak bir yapıdır.",
        "Teknik kusur bazen tolere edilir; anlamsızlık tolere edilmez.",
        "Fotoğrafın dili, etkisinden önce gelir.",
        "Beğenilmekten önce anlaşılmak önemlidir.",
    ],
    "rubric": {
        "ilk_etki_ve_gorsel_cagri": 0.10,
        "teknik_okuma": 0.14,
        "kompozisyon_ve_grafik_yapi": 0.17,
        "kadraj_hiyerarsisi_ve_odak": 0.14,
        "anlati_ve_duygu": 0.16,
        "soyutlama_ve_gorsel_dil": 0.09,
        "sadelesme_ve_disarida_birakma": 0.10,
        "niyet_ve_tutarlilik": 0.10,
    },
}

RUBRIC_LABELS = {
    "ilk_etki_ve_gorsel_cagri": "İlk Etki ve Görsel Çağrı",
    "teknik_okuma": "Teknik Okuma",
    "kompozisyon_ve_grafik_yapi": "Kompozisyon ve Grafik Yapı",
    "kadraj_hiyerarsisi_ve_odak": "Kadraj Hiyerarşisi ve Odak",
    "anlati_ve_duygu": "Anlatı ve Duygu",
    "soyutlama_ve_gorsel_dil": "Soyutlama ve Görsel Dil",
    "sadelesme_ve_disarida_birakma": "Sadeleşme ve Dışarıda Bırakma",
    "niyet_ve_tutarlilik": "Niyet ve Tutarlılık",
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


@dataclass
class CritiqueResult:
    total_score: float
    rubric_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    critique_short: str
    critique_long: str
    editing_suggestions: List[str]
    reading_prompts: List[str]
    metrics: Dict


def find_logo_file() -> Optional[str]:
    candidates = ["logo.png", "logo.jpg", "logo.jpeg", "2.jpeg", "2.jpg"]
    for name in candidates:
        if Path(name).exists():
            return name
    return None


def load_image_from_upload(uploaded_file) -> Image.Image:
    return Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")


def pil_to_gray_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def estimate_focus_score(gray: np.ndarray) -> float:
    if cv2 is None:
        gy, gx = np.gradient(gray.astype(np.float32))
        return float(np.var(gx) + np.var(gy))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def estimate_edge_density(gray: np.ndarray) -> float:
    if cv2 is None:
        gy, gx = np.gradient(gray.astype(np.float32))
        mag = np.sqrt(gx ** 2 + gy ** 2)
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
    local_activity = np.sqrt(gx ** 2 + gy ** 2)
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
    mag = np.sqrt(gx ** 2 + gy ** 2)
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


def extract_metrics(img: Image.Image) -> ImageMetrics:
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
        brightness_mean,
        brightness_std,
        highlight_clip_ratio,
        shadow_clip_ratio,
    )
    dynamic_tension_score = estimate_dynamic_tension(
        center_of_mass_x,
        center_of_mass_y,
        symmetry_score,
    )

    return ImageMetrics(
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
    )


def score_first_impact(metrics: ImageMetrics) -> float:
    focus = clamp01(1 - abs((math.log1p(metrics.focus_score) - 4.2) / 3.0))
    tension = metrics.dynamic_tension_score
    tonal = metrics.tonal_balance_score
    return clamp01(0.35 * focus + 0.35 * tension + 0.30 * tonal)


def score_technical(metrics: ImageMetrics) -> float:
    clip_penalty = min(1.0, (metrics.highlight_clip_ratio + metrics.shadow_clip_ratio) * 8)
    focus = clamp01(1 - abs((math.log1p(metrics.focus_score) - 4.2) / 3.0))
    return clamp01(0.4 * metrics.tonal_balance_score + 0.35 * focus + 0.25 * (1 - clip_penalty))


def score_composition(metrics: ImageMetrics) -> float:
    return clamp01(
        0.34 * metrics.thirds_alignment_score
        + 0.22 * metrics.dynamic_tension_score
        + 0.22 * metrics.symmetry_score
        + 0.22 * (1 - abs(metrics.negative_space_score - 0.45))
    )


def score_hierarchy(metrics: ImageMetrics) -> float:
    focus = clamp01(1 - abs((math.log1p(metrics.focus_score) - 4.2) / 3.0))
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


def build_rubric_scores(metrics: ImageMetrics) -> Dict[str, float]:
    return {
        "ilk_etki_ve_gorsel_cagri": score_first_impact(metrics),
        "teknik_okuma": score_technical(metrics),
        "kompozisyon_ve_grafik_yapi": score_composition(metrics),
        "kadraj_hiyerarsisi_ve_odak": score_hierarchy(metrics),
        "anlati_ve_duygu": score_narrative(metrics),
        "soyutlama_ve_gorsel_dil": score_abstraction(metrics),
        "sadelesme_ve_disarida_birakma": score_simplification(metrics),
        "niyet_ve_tutarlilik": score_intention(metrics),
    }


def weighted_total(scores: Dict[str, float]) -> float:
    total = sum(CULTURE["rubric"][key] * val for key, val in scores.items())
    return round(total * 100, 1)


def pick_strengths(scores: Dict[str, float]) -> List[str]:
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    mapping = {
        "ilk_etki_ve_gorsel_cagri": "İlk bakışta görsel bir çağrı kuruyor; fotoğraf izleyiciyi kapıdan içeri alabiliyor.",
        "teknik_okuma": "Teknik tercihler tamamen başıboş değil; ışık, ton ve netlik belli bir niyet taşıyor.",
        "kompozisyon_ve_grafik_yapi": "Kadrajın iskeleti dağılmıyor; çizgi, denge ve ritim belli ölçüde çalışıyor.",
        "kadraj_hiyerarsisi_ve_odak": "Gözün tutunacağı bir hiyerarşi kurulmuş; fotoğraf kendi içinde okunabiliyor.",
        "anlati_ve_duygu": "Görüntü yalnızca göstermiyor; izleyicide bir duygu alanı açıyor.",
        "soyutlama_ve_gorsel_dil": "Fotoğraf açık anlamdan çok görsel dil üzerinden çalışsa da tutarlılığını koruyor.",
        "sadelesme_ve_disarida_birakma": "Kare gereksiz yükten nispeten arınmış; dışarıda bırakma kararı fotoğrafa alan açıyor.",
        "niyet_ve_tutarlilik": "Fotoğraf tesadüfi görünmüyor; belli bir bakış ve niyet duygusu taşıyor.",
    }
    return [mapping[k] for k, _ in ordered[:3]]


def pick_weaknesses(scores: Dict[str, float]) -> List[str]:
    ordered = sorted(scores.items(), key=lambda x: x[1])
    mapping = {
        "ilk_etki_ve_gorsel_cagri": "İlk bakışta tutunacak kadar güçlü bir çağrı kuramıyor; fotoğraf izleyiciyi yeterince durdurmuyor.",
        "teknik_okuma": "Teknik tercihler anlamı taşımak yerine yer yer onu zayıflatıyor; ton ve ışık kararsız kalıyor.",
        "kompozisyon_ve_grafik_yapi": "Kadrajın grafik yapısı yeterince disiplinli değil; denge ve ritim dağınık görünüyor.",
        "kadraj_hiyerarsisi_ve_odak": "Fotoğraf neye bakmamızı istediğini tam söyleyemiyor; odak ve hiyerarşi zayıf.",
        "anlati_ve_duygu": "Görüntü bir anlam ihtimali taşısa da duygu ve anlatı düzeyinde tam açılmıyor.",
        "soyutlama_ve_gorsel_dil": "Soyut kalıyorsa bile güçlü bir görsel dil kuramıyor; iz bırakmak yerine belirsizleşiyor.",
        "sadelesme_ve_disarida_birakma": "Görsel gürültü fazla; dışarıda bırakılması gereken şeyler kadraj içinde kalmış.",
        "niyet_ve_tutarlilik": "Fotoğraf neden var sorusuna net bir karşılık veremiyor; niyet yeterince görünür değil.",
    }
    return [mapping[k] for k, _ in ordered[:3]]


def build_editing_suggestions(metrics: ImageMetrics, scores: Dict[str, float]) -> List[str]:
    suggestions = []
    if metrics.highlight_clip_ratio > 0.03:
        suggestions.append("Patlayan parlak alanları geri çekin; vurgu anlamı taşımalı, dikkat dağıtmamalı.")
    if metrics.shadow_clip_ratio > 0.08:
        suggestions.append("Gölgeleri tamamen boğmayın; siyahı derinleştirirken bilgi kaybını azaltın.")
    if scores["kadraj_hiyerarsisi_ve_odak"] < 0.50:
        suggestions.append("Kadrajın hiyerarşisini güçlendirin; ana ağırlık merkezini daha belirgin hale getirin.")
    if scores["kompozisyon_ve_grafik_yapi"] < 0.50:
        suggestions.append("Çizgileri, boşlukları ve görsel ritmi daha bilinçli düzenleyin; iskeleti sadeleştirin.")
    if scores["sadelesme_ve_disarida_birakma"] < 0.50:
        suggestions.append("Kadrajdan neyi çıkaracağınızı düşünün; dışarıda bırakmak fotoğrafı güçlendirebilir.")
    if scores["anlati_ve_duygu"] < 0.50:
        suggestions.append("Fotoğrafın ne hissettirdiğini netleştirin; teknik etkiyi duygu ve anlamın önüne geçirmeyin.")
    if not suggestions:
        suggestions.append("Bu kareyi geliştirecek şey filtre değil; kararların daha bilinçli ve daha tutarlı hale gelmesi.")
    return suggestions[:4]


def build_reading_prompts(scores: Dict[str, float]) -> List[str]:
    ordered = sorted(scores.items(), key=lambda x: x[1])
    prompts_by_key = {
        "ilk_etki_ve_gorsel_cagri": "İlk bakışta beni gerçekten durduran şey ne, yoksa kare akıp mı gidiyor?",
        "teknik_okuma": "Işık, ton ve netlik burada anlamı mı taşıyor, yoksa sadece efekt mi üretiyor?",
        "kompozisyon_ve_grafik_yapi": "Çizgiler, tekrarlar ve boşluklar fotoğrafın düşüncesine hizmet ediyor mu?",
        "kadraj_hiyerarsisi_ve_odak": "Gözüm nereye gidiyor ve orada kalmak için yeterli sebep buluyor mu?",
        "anlati_ve_duygu": "Bu görüntü bana ne hissettiriyor ve bu his yalnızca bana mı ait?",
        "soyutlama_ve_gorsel_dil": "Fotoğraf açık değilse bile kendi görsel dilinde tutarlı mı?",
        "sadelesme_ve_disarida_birakma": "Bu karede dışarıda bırakılması gereken ne var?",
        "niyet_ve_tutarlilik": "Bu fotoğraf neden var ve bunu kendi yapısıyla hissettirebiliyor mu?",
    }
    return [prompts_by_key[k] for k, _ in ordered[:3]]


def build_short_critique(total: float, weaknesses: List[str]) -> str:
    level = "zayıf" if total < 45 else "orta" if total < 65 else "güçlü" if total < 80 else "çok güçlü"
    return f"ÇOFSAT açısından bu kare {level} bir okuma alanı açıyor. En kritik kırılma şu: {weaknesses[0].lower()}"


def build_long_critique(strengths: List[str], weaknesses: List[str], total: float) -> str:
    return (
        f"Bu fotoğraf ÇOFSAT manifestosu ve fotoğraf okuma kılavuzuna göre {total}/100 düzeyinde okunuyor. "
        f"Yani görüntü yalnızca ne gösterdiğiyle değil, nasıl kurulduğuyla da değerlendiriliyor. "
        f"Güçlü taraflarında şunlar öne çıkıyor: {strengths[0]} {strengths[1]} "
        f"Buna karşılık temel kırılmalar şunlar: {weaknesses[0]} {weaknesses[1]} {weaknesses[2]} "
        f"Bu sistem doğruyu bulmaya değil, görmeyi derinleştirmeye çalışır. Dolayısıyla mesele yalnızca teknik doğruluk değil; "
        f"kadrajın neyi dahil edip neyi dışarıda bıraktığı, gözün nerede durduğu, görüntünün sessiz mi gergin mi hareketli mi okunduğu ve "
        f"fotoğrafın izleyicide düşünsel ya da duygusal bir iz bırakıp bırakmadığıdır. Son soru yine aynıdır: '{CULTURE['temel_soru']}'"
    )


def critique_image(img: Image.Image) -> CritiqueResult:
    metrics = extract_metrics(img)
    scores = build_rubric_scores(metrics)
    total = weighted_total(scores)
    strengths = pick_strengths(scores)
    weaknesses = pick_weaknesses(scores)
    return CritiqueResult(
        total_score=total,
        rubric_scores={k: round(v * 100, 1) for k, v in scores.items()},
        strengths=strengths,
        weaknesses=weaknesses,
        critique_short=build_short_critique(total, weaknesses),
        critique_long=build_long_critique(strengths, weaknesses, total),
        editing_suggestions=build_editing_suggestions(metrics, scores),
        reading_prompts=build_reading_prompts(scores),
        metrics=asdict(metrics),
    )


def score_label(score: float) -> str:
    if score < 45:
        return "Zayıf"
    if score < 65:
        return "Orta"
    if score < 80:
        return "Güçlü"
    return "Çok Güçlü"


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(212,175,55,0.08), transparent 28%),
            radial-gradient(circle at bottom right, rgba(212,175,55,0.05), transparent 24%),
            linear-gradient(180deg, #050505 0%, #0d0d0d 100%);
        color: #f5f5f5;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
        animation: fadeInUp .65s ease;
    }

    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translateY(14px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    section[data-testid="stSidebar"] {
        background: #0a0a0a;
        border-right: 1px solid rgba(212,175,55,.16);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }

    .hero {
        padding: 1.8rem;
        border: 1px solid rgba(212,175,55,.18);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(255,255,255,.04), rgba(212,175,55,.08));
        box-shadow: 0 14px 40px rgba(0,0,0,.35);
        margin-bottom: 1rem;
        transition: transform .28s ease, box-shadow .28s ease, border-color .28s ease;
        animation: fadeInUp .7s ease;
    }

    .hero:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 48px rgba(0,0,0,.42);
        border-color: rgba(212,175,55,.28);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.8rem;
        line-height: 1.05;
        color: #fff7dc;
        letter-spacing: 0.2px;
    }

    .hero p {
        margin-top: .8rem;
        font-size: 1.06rem;
        color: rgba(255,255,255,.97);
        line-height: 1.65;
    }

    .hero-badge {
        display: inline-block;
        margin-top: .9rem;
        padding: .45rem .85rem;
        border-radius: 999px;
        border: 1px solid rgba(212,175,55,.20);
        background: rgba(212,175,55,.08);
        color: #f0d782;
        font-size: .92rem;
        font-weight: 600;
        transition: all .25s ease;
    }

    .hero-badge:hover {
        background: rgba(212,175,55,.14);
        border-color: rgba(212,175,55,.35);
        color: #fff1b0;
    }

    .soft-card {
        border: 1px solid rgba(212,175,55,.12);
        border-radius: 18px;
        padding: 1rem 1rem .9rem 1rem;
        background: rgba(255,255,255,.03);
        margin-bottom: .9rem;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
        animation: fadeInUp .72s ease;
    }

    .soft-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0,0,0,.24);
        border-color: rgba(212,175,55,.22);
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: .45rem;
        color: #f1d67a;
    }

    .mini-note {
        font-size: .97rem;
        line-height: 1.7;
        color: rgba(255,255,255,.98) !important;
        opacity: 1 !important;
    }

    .score-box {
        border: 1px solid rgba(212,175,55,.14);
        border-radius: 18px;
        padding: 1rem 1rem .9rem 1rem;
        background: rgba(255,255,255,.03);
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
        animation: fadeInUp .74s ease;
    }

    .score-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0,0,0,.24);
        border-color: rgba(212,175,55,.22);
    }

    .quote-box {
        border-left: 4px solid rgba(212,175,55,.92);
        padding: .95rem 1rem;
        background: rgba(212,175,55,.08);
        border-radius: 0 14px 14px 0;
        margin: .5rem 0 1rem 0;
        color: #fff3cf;
        transition: all .25s ease;
        animation: fadeInUp .76s ease;
    }

    .quote-box:hover {
        background: rgba(212,175,55,.12);
        border-left-color: rgba(255,220,120,.95);
    }

    .upload-label {
        font-size: 1rem;
        font-weight: 700;
        color: #f1d67a;
        margin-bottom: .35rem;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,.03);
        border: 1px solid rgba(212,175,55,.12);
        border-radius: 18px;
        padding: .35rem .7rem .6rem .7rem;
        transition: all .25s ease;
        animation: fadeInUp .78s ease;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(212,175,55,.26);
        box-shadow: 0 10px 26px rgba(0,0,0,.18);
        transform: translateY(-1px);
    }

    div[data-testid="stFileUploader"] section {
        border: 1px dashed rgba(212,175,55,.22) !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,.02);
        transition: all .25s ease;
    }

    div[data-testid="stFileUploader"] section:hover {
        border-color: rgba(255,220,120,.35) !important;
        background: rgba(255,255,255,.03);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.03);
        border: 1px solid rgba(212,175,55,.10);
        padding: .85rem;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,.16);
        transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
        animation: fadeInUp .8s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(0,0,0,.22);
        border-color: rgba(212,175,55,.24);
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.55rem !important;
        opacity: 1 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #ffffff !important;
        opacity: .92 !important;
    }

    div[data-testid="stProgressBar"] > div {
        border-radius: 999px;
        overflow: hidden;
    }

    details {
        background: rgba(255,255,255,.03);
        border: 1px solid rgba(212,175,55,.10);
        border-radius: 14px;
        padding: .4rem .7rem;
        transition: all .25s ease;
    }

    details:hover {
        border-color: rgba(212,175,55,.24);
        box-shadow: 0 8px 20px rgba(0,0,0,.18);
    }

    h2, h3, h4 {
        color: #fff2c2;
        letter-spacing: 0.2px;
    }

    p, li, label {
        color: rgba(255,255,255,.96) !important;
    }

    hr {
        border: none;
        border-top: 1px solid rgba(212,175,55,.12);
        margin: 1.6rem 0;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        color: rgba(255,255,255,.97) !important;
    }

    div[data-testid="stInfo"] {
        background: rgba(255,255,255,.04) !important;
        border: 1px solid rgba(212,175,55,.12) !important;
        color: #ffffff !important;
        border-radius: 14px !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 14px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, rgba(212,175,55,.18), rgba(212,175,55,.08));
        color: #fff8dc;
        border: 1px solid rgba(212,175,55,.28);
        border-radius: 14px;
        padding: 0.65rem 1rem;
        font-weight: 700;
        transition: all .25s ease;
        box-shadow: 0 8px 18px rgba(0,0,0,.16);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, rgba(212,175,55,.26), rgba(212,175,55,.12));
        border-color: rgba(255,220,120,.45);
        box-shadow: 0 12px 26px rgba(0,0,0,.24);
        color: #ffffff;
    }

    .stButton > button:focus,
    .stButton > button:active {
        outline: none !important;
        box-shadow: 0 0 0 0.15rem rgba(212,175,55,.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

logo_file = find_logo_file()

with st.sidebar:
    if logo_file:
        st.image(logo_file, use_container_width=True)
    st.markdown("## ÇOFSAT")
    st.markdown("### Temel soru")
    st.warning(CULTURE["temel_soru"])
    st.markdown("### Fotoğrafa yaklaşım")
    for q in CULTURE["okuma_sorulari"][:3]:
        st.markdown(f"- {q}")
    st.markdown(
        "<div class='mini-note'>Bu sistem sahnenin niyetini doğrudan bilemez; teknik ve yapısal izlerden yaklaşık bir okuma üretir.</div>",
        unsafe_allow_html=True,
    )

header_left, header_right = st.columns([0.18, 0.82])

with header_left:
    if logo_file:
        st.image(logo_file, use_container_width=True)

with header_right:
    st.markdown(
        """
        <div class="hero">
            <h1>ÇOFSAT Fotoğraf Ön Değerlendirme</h1>
            <p>
                Fotoğrafı yalnızca göstermek için değil, okumak için ele alan
                ÇOFSAT temelli ön değerlendirme sistemi.
            </p>
            <div class="hero-badge">Kadraj · Niyet · Anlatı · Sadelik · Görsel Dil</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

col_intro_1, col_intro_2 = st.columns([1.15, 1])

with col_intro_1:
    st.markdown(
        """
        <div class="soft-card">
            <div class="section-title">Bu sistem ne yapar?</div>
            <div class="mini-note">
                Fotoğrafı teknik açıdan ölçer; sonra bunu ÇOFSAT’ın niyet, kadraj,
                anlatı, sadelik ve görsel dil anlayışıyla yorumlar.
                Amaç hüküm vermek değil, görmeyi derinleştirmektir.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_intro_2:
    st.markdown(
        f"""
        <div class="quote-box">
            <strong>Manifesto sorusu:</strong><br>{CULTURE['temel_soru']}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='upload-label'>Fotoğraf yükleyin</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
    help="JPG, JPEG, PNG, WEBP, TIF ve TIFF desteklenir.",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = load_image_from_upload(uploaded_file)
    result = critique_image(image)

    col1, col2 = st.columns([1.05, 0.95])

    with col1:
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with col2:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.metric("ÇOFSAT Skoru", f"{result.total_score}/100", score_label(result.total_score))
        st.progress(result.total_score / 100)
        st.markdown("#### Kısa okuma")
        st.info(result.critique_short)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Derin Okuma")
    st.write(result.critique_long)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Güçlü yanlar")
        for item in result.strengths:
            st.markdown(f"- {item}")

    with c2:
        st.markdown("### Zayıf yanlar")
        for item in result.weaknesses:
            st.markdown(f"- {item}")

    st.markdown("---")
    st.markdown("## Değerlendirme Kategorileri")

    score_items = list(result.rubric_scores.items())
    row1 = st.columns(4)
    for i, (key, val) in enumerate(score_items[:4]):
        row1[i].metric(RUBRIC_LABELS[key], f"{val}")

    row2 = st.columns(4)
    for i, (key, val) in enumerate(score_items[4:]):
        row2[i].metric(RUBRIC_LABELS[key], f"{val}")

    st.markdown("---")
    q1, q2 = st.columns(2)

    with q1:
        st.markdown("## Okumayı derinleştiren sorular")
        for item in result.reading_prompts:
            st.markdown(f"- {item}")

    with q2:
        st.markdown("## Düzenleme önerileri")
        for item in result.editing_suggestions:
            st.markdown(f"- {item}")

    with st.expander("Teknik metrikler"):
        st.json(result.metrics)

else:
    st.markdown(
        """
        <div class="soft-card">
            <div class="section-title">Başlamak için bir fotoğraf yükleyin</div>
            <div class="mini-note">
                Fotoğrafı yüklediğiniz anda sistem önce görsel yapıyı ölçer,
                sonra ÇOFSAT diline yakın bir okuma üretir.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
