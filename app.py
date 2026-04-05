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
    "amac": "Fotoğrafı yalnızca beğeni nesnesi olmaktan çıkarıp, niyet, kadraj, anlatı ve görsel dil üzerinden okumak.",
    "temel_soru": "Bu fotoğraf neden var?",
    "okuma_sorulari": [
        "İlk bakışta beni durduran ne?",
        "Gözüm kadrajda nereye gidiyor?",
        "Fotoğraf sessiz mi, gergin mi, hareketli mi?",
        "Bu görüntü neyi gösteriyor ve neyi dışarıda bırakıyor?",
        "Bu fotoğraf bende nasıl bir his bırakıyor?",
    ],
    "ilkeler": [
        "Görmek yetmez; görüntü düşünceye dönüşmelidir.",
        "Eleştiri kişiye değil fotoğrafa yönelir.",
        "Niyet, tesadüften değerlidir.",
        "Kadraj sade ve bilinçli olmalıdır.",
        "Işık ve ton gösteriş için değil anlam için kullanılmalıdır.",
        "Görsel gürültü anlatıyı zayıflatır.",
        "Fotoğraf hızlı tüketilen içerik değil, okunacak bir yapıdır.",
        "Teknik kusur bazen tolere edilir; anlamsızlık tolere edilmez.",
        "Fotoğrafın dili, etkisinden önce gelir.",
        "Beğenilmekten önce anlaşılmak önemlidir.",
    ],
    "rubric": {
        "ilk_etki": 0.10,
        "teknik_butunluk": 0.14,
        "kompozisyon": 0.17,
        "odak_ve_hiyerarsi": 0.14,
        "anlati_gucu": 0.16,
        "gorsel_dil": 0.09,
        "sadelik": 0.10,
        "niyet_tutarliligi": 0.10,
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


def normalize_focus(metrics: ImageMetrics) -> float:
    return clamp01(1 - abs((math.log1p(metrics.focus_score) - 4.2) / 3.0))


def mode_adjustment(scores: Dict[str, float], mode: str) -> Dict[str, float]:
    adjusted = scores.copy()

    if mode == "Sokak":
        adjusted["anlati_gucu"] = clamp01(adjusted["anlati_gucu"] * 1.08)
        adjusted["ilk_etki"] = clamp01(adjusted["ilk_etki"] * 1.05)
        adjusted["sadelik"] = clamp01(adjusted["sadelik"] * 0.98)

    elif mode == "Portre":
        adjusted["odak_ve_hiyerarsi"] = clamp01(adjusted["odak_ve_hiyerarsi"] * 1.08)
        adjusted["anlati_gucu"] = clamp01(adjusted["anlati_gucu"] * 1.06)
        adjusted["kompozisyon"] = clamp01(adjusted["kompozisyon"] * 1.02)

    elif mode == "Belgesel":
        adjusted["niyet_tutarliligi"] = clamp01(adjusted["niyet_tutarliligi"] * 1.08)
        adjusted["anlati_gucu"] = clamp01(adjusted["anlati_gucu"] * 1.06)
        adjusted["teknik_butunluk"] = clamp01(adjusted["teknik_butunluk"] * 0.98)

    elif mode == "Soyut":
        adjusted["gorsel_dil"] = clamp01(adjusted["gorsel_dil"] * 1.12)
        adjusted["kompozisyon"] = clamp01(adjusted["kompozisyon"] * 1.06)
        adjusted["sadelik"] = clamp01(adjusted["sadelik"] * 1.05)
        adjusted["anlati_gucu"] = clamp01(adjusted["anlati_gucu"] * 0.95)

    return adjusted


def score_first_impact(metrics: ImageMetrics) -> float:
    focus = normalize_focus(metrics)
    tension = metrics.dynamic_tension_score
    tonal = metrics.tonal_balance_score
    return clamp01(0.35 * focus + 0.35 * tension + 0.30 * tonal)


def score_technical(metrics: ImageMetrics) -> float:
    clip_penalty = min(1.0, (metrics.highlight_clip_ratio + metrics.shadow_clip_ratio) * 8)
    focus = normalize_focus(metrics)
    return clamp01(0.4 * metrics.tonal_balance_score + 0.35 * focus + 0.25 * (1 - clip_penalty))


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


def overall_tag_from_scores(scores_100: Dict[str, float], mode: str) -> str:
    if mode == "Sokak":
        if scores_100["anlati_gucu"] >= 75:
            return "Sahne duygusu güçlü"
        if scores_100["odak_ve_hiyerarsi"] < 60:
            return "An var, vurgu zayıf"
        return "Sokak potansiyeli yüksek"

    if mode == "Portre":
        if scores_100["odak_ve_hiyerarsi"] >= 75:
            return "Yüz ve ifade okunuyor"
        if scores_100["anlati_gucu"] >= 75:
            return "Duygusal temas güçlü"
        return "Portre bağı kuruluyor"

    if mode == "Belgesel":
        if scores_100["niyet_tutarliligi"] >= 75:
            return "Tanıklık hissi güçlü"
        if scores_100["anlati_gucu"] >= 70:
            return "Bağlam taşıyor"
        return "Belgesel değeri var"

    if mode == "Soyut":
        if scores_100["gorsel_dil"] >= 75:
            return "Görsel dili güçlü"
        if scores_100["kompozisyon"] >= 75:
            return "Biçimsel olarak etkili"
        return "Soyut potansiyel taşıyor"

    return "Potansiyeli olan kare"


def pick_strengths(scores_100: Dict[str, float], mode: str) -> List[str]:
    ordered = sorted(scores_100.items(), key=lambda x: x[1], reverse=True)
    common = {
        "ilk_etki": "Fotoğraf ilk bakışta kendine alan açabiliyor; izleyiciyi tamamen dışarıda bırakmıyor.",
        "teknik_butunluk": "Işık, ton ve netlik tamamen rastlantısal görünmüyor; teknik yapı anlatıyı belli ölçüde taşıyabiliyor.",
        "kompozisyon": "Kadrajın iskeleti dağılmıyor; çizgi, denge ve ritim birbirini destekliyor.",
        "odak_ve_hiyerarsi": "Gözün tutunacağı yer büyük ölçüde belli; fotoğraf neye bakmamızı istediğini söyleyebiliyor.",
        "anlati_gucu": "Kare yalnızca bir görüntü değil; küçük de olsa bir duygu ve anlatı alanı açıyor.",
        "gorsel_dil": "Fotoğraf biçimsel olarak kendi dilini kurmaya çalışıyor ve bu çaba hissediliyor.",
        "sadelik": "Gereksiz yük nispeten geri çekilmiş; fotoğraf nefes alabiliyor.",
        "niyet_tutarliligi": "Kare tesadüfi görünmüyor; arkasında belli bir niyet ve karar hissi var.",
    }

    mode_extras = {
        "Sokak": {
            "anlati_gucu": "Sahnenin içindeki an hissi korunmuş; kare yaşanmış bir karşılaşma duygusu taşıyor.",
            "kompozisyon": "Sahne içindeki katmanlar ve şehir akışı fotoğrafın enerjisini destekliyor.",
        },
        "Portre": {
            "odak_ve_hiyerarsi": "Yüz, bakış veya beden dili izleyiciyle bağ kurabiliyor.",
            "anlati_gucu": "Portrede yalnızca görünüş değil, bir ruh hali de hissediliyor.",
        },
        "Belgesel": {
            "niyet_tutarliligi": "Fotoğraf yalnızca estetik değil; tanıklık ve bağlam duygusu da taşıyor.",
            "anlati_gucu": "Kare, bir durumun ya da gerçeğin izini taşımaya çalışıyor.",
        },
        "Soyut": {
            "gorsel_dil": "Biçim, yüzey, ton ve ritim kendi başına bir görsel alan kurabiliyor.",
            "kompozisyon": "Soyut okuma için gerekli biçimsel denge büyük ölçüde kurulmuş.",
        },
    }

    mapping = common.copy()
    mapping.update(mode_extras.get(mode, {}))
    return [mapping[k] for k, _ in ordered[:3]]


def pick_development_areas(scores_100: Dict[str, float], mode: str) -> List[str]:
    ordered = sorted(scores_100.items(), key=lambda x: x[1])
    common = {
        "ilk_etki": "İlk bakışta izleyiciyi durduracak kadar güçlü bir giriş henüz tam kurulamıyor.",
        "teknik_butunluk": "Ton, ışık veya netlik düzeyi anlamı desteklemek yerine yer yer onu zayıflatıyor.",
        "kompozisyon": "Kadrajın grafik yapısı daha bilinçli kurulursa fotoğraf çok daha güçlü okunabilir.",
        "odak_ve_hiyerarsi": "Ana ağırlık merkezi biraz daha netleşirse göz fotoğraf içinde daha kararlı dolaşır.",
        "anlati_gucu": "Fotoğrafın his ve anlatı tarafı var ama henüz tam açılmamış; biraz daha net bir yön istiyor.",
        "gorsel_dil": "Biçimsel tercihlerin ortak bir dile dönüşmesi için birkaç kararın daha netleşmesi gerekiyor.",
        "sadelik": "Kadrajdan bazı şeyleri çıkarmak ya da geri itmek fotoğrafın anlatısını rahatlatabilir.",
        "niyet_tutarliligi": "Fotoğrafın neden var olduğu biraz daha görünür hale gelirse etkisi belirginleşir.",
    }

    mode_extras = {
        "Sokak": {
            "anlati_gucu": "Sokak fotoğrafında an hissi biraz daha belirginleşirse kare daha canlı bir enerji kazanabilir.",
            "odak_ve_hiyerarsi": "Sahnenin içindeki ana karşılaşma veya jest biraz daha net seçilirse okuma güçlenir.",
        },
        "Portre": {
            "odak_ve_hiyerarsi": "Portrede yüz ve ifade biraz daha okunur hale gelirse bağ çok daha doğrudan kurulur.",
            "sadelik": "Arka planı biraz daha sakinleştirmek portreyi nefes aldırabilir.",
        },
        "Belgesel": {
            "niyet_tutarliligi": "Bağlam biraz daha görünür hale gelirse fotoğrafın tanıklık gücü artabilir.",
            "teknik_butunluk": "Belgesel karede teknik müdahale hissi değil, sahnenin açıklığı daha önemli olabilir.",
        },
        "Soyut": {
            "gorsel_dil": "Soyut yapıda biçimsel kararlar biraz daha netleşirse fotoğraf çok daha bütünlüklü görünür.",
            "kompozisyon": "Ritim ve boşluk ilişkisi daha kararlı kurulursa soyut etki büyür.",
        },
    }

    mapping = common.copy()
    mapping.update(mode_extras.get(mode, {}))
    return [mapping[k] for k, _ in ordered[:3]]


def build_reading_prompts(scores_100: Dict[str, float], mode: str) -> List[str]:
    base = {
        "ilk_etki": "Bu kare ilk anda beni gerçekten durduruyor mu, yoksa akıp gidiyor mu?",
        "teknik_butunluk": "Işık ve ton burada anlamı mı taşıyor, yoksa sadece etki mi üretiyor?",
        "kompozisyon": "Çizgiler, boşluklar ve kadraj dengesi fotoğrafın düşüncesine hizmet ediyor mu?",
        "odak_ve_hiyerarsi": "Gözüm nereye gidiyor ve orada kalmak için yeterli sebep buluyor mu?",
        "anlati_gucu": "Bu görüntü bana ne hissettiriyor ve bunu hangi yapısal tercihler kuruyor?",
        "gorsel_dil": "Fotoğraf kendi dili içinde tutarlı mı, yoksa parçalar ayrı ayrı mı çalışıyor?",
        "sadelik": "Bu karede dışarıda bırakılması gereken şey ne olabilir?",
        "niyet_tutarliligi": "Bu fotoğraf neden var ve bunu yapısıyla hissettirebiliyor mu?",
    }

    mode_prompts = {
        "Sokak": {
            "anlati_gucu": "Bu karede gerçekten yaşanmış bir an var mı, yoksa yalnızca bir görüntü mü görüyorum?",
            "kompozisyon": "Sahne içindeki katmanlar birbiriyle konuşuyor mu?",
        },
        "Portre": {
            "odak_ve_hiyerarsi": "Yüz, bakış ve beden dili arasında en güçlü bağ nerede kuruluyor?",
            "anlati_gucu": "Bu portre bana kişi hakkında ne hissettiriyor?",
        },
        "Belgesel": {
            "niyet_tutarliligi": "Bu kare yalnızca güzel mi, yoksa bir duruma tanıklık ediyor mu?",
            "anlati_gucu": "Bağlam fotoğrafta yeterince hissediliyor mu?",
        },
        "Soyut": {
            "gorsel_dil": "Bu karede anlamı biçim mi taşıyor, yoksa görüntü yalnızca karmaşık mı kalıyor?",
            "kompozisyon": "Ton, yüzey, tekrar ve boşluk arasında bir ritim var mı?",
        },
    }

    merged = base.copy()
    merged.update(mode_prompts.get(mode, {}))
    ordered = sorted(scores_100.items(), key=lambda x: x[1])
    return [merged[k] for k, _ in ordered[:3]]


def build_shooting_notes(metrics: ImageMetrics, scores_100: Dict[str, float], mode: str) -> List[str]:
    notes: List[str] = []

    if scores_100["odak_ve_hiyerarsi"] < 60:
        notes.append("Çekim anında ana öznenin görsel ağırlığını biraz daha net kurmak kareyi belirgin biçimde güçlendirebilir.")
    if scores_100["sadelik"] < 60:
        notes.append("Kadrajı sadeleştirip dikkat dağıtan alanları azaltmak fotoğrafın nefesini açar.")
    if scores_100["kompozisyon"] < 60:
        notes.append("Boşluklar ve çizgiler daha bilinçli kullanılırsa göz akışı çok daha güçlü hale gelir.")
    if metrics.highlight_clip_ratio > 0.03:
        notes.append("Parlak alanların patlamaması için çekimde ışığı biraz daha kontrollü karşılamak faydalı olabilir.")

    if mode == "Sokak":
        notes.append("Sokak karelerinde yarım saniyelik bir zamanlama farkı bazen bütün anlatıyı değiştirebilir; anı biraz daha beklemek işe yarayabilir.")
    elif mode == "Portre":
        notes.append("Portrede özneyle kurulan küçük bir güven ve rahatlık hissi, teknik her şeyden daha güçlü bir karşılık verebilir.")
    elif mode == "Belgesel":
        notes.append("Belgesel karede bağlamı biraz daha görünür bırakmak fotoğrafın tanıklık gücünü artırabilir.")
    elif mode == "Soyut":
        notes.append("Soyut çekimlerde biçimsel sadelik ve tekrar ilişkisini biraz daha kararlı kurmak etkiyi büyütür.")

    # benzersiz ve kısa tut
    unique = []
    for n in notes:
        if n not in unique:
            unique.append(n)
    return unique[:4] if unique else ["Bu kare küçük kararlarla daha da güçlenebilir."]


def build_editing_notes(metrics: ImageMetrics, scores_100: Dict[str, float], mode: str) -> List[str]:
    notes: List[str] = []

    if metrics.highlight_clip_ratio > 0.03:
        notes.append("Highlight bölgelerini biraz geri çekmek, fotoğrafın dikkat dengesini daha kontrollü hale getirir.")
    if metrics.shadow_clip_ratio > 0.08:
        notes.append("Siyahları derinleştirirken bilgi kaybını azaltmak kareye daha rafine bir ton dengesi kazandırır.")
    if scores_100["odak_ve_hiyerarsi"] < 60:
        notes.append("Ana özne çevresinde lokal kontrast veya parlaklık desteği vermek gözün tutunmasını kolaylaştırır.")
    if scores_100["sadelik"] < 60:
        notes.append("Arka planı bir miktar bastırmak ya da orta tonları sadeleştirmek anlatıyı öne çıkarabilir.")

    if mode == "Sokak":
        notes.append("Sokak fotoğrafında aşırı temizlik yerine sahnenin canlılığını koruyan hafif düzenlemeler daha doğal sonuç verir.")
    elif mode == "Portre":
        notes.append("Portrede yüz tonlarını fazla parlatmadan, ifadeyi öne çıkaran yumuşak geçişler daha güçlü çalışır.")
    elif mode == "Belgesel":
        notes.append("Belgesel karede düzenleme hissinin öne çıkmaması, sahnenin inandırıcılığını korumak açısından değerlidir.")
    elif mode == "Soyut":
        notes.append("Soyut fotoğrafta tonları sadeleştirmek ve ritmi belirginleştirmek biçimsel etkiyi güçlendirebilir.")

    unique = []
    for n in notes:
        if n not in unique:
            unique.append(n)
    return unique[:4] if unique else ["Bu kare için aşırı filtre yerine küçük ve bilinçli ton düzenlemeleri en iyi sonucu verir."]


def build_first_reading(total: float, scores_100: Dict[str, float], mode: str) -> str:
    level = score_band(total).lower()

    if mode == "Sokak":
        if scores_100["ilk_etki"] >= 75:
            return f"İlk anda bu kare bir sokak karşılaşması duygusu kurabiliyor. Şehrin akışı içinde kendine yer açan, {level} düzeyin üzerinde bir ilk temas hissi var."
        return "İlk bakışta sahne hissediliyor; ancak sokak fotoğrafının vurucu an etkisi biraz daha güçlenirse kare daha akılda kalıcı olabilir."

    if mode == "Portre":
        if scores_100["anlati_gucu"] >= 75:
            return f"İlk anda portrenin duygusal alanı açılıyor. Yüz, bakış ya da beden dili izleyiciyle {level} düzeye yaklaşan bir bağ kurabiliyor."
        return "İlk anda özne görülüyor; fakat portreyle kurulan duygusal temas biraz daha derinleşirse kare daha etkili hale gelebilir."

    if mode == "Belgesel":
        if scores_100["niyet_tutarliligi"] >= 75:
            return f"İlk anda kare yalnızca estetik değil; bir bağlam ve tanıklık duygusu da taşıyor. Bu belgesel okuma için çok değerli bir başlangıç."
        return "İlk bakışta sahnenin bir anlamı olduğu hissediliyor; bağlam biraz daha belirginleşirse belgesel gücü artabilir."

    if mode == "Soyut":
        if scores_100["gorsel_dil"] >= 75:
            return "İlk anda görüntü nesne aratmıyor; biçim, ton ve yüzey kendi başına bir etki alanı kurabiliyor."
        return "İlk anda biçimsel bir ilgi doğuyor; görsel dil biraz daha netleşirse soyut etki çok daha güçlü olabilir."

    return "İlk bakışta fotoğraf bir niyet hissi taşıyor."


def build_structural_reading(metrics: ImageMetrics, scores_100: Dict[str, float], mode: str) -> str:
    parts = []

    if scores_100["kompozisyon"] >= 70:
        parts.append("kadrajın iskeleti genel olarak toparlanmış")
    else:
        parts.append("kadrajın iskeleti biraz daha disiplin isteyebilir")

    if scores_100["odak_ve_hiyerarsi"] >= 70:
        parts.append("gözün tutunacağı alan büyük ölçüde belli")
    else:
        parts.append("odak ve hiyerarşi biraz daha netleşirse okuma rahatlar")

    if scores_100["sadelik"] >= 65:
        parts.append("görsel yük büyük ölçüde kontrol altında")
    else:
        parts.append("bazı unsurlar ana anlatının önüne geçebiliyor")

    if metrics.tonal_balance_score >= 0.65:
        parts.append("ton dengesi fotoğrafa destek veriyor")
    else:
        parts.append("ton yapısı biraz daha rafine edilirse kare daha olgun görünür")

    if mode == "Sokak":
        ending = " Özellikle an, katman ve sahne akışı sokak okumasında belirleyici görünüyor."
    elif mode == "Portre":
        ending = " Özellikle yüzün okunurluğu ve özne-arka plan ilişkisi portre etkisini doğrudan belirliyor."
    elif mode == "Belgesel":
        ending = " Özellikle bağlamın görünürlüğü ve sahnenin dürüstlüğü belgesel değerini etkiliyor."
    else:
        ending = " Özellikle biçimsel ritim ve görsel dilin tutarlılığı soyut okumada daha fazla öne çıkıyor."

    return "Yapısal olarak bakıldığında " + ", ".join(parts) + "." + ending


def build_editorial_result(total: float, scores_100: Dict[str, float], mode: str) -> str:
    if total >= 80:
        return f"Genel sonuç olarak bu {mode.lower()} kare yalnızca doğru kararlar içermiyor; aynı zamanda kendi dilini hissettirebiliyor. Editöryel olarak güçlü bir zemini var."
    if total >= 65:
        return f"Genel sonuç olarak bu {mode.lower()} kare güçlü bir potansiyel taşıyor. Doğru yerlere küçük dokunuşlar gelirse etkisi belirgin biçimde artar."
    if total >= 45:
        return f"Genel sonuç olarak karede iyi bir niyet var. Şimdilik bazı kararlar tam yerine oturmamış olsa da üzerinde düşünülmüş bir yön hissediliyor."
    return f"Genel sonuç olarak bu {mode.lower()} kare henüz tam açılmamış görünüyor. Yine de burada önemli olan eksik değil; hangi kararların fotoğrafı ileri taşıyacağını görebilmektir."


def build_editor_summary(total: float, strengths: List[str], dev_areas: List[str], mode: str) -> str:
    level = score_band(total)
    return (
        f"Bu **{mode.lower()}** kare şu an için **{level}** bir potansiyel gösteriyor. "
        f"En güçlü taraflarından biri şu: {strengths[0].lower()} "
        f"Gelişime en açık yer ise şu görünüyor: {dev_areas[0].lower()}"
    )


def build_tags(scores_100: Dict[str, float], total: float, mode: str) -> List[str]:
    tags = [mode]

    if scores_100["anlati_gucu"] >= 75:
        tags.append("Güçlü duygu")
    if scores_100["kompozisyon"] >= 75:
        tags.append("Sağlam kompozisyon")
    if scores_100["teknik_butunluk"] >= 75:
        tags.append("Teknik bütünlük")
    if scores_100["sadelik"] < 55:
        tags.append("Görsel gürültü")
    if scores_100["odak_ve_hiyerarsi"] < 55:
        tags.append("Odak zayıf")
    if scores_100["niyet_tutarliligi"] >= 70:
        tags.append("Niyeti hissediliyor")
    if scores_100["gorsel_dil"] >= 75 and mode == "Soyut":
        tags.append("Biçimsel güç")
    if total >= 80:
        tags.append("Editoryal olarak güçlü")
    elif total >= 65:
        tags.append("Yüksek potansiyel")
    else:
        tags.append("Geliştirilebilir")

    unique = []
    for tag in tags:
        if tag not in unique:
            unique.append(tag)
    return unique[:5]


def critique_image(img: Image.Image, mode: str) -> CritiqueResult:
    metrics = extract_metrics(img)
    scores = build_rubric_scores(metrics, mode)
    total = weighted_total(scores)
    scores_100 = {k: round(v * 100, 1) for k, v in scores.items()}

    strengths = pick_strengths(scores_100, mode)
    dev_areas = pick_development_areas(scores_100, mode)

    return CritiqueResult(
        total_score=total,
        overall_level=score_band(total),
        overall_tag=overall_tag_from_scores(scores_100, mode),
        rubric_scores=scores_100,
        strengths=strengths,
        development_areas=dev_areas,
        editor_summary=build_editor_summary(total, strengths, dev_areas, mode),
        first_reading=build_first_reading(total, scores_100, mode),
        structural_reading=build_structural_reading(metrics, scores_100, mode),
        editorial_result=build_editorial_result(total, scores_100, mode),
        shooting_notes=build_shooting_notes(metrics, scores_100, mode),
        editing_notes=build_editing_notes(metrics, scores_100, mode),
        reading_prompts=build_reading_prompts(scores_100, mode),
        tags=build_tags(scores_100, total, mode),
        metrics=asdict(metrics),
    )


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
        0% { opacity: 0; transform: translateY(14px); }
        100% { opacity: 1; transform: translateY(0); }
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

    .soft-card {
        border: 1px solid rgba(212,175,55,.12);
        border-radius: 18px;
        padding: 1rem 1rem .9rem 1rem;
        background: rgba(255,255,255,.03);
        margin-bottom: .9rem;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
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

    .score-box, .editor-card {
        border: 1px solid rgba(212,175,55,.14);
        border-radius: 18px;
        padding: 1rem;
        background: rgba(255,255,255,.03);
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
        margin-bottom: 1rem;
    }

    .quote-box {
        border-left: 4px solid rgba(212,175,55,.92);
        padding: .95rem 1rem;
        background: rgba(212,175,55,.08);
        border-radius: 0 14px 14px 0;
        margin: .5rem 0 1rem 0;
        color: #fff3cf;
    }

    .upload-label {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: .35rem;
    }

    .tag-pill {
        display: inline-block;
        padding: .35rem .65rem;
        border-radius: 999px;
        background: rgba(212,175,55,.10);
        border: 1px solid rgba(212,175,55,.18);
        color: #fff0b8;
        font-size: .88rem;
        margin: 0 .35rem .45rem 0;
    }

    .editor-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1d67a;
        margin-bottom: .45rem;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,.03);
        border: 1px solid rgba(212,175,55,.12);
        border-radius: 18px;
        padding: .35rem .7rem .6rem .7rem;
    }

    div[data-testid="stFileUploader"] section {
        border: 1px dashed rgba(212,175,55,.22) !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,.02);
    }

    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] p {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.03);
        border: 1px solid rgba(212,175,55,.10);
        padding: .85rem;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,.16);
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

    details {
        background: rgba(255,255,255,.03);
        border: 1px solid rgba(212,175,55,.10);
        border-radius: 14px;
        padding: .4rem .7rem;
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

    .stButton > button {
        background: linear-gradient(135deg, rgba(212,175,55,.18), rgba(212,175,55,.08));
        color: #fff8dc;
        border: 1px solid rgba(212,175,55,.28);
        border-radius: 14px;
        padding: 0.78rem 1.2rem;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(0,0,0,.16);
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
    st.markdown("### Değerlendirme modu")
    selected_mode = st.radio(
        label="",
        options=list(MODE_PROFILES.keys()),
        index=0,
        label_visibility="collapsed",
    )
    st.markdown(f"**{selected_mode}**")
    st.markdown(MODE_PROFILES[selected_mode]["description"])
    st.markdown("### Temel soru")
    st.warning(CULTURE["temel_soru"])
    st.markdown("### Fotoğrafa yaklaşım")
    for q in CULTURE["okuma_sorulari"][:3]:
        st.markdown(f"- {q}")
    st.markdown(
        f"<div class='mini-note'>{MODE_PROFILES[selected_mode]['focus_hint']}</div>",
        unsafe_allow_html=True,
    )

header_left, header_right = st.columns([0.18, 0.82])

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
                yapıcı, dikkatli ve insancıl bir ön değerlendirme sistemi.
            </p>
            <div class="hero-badge">Aktif mod: {selected_mode}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.button("Fotoğraf yüklemeye başla", use_container_width=False)
st.markdown("<div class='upload-label'>Fotoğraf yükleyin</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
    help="JPG, JPEG, PNG, WEBP, TIF ve TIFF desteklenir.",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = load_image_from_upload(uploaded_file)
    result = critique_image(image, selected_mode)

    col1, col2 = st.columns([1.02, 0.98])

    with col1:
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with col2:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.metric("Genel Seviye", result.overall_level, result.overall_tag)
        st.progress(result.total_score / 100)
        st.markdown("#### Editör özeti")
        st.info(result.editor_summary)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Kısa etiketler")
    st.markdown(
        "".join([f"<span class='tag-pill'>{tag}</span>" for tag in result.tags]),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(f"## {selected_mode} Okuması")

    st.markdown(
        f"""
        <div class="editor-card">
            <div class="editor-title">İlk okuma</div>
            <div class="mini-note">{result.first_reading}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="editor-card">
            <div class="editor-title">Yapısal okuma</div>
            <div class="mini-note">{result.structural_reading}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="editor-card">
            <div class="editor-title">Editöryel sonuç</div>
            <div class="mini-note">{result.editorial_result}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Güçlü taraflar")
        for item in result.strengths:
            st.markdown(f"- {item}")

    with c2:
        st.markdown("### Gelişime açık alanlar")
        for item in result.development_areas:
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
        st.markdown("## Bir sonraki çekim için notlar")
        for item in result.shooting_notes:
            st.markdown(f"- {item}")

    with q2:
        st.markdown("## Düzenleme için notlar")
        for item in result.editing_notes:
            st.markdown(f"- {item}")

    st.markdown("---")
    st.markdown("## Okumayı derinleştiren sorular")
    for item in result.reading_prompts:
        st.markdown(f"- {item}")

    with st.expander("Teknik metrikler"):
        st.json(result.metrics)

else:
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="section-title">Başlamak için bir fotoğraf yükleyin</div>
            <div class="mini-note">
                Aktif mod şu anda <strong>{selected_mode}</strong>. Bu modda değerlendirme;
                {MODE_PROFILES[selected_mode]["description"].lower()}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
