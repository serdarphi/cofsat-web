import os
import io
import re
import json
import base64
import random
from typing import Dict, Any, List, Tuple

import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageFilter, ImageDraw

# Optional OpenAI integration
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# -----------------------------
# Page setup and theme helpers
# -----------------------------
st.set_page_config(
    page_title="ÇOFSAT PRO V3",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --melon: #f28c52;
  --melon-2: #f6a06f;
  --ink: #171717;
  --soft: #f7f3ef;
  --card: rgba(255,255,255,0.82);
  --border: rgba(242,140,82,0.25);
  --muted: #6b7280;
}
html, body, [class*="css"] { font-family: Inter, Arial, sans-serif; }
.stApp {
  background: radial-gradient(circle at top left, #fff8f2 0%, #fff 40%, #f7f7f7 100%);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f28c52 0%, #ec8350 100%);
}
section[data-testid="stSidebar"] * {
  color: white !important;
}
.block-container { padding-top: 1.05rem; padding-bottom: 1.4rem; }
.hero {
  background: linear-gradient(135deg, rgba(242,140,82,0.18), rgba(242,140,82,0.08));
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 20px 22px;
  margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}
.metric-card, .glass, .result-card, .editor-card {
  background: var(--card);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.7);
  border-radius: 22px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}
.metric-card { padding: 16px 18px; min-height: 118px; }
.result-card { padding: 18px; margin-bottom: 14px; }
.editor-card { padding: 18px; margin-bottom: 14px; }
.mini-label { font-size: 0.84rem; color: #6b7280; margin-bottom: 4px; }
.big-number { font-size: 2rem; font-weight: 800; color: var(--ink); line-height: 1.05; }
.kicker { color: var(--melon); font-weight: 800; letter-spacing: .03em; text-transform: uppercase; font-size: .8rem; }
.title { color: var(--ink); font-size: 2rem; font-weight: 900; margin: 0; }
.subtitle { color: #4b5563; margin-top: 8px; }
.badge {
  display:inline-block; padding:6px 10px; border-radius:999px;
  background:rgba(242,140,82,0.13); color:#b45309; font-weight:700;
  margin-right: 6px; margin-bottom: 6px; font-size:.85rem;
}
.badge-soft {
  display:inline-block; padding:5px 9px; border-radius:999px;
  background:rgba(17,24,39,0.06); color:#374151; font-weight:700;
  margin-right: 6px; margin-bottom: 6px; font-size:.80rem;
}
.section-title { font-size: 1.15rem; font-weight: 800; margin-bottom: 10px; color: #111827; }
.helper-note {
  border-left: 4px solid var(--melon); padding: 10px 12px;
  background: rgba(242,140,82,0.08); border-radius: 10px; color: #7c2d12;
}
.editor-name { font-weight: 900; color: #111827; font-size: 1.05rem; }
.editor-sub { color: var(--muted); font-size: .86rem; margin-top: 4px; margin-bottom: 10px; }
hr.soft {
  border: none; height: 1px;
  background: linear-gradient(90deg, rgba(242,140,82,0), rgba(242,140,82,.4), rgba(242,140,82,0));
  margin: 14px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Editor system
# -----------------------------
EDITORS: Dict[str, Dict[str, Any]] = {
    "selahattin": {
        "name": "Selahattin Kalaycı",
        "subtitle": "editoryal • kültürel hafıza • derin okuma",
        "style": "cultural_technical",
        "tone": "calm_authoritative",
        "badges": ["kompozisyon", "perspektif", "bağlam", "görsel hafıza"],
    },
    "guler": {
        "name": "Güler Ataşer",
        "subtitle": "sıcak • ilişki odaklı • duygusal kompozisyon",
        "style": "emotional_compositional",
        "tone": "warm_observational",
        "badges": ["duygu", "bağ", "ışık", "doku"],
    },
    "sevgin": {
        "name": "Sevgin Cingöz",
        "subtitle": "kısa • temiz • pozitif görsel okuma",
        "style": "minimal_visual",
        "tone": "positive_clean",
        "badges": ["kompozisyon", "yerleşim", "ton", "atmosfer"],
    },
}

SYNONYMS = {
    "dağınık": ["fazla konuşuyor", "gereğinden yoğun", "dikkati parçalıyor"],
    "etkili": ["güçlü", "yerinde", "karşılık bulan"],
    "hikâye": ["anlatı", "duygu hattı", "görsel cümle"],
    "kompozisyon": ["kadraj düzeni", "yerleşim kurgusu", "görsel yapı"],
    "ton": ["tonlama", "ışık dengesi", "ton geçişi"],
}

REPEAT_BREAKER = {
    "etkili kompozisyon": ["yerinde bir görsel yapı", "karşılığı olan bir kadraj", "güçlü bir kompozisyon"],
    "güzel bir kare": ["yerinde bir fotoğraf", "karşılık bulan bir kare", "temiz bir sonuç"],
    "emeğinize sağlık": ["kutlarım", "eline sağlık", "emeğiniz karşılık bulmuş"],
    "güzel fotoğraf": ["yerinde bir fotoğraf", "güçlü bir görüntü", "temiz bir çalışma"],
    "fotoğraf": ["kare", "görüntü", "sahne"],
    "kadraj": ["çerçeve", "yerleşim", "görsel alan"],
    "kompozisyon": ["görsel yapı", "kadraj düzeni", "yerleşim kurgusu"],
    "ton": ["tonlama", "ışık dengesi", "ton geçişi"],
    "hikâye": ["anlatı", "duygu hattı", "görsel cümle"],
}

STOPWORDS = {
    "ve", "bir", "bu", "ile", "da", "de", "için", "gibi", "ama", "çok", "daha", "olarak", "ise",
    "olan", "kadar", "göre", "ya", "hem", "sonuç", "fotoğraf", "kare", "sahne"
}
MEMORY_LIMIT = 8


# -----------------------------
# Utilities
# -----------------------------
def image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    if img.mode in ("RGBA", "LA") and fmt.upper() == "JPEG":
        img = img.convert("RGB")
    img.save(buffer, format=fmt, quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_from_upload(uploaded_file) -> Image.Image:
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


def avg_brightness(img: Image.Image) -> float:
    gray = img.convert("L")
    return float(ImageStat.Stat(gray).mean[0])


def contrast_score(img: Image.Image) -> float:
    gray = img.convert("L")
    return float(min(100, ImageStat.Stat(gray).stddev[0] * 1.6))


def saturation_score(img: Image.Image) -> float:
    rgb = img.convert("RGB")
    pixels = list(rgb.resize((160, 160)).getdata())
    sats = []
    for r, g, b in pixels:
        mx, mn = max(r, g, b), min(r, g, b)
        sats.append(0 if mx == 0 else (mx - mn) / mx)
    return float(sum(sats) / max(len(sats), 1) * 100)


def edge_density(img: Image.Image) -> float:
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(min(100, ImageStat.Stat(edges).mean[0] * 1.6))


def subject_box_guess(img: Image.Image) -> Tuple[int, int, int, int]:
    w, h = img.size
    return int(w * 0.22), int(h * 0.18), int(w * 0.78), int(h * 0.82)


def scene_tags_heuristic(img: Image.Image) -> List[str]:
    w, h = img.size
    ratio = w / max(h, 1)
    bright = avg_brightness(img)
    contrast = contrast_score(img)
    edge = edge_density(img)
    sat = saturation_score(img)
    tags = []
    if ratio > 1.5:
        tags.append("geniş kadraj")
    if bright < 70:
        tags.append("düşük ışık")
    if contrast > 55:
        tags.append("yüksek kontrast")
    if sat < 18:
        tags.append("düşük doygunluk")
    if edge > 42:
        tags.append("detay yoğun")
    if bright > 160:
        tags.append("aydınlık sahne")
    return tags[:5]


def classify_photo_type_heuristic(img: Image.Image) -> Tuple[str, float, Dict[str, float]]:
    w, h = img.size
    ratio = w / max(h, 1)
    bright = avg_brightness(img)
    contrast = contrast_score(img)
    sat = saturation_score(img)
    edge = edge_density(img)

    scores = {
        "Sokak": 0.0,
        "Portre": 0.0,
        "Belgesel": 0.0,
        "Mimari": 0.0,
        "Manzara": 0.0,
        "Minimal": 0.0,
        "Gece": 0.0,
    }

    if edge > 42:
        scores["Sokak"] += 24
        scores["Belgesel"] += 20
        scores["Mimari"] += 24
    if sat < 20:
        scores["Minimal"] += 20
        scores["Belgesel"] += 8
    if ratio > 1.35:
        scores["Manzara"] += 26
        scores["Mimari"] += 10
    if 0.8 < ratio < 1.25:
        scores["Portre"] += 16
        scores["Sokak"] += 10
    if bright < 75:
        scores["Gece"] += 30
        scores["Sokak"] += 8
    if contrast > 48:
        scores["Belgesel"] += 12
        scores["Sokak"] += 8
        scores["Gece"] += 4
    if edge < 28 and sat < 25:
        scores["Minimal"] += 22
        scores["Portre"] += 8
    if edge > 50 and ratio > 1.2:
        scores["Mimari"] += 14
    if bright > 130 and ratio > 1.25:
        scores["Manzara"] += 12

    label = max(scores, key=scores.get)
    conf = max(40.0, min(94.0, scores[label] + 35.0))
    return label, conf, scores


def overlay_analysis(img: Image.Image) -> Dict[str, Image.Image]:
    base = img.convert("RGBA")
    w, h = base.size

    focus = base.copy()
    draw = ImageDraw.Draw(focus, "RGBA")
    box = subject_box_guess(img)
    draw.rounded_rectangle(box, radius=18, outline=(255, 188, 66, 255), width=6)
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    draw.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), outline=(255, 80, 80, 255), width=5)

    eye = base.copy()
    draw = ImageDraw.Draw(eye, "RGBA")
    points = [
        (int(w * 0.12), int(h * 0.68)),
        (int(w * 0.35), int(h * 0.56)),
        (int(w * 0.52), int(h * 0.48)),
        (int(w * 0.68), int(h * 0.42)),
        (int(w * 0.82), int(h * 0.38)),
    ]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=(64, 191, 255, 230), width=8)
        x, y = points[i + 1]
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(64, 191, 255, 220))

    dis = base.copy()
    draw = ImageDraw.Draw(dis, "RGBA")
    regions = [
        (int(w * 0.02), int(h * 0.05), int(w * 0.22), int(h * 0.24)),
        (int(w * 0.78), int(h * 0.02), int(w * 0.98), int(h * 0.18)),
        (int(w * 0.80), int(h * 0.72), int(w * 0.98), int(h * 0.96)),
    ]
    for r in regions:
        draw.rounded_rectangle(r, radius=14, outline=(255, 72, 72, 255), width=5, fill=(255, 72, 72, 40))

    heat = base.copy()
    draw = ImageDraw.Draw(heat, "RGBA")
    heat_regions = [
        (int(w * 0.28), int(h * 0.28), int(w * 0.72), int(h * 0.78), (255, 160, 0, 55)),
        (int(w * 0.34), int(h * 0.34), int(w * 0.66), int(h * 0.70), (255, 70, 0, 70)),
        (int(w * 0.42), int(h * 0.40), int(w * 0.58), int(h * 0.60), (255, 0, 0, 85)),
    ]
    for l, t, r, b, color in heat_regions:
        draw.ellipse((l, t, r, b), fill=color)

    return {
        "Odak Noktası": focus,
        "Göz Akışı": eye,
        "Dikkat Dağıtan Alanlar": dis,
        "Görsel Ağırlık / Isı Haritası": heat,
    }


def score_bundle(img: Image.Image) -> Dict[str, int]:
    bright = avg_brightness(img)
    contrast = contrast_score(img)
    sat = saturation_score(img)
    edge = edge_density(img)
    story = int(max(48, min(93, 55 + (contrast * 0.18) + (edge * 0.12))))
    composition = int(max(44, min(94, 58 + (edge * 0.15) + (10 if 0.8 < img.size[0] / max(img.size[1], 1) < 1.8 else 0))))
    light = int(max(40, min(95, 64 - abs(bright - 118) * 0.34 + contrast * 0.12)))
    tonal = int(max(42, min(94, 56 + contrast * 0.2 + (8 if sat < 28 else 0))))
    focus = int(max(43, min(95, 60 + edge * 0.14 - sat * 0.04)))
    return {
        "Hikâye": story,
        "Kompozisyon": composition,
        "Işık": light,
        "Tonlama": tonal,
        "Özne Gücü": focus,
    }


def build_recipe(photo_type: str, img: Image.Image, scores: Dict[str, int]) -> Dict[str, Any]:
    bright = avg_brightness(img)
    contrast = contrast_score(img)
    sat = saturation_score(img)

    exposure = "+0.20" if bright < 92 else ("-0.15" if bright > 150 else "+0.00")
    highlights = "-35" if bright > 140 else "-18"
    shadows = "+28" if bright < 92 else "+12"
    whites = "-12" if bright > 150 else "+8"
    blacks = "-18" if contrast < 38 else "-8"
    clarity = "+8" if photo_type in ["Sokak", "Belgesel", "Mimari"] else "+3"
    texture = "+10" if photo_type in ["Belgesel", "Mimari"] else "+4"
    dehaze = "+7" if bright < 90 else "+3"
    vibrance = "+5" if sat < 20 else "+0"
    vignette = "-10" if photo_type in ["Portre", "Gece", "Minimal"] else "-4"

    type_rules = {
        "Sokak": {
            "tek_hamle": "arka planın dikkat çalan parlak bölgelerini bastırıp ana özneyi lokal kontrastla ayır",
            "crop": "kenarlardaki gereksiz ağırlığı %5–8 kırp",
            "local": "öznenin yüz/beden hattına hafif dodge, çevreye yumuşak burn uygula",
        },
        "Portre": {
            "tek_hamle": "yüzü sahnenin en kontrollü parlak alanı yap",
            "crop": "göz hizasını üst üçte bire yaklaştır",
            "local": "göz, yüz ve el bölgelerinde yumuşak lokal aydınlatma kullan",
        },
        "Belgesel": {
            "tek_hamle": "hikâyeyi dağıtan ikinci parlak odağı bastır",
            "crop": "anlatıya hizmet etmeyen boşluğu azalt",
            "local": "ana olay çevresinde mikro kontrastı artır",
        },
        "Mimari": {
            "tek_hamle": "çizgisel yapıyı ton ayrımıyla daha net vurgula",
            "crop": "eğim hissi veren boş kenarları temizle",
            "local": "yapısal yüzeylere texture ve kontrollü clarity ver",
        },
        "Manzara": {
            "tek_hamle": "ön plan-orta plan-arka plan katmanlarını tonla ayır",
            "crop": "ufuk hattını daha kararlı konuma taşı",
            "local": "gökyüzü ve ön plan için ayrı maske kullan",
        },
        "Minimal": {
            "tek_hamle": "sahneyi iki ya da üç ana tona indirerek sadeliği güçlendir",
            "crop": "fazla bilgiyi cesurca kes",
            "local": "tek özne dışındaki alanlarda dikkat azalt",
        },
        "Gece": {
            "tek_hamle": "gürültüyü artırmadan özneyi karanlıktan ayır",
            "crop": "parazit oluşturan köşe ışıklarını temizle",
            "local": "ışık kaynaklarının çevresini kontrol ederek glow’u sınırlı tut",
        },
    }
    rule = type_rules.get(photo_type, type_rules["Sokak"])

    return {
        "temel": {
            "Pozlama": exposure,
            "Highlights": highlights,
            "Shadows": shadows,
            "Whites": whites,
            "Blacks": blacks,
            "Clarity": clarity,
            "Texture": texture,
            "Dehaze": dehaze,
            "Vibrance": vibrance,
            "Vignette": vignette,
        },
        "tek_hamle": rule["tek_hamle"],
        "kırpma": rule["crop"],
        "lokal": rule["local"],
        "akış": [
            "Önce global pozlama ve highlight/shadow dengesini kur.",
            "Sonra ana özne ile arka planı lokal maske ile ayır.",
            "En sonda dikkat dağıtan parlaklıkları ve köşeleri sakinleştir.",
        ],
    }


def build_core_analysis(photo_type: str, img: Image.Image, scores: Dict[str, int], recipe: Dict[str, Any]) -> Dict[str, str]:
    bright = avg_brightness(img)
    contrast = contrast_score(img)
    sat = saturation_score(img)
    edge = edge_density(img)

    if scores["Kompozisyon"] >= 78:
        composition = random.choice([
            "Kadraj yerleşimi sahneyi taşımayı başarıyor",
            "Kompozisyon ana karar noktasını fena kurmuyor",
            "Yerleşim kurgusu özneye doğru çalışan bir yapı oluşturuyor",
        ])
    else:
        composition = random.choice([
            "Kadraj kararında küçük bir sıkışma hissi var",
            "Yerleşim tam oturmadığı için sahnenin etkisi bölünüyor",
            "Kompozisyon tarafında biraz daha sade bir karar fotoğrafı güçlendirebilirdi",
        ])

    if scores["Hikâye"] >= 76:
        story = random.choice([
            "Fotoğrafın bir anlatı kurma isteği belirgin",
            "Sahne kendi hikâyesini izleyiciye geçirebiliyor",
            "Karede bir duygu ve anlatı hattı kurulmuş durumda",
        ])
    else:
        story = random.choice([
            "Anlatı var ama tam kristalleşmiyor",
            "Hikâye duygusu kuruluyor, fakat merkez daha net seçilebilirdi",
            "Fotoğraf bir şey söylüyor ama sesi yer yer dağılabiliyor",
        ])

    if scores["Işık"] >= 74:
        light = random.choice([
            "Işık özneyi destekleyen bir denge kuruyor",
            "Işık kullanımı sahnenin atmosferini taşıyor",
            "Parlaklık ilişkisi fotoğrafın kararını destekliyor",
        ])
    else:
        light = random.choice([
            "Işık tarafında özneyi daha belirgin ayıracak bir karar gerekebilirdi",
            "Parlak bölgeler ana anlatıyla yarışıyor",
            "Ton dengesinde biraz daha kontrollü bir ışık dağılımı iyi olurdu",
        ])

    if scores["Tonlama"] >= 74:
        tones = random.choice([
            "Ton geçişleri görüntünün ritmini taşıyor",
            "Tonlama sahnenin duygusuna karşılık veriyor",
            "Işık ve gölge arasındaki ilişki okunabilir durumda",
        ])
    else:
        tones = random.choice([
            "Ton tarafında biraz daha sadeleştirme fotoğrafı yükseltebilirdi",
            "Orta ton yoğunluğu dikkat akışını dağıtıyor",
            "Tonlama kararları özneyi yeterince yalnız bırakmıyor",
        ])

    if scores["Özne Gücü"] >= 74:
        subject = random.choice([
            "Ana özne sahnenin yükünü taşımayı başarıyor",
            "Özne yeterince görünür ve tutarlı duruyor",
            "Bakışın toplandığı merkez büyük ölçüde belli",
        ])
    else:
        subject = random.choice([
            "Ana özne biraz daha belirginleştirilebilirdi",
            "Bakışın ilk tutunduğu alan daha kararlı seçilmeliydi",
            "Özne çevresiyle rekabet ediyor",
        ])

    relation = random.choice([
        "Figürler veya öğeler arasında bir bağ hissi oluşuyor",
        "Sahnedeki unsurlar birbirini destekleyen bir ilişki kuruyor",
        "Görsel yapı ile duygu hattı arasında bir temas var",
    ])

    emotion = random.choice([
        "yalnızlık ve direnç",
        "yakınlık ve aidiyet",
        "sessizlik ve bekleyiş",
        "hareket ile durağanlık arasındaki gerilim",
        "tanıklık duygusu",
    ])

    texture = "Dokular ve yüzeyler sahnenin ikincil gücüne katkı sağlıyor" if edge > 42 else "Doku tarafı geri planda ama sahneyi boğmuyor"
    lines = "Çizgisel akış bakışı belirli bir yöne taşıyor" if contrast > 40 else "Çizgiler yeterince baskın olmasa da sahnenin düzenini destekliyor"
    atmosphere = "Atmosfer fotoğrafın duygusunu kuruyor" if bright < 110 or sat < 22 else "Atmosfer daha çok ışık ve yerleşim kararından besleniyor"
    reflection = "Yansıma ve yüzey ilişkileri sahneye ikinci bir katman ekliyor" if sat > 18 and bright > 95 else "Yansıma etkisi bu karede ana unsur değil"
    gesture = "Beden dili sahnenin duygusunu güçlendiriyor" if scores["Hikâye"] > 72 else "Beden dili daha okunur olsaydı anlatı kuvvetlenebilirdi"
    visual_flow = "Göz akışı fotoğraf içinde belirli bir rota izliyor" if scores["Kompozisyon"] > 72 else "Göz akışı yer yer dağılma eğilimi gösteriyor"
    one_move = recipe["tek_hamle"]

    return {
        "photo_type": photo_type,
        "composition": composition,
        "story": story,
        "light": light,
        "tones": tones,
        "subject": subject,
        "relationship": relation,
        "emotion": emotion,
        "texture": texture,
        "lines": lines,
        "atmosphere": atmosphere,
        "reflection": reflection,
        "gesture": gesture,
        "visual_flow": visual_flow,
        "one_move": one_move,
    }


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("â", "a").replace("î", "i").replace("û", "u")
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_memory(text: str) -> List[str]:
    tokens: List[str] = []
    for token in _normalize_text(text).split():
        if len(token) < 4 or token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def recent_memory_for(editor_key: str) -> Dict[str, Any]:
    if "editor_memory" not in st.session_state:
        st.session_state.editor_memory = {}
    return st.session_state.editor_memory.setdefault(editor_key, {"reviews": [], "tokens": [], "phrases": []})


def phrase_seen(phrase: str, memory: Dict[str, Any]) -> bool:
    normalized = _normalize_text(phrase)
    return normalized in memory.get("phrases", [])


def remember_review(editor_key: str, review: str) -> None:
    memory = recent_memory_for(editor_key)
    memory["reviews"].append(review)
    memory["reviews"] = memory["reviews"][-MEMORY_LIMIT:]
    memory["tokens"].extend(_tokenize_memory(review))
    memory["tokens"] = memory["tokens"][-220:]
    chunks = re.split(r"[.!?]+", review)
    phrases = [_normalize_text(chunk) for chunk in chunks if _normalize_text(chunk)]
    memory["phrases"].extend(phrases)
    memory["phrases"] = memory["phrases"][-MEMORY_LIMIT * 4:]


def choose_phrase(options: List[str], memory: Dict[str, Any], fallback: str = "") -> str:
    shuffled = options[:]
    random.shuffle(shuffled)
    for option in shuffled:
        if not phrase_seen(option, memory):
            return option
    return shuffled[0] if shuffled else fallback


def choose_variant(word: str, seen: set) -> str:
    options = SYNONYMS.get(word, [word])
    shuffled = options[:]
    random.shuffle(shuffled)
    for candidate in shuffled:
        if candidate not in seen:
            seen.add(candidate)
            return candidate
    seen.add(word)
    return word


def apply_repeat_breaker(text: str, editor_key: str, seen: set) -> str:
    memory = recent_memory_for(editor_key)
    recent_tokens = memory.get("tokens", [])
    token_counts = {tok: recent_tokens.count(tok) for tok in set(recent_tokens)}

    for phrase, replacements in REPEAT_BREAKER.items():
        normalized_phrase = _normalize_text(phrase)
        should_swap = normalized_phrase in memory.get("phrases", []) or any(
            tok in token_counts and token_counts[tok] >= 2 for tok in _tokenize_memory(phrase)
        )
        if should_swap and re.search(re.escape(phrase), text, flags=re.IGNORECASE):
            replacement = choose_phrase(replacements, memory, replacements[0])
            text = re.sub(re.escape(phrase), replacement, text, count=1, flags=re.IGNORECASE)

    words = re.findall(r"\b[\wçğıöşüÇĞİÖŞÜ]+\b", text, flags=re.UNICODE)
    for word in words:
        normalized_word = _normalize_text(word)
        if normalized_word in token_counts and token_counts[normalized_word] >= 3:
            replacement = choose_variant(normalized_word, seen)
            if replacement != normalized_word:
                text = re.sub(rf"\b{re.escape(word)}\b", replacement, text, count=1, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip()


def shorten_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text + "."


def write_as_selahattin(core: Dict[str, str], seen: set, memory: Dict[str, Any]) -> str:
    openers = [
        shorten_sentence(core["composition"]),
        shorten_sentence(core["subject"]),
        shorten_sentence(core["light"]),
    ]
    opener = choose_phrase(openers, memory, openers[0])

    structure = [
        shorten_sentence(core["visual_flow"] + " ve bu durum özneye doğru çalışan bir dil kuruyor"),
        shorten_sentence(core["story"]),
    ]

    cultural = choose_phrase([
        "Fotoğraf, görsel kararının ötesinde bir hafızaya da dokunuyor.",
        "Sahnenin taşıdığı hava yalnız bugünü değil, geçmişi de çağırıyor.",
        "Bu kare, yalnızca görüleni değil, arkasındaki kültürel sesi de hissettiriyor.",
    ], memory)

    imagination = ""
    if random.random() > 0.45:
        imagination = choose_phrase([
            "Zaman burada biraz yavaşlıyor; mekân yalnızca fon değil, anlatının sessiz ortağı gibi duruyor.",
            "Figür ya da ana öğe, sanki sahnenin içinden geçip başka bir zamana doğru ilerliyormuş hissi bırakıyor.",
            "Mekânın kendisi de fotoğrafın öznesine eşlik eden ikinci bir karakter gibi davranıyor.",
        ], memory)

    closer = choose_phrase([
        f"Bu haliyle de {choose_variant('etkili', seen)} bir fotoğraf.",
        "Birkaç küçük karar daha netleştiğinde çok daha kalıcı bir karşılık bulabilir.",
        "Yine de fotoğrafın kurduğu dil izleyicide karşılık buluyor.",
    ], memory)

    parts = [opener, structure[0], structure[1], cultural, imagination, closer]
    text = " ".join([p for p in parts if p])
    return apply_repeat_breaker(text, "selahattin", seen)


def write_as_guler(core: Dict[str, str], seen: set, memory: Dict[str, Any]) -> str:
    lead = choose_phrase([
        f"{core['relationship']}; bu durum fotoğrafın {core['emotion']} duygusunu güçlendiriyor.",
        f"Fotoğrafta {core['emotion']} hissi belirgin ve bu duygu sahnedeki öğeler arasındaki bağla büyüyor.",
        f"Sahnedeki ilişki, fotoğrafın duygusal ağırlığını belirleyen en önemli unsur gibi duruyor.",
    ], memory)

    middle = choose_phrase([
        f"{core['composition']}, {core['light']} ve {core['tones'].lower()}.",
        f"{core['texture']}, {core['lines'].lower()} ve {core['atmosphere'].lower()}.",
        f"{core['gesture']}, {core['composition'].lower()} ve {core['tones'].lower()}.",
    ], memory)

    closer = choose_phrase([
        "Bütün bunlar bir araya gelince sahne hem duygusal hem görsel olarak karşılık buluyor.",
        f"Bu yüzden kare {choose_variant('etkili', seen)} bir bütünlük taşıyor.",
        "Sonuçta fotoğrafın hem duygusu hem de görsel yapısı birbirini destekliyor.",
    ], memory)

    text = " ".join([lead, middle, closer])
    return apply_repeat_breaker(text, "guler", seen)


def write_as_sevgin(core: Dict[str, str], seen: set, memory: Dict[str, Any]) -> str:
    start = choose_phrase([
        "Etkili kompozisyon.",
        "Güzel bir kadraj düzeni.",
        "Yerleşim kararı başarılı.",
    ], memory)
    extra = choose_phrase([
        f"{core['gesture']}",
        f"{core['tones']}",
        f"{core['atmosphere']}",
        f"{core['subject']}",
    ], memory)
    closer = choose_phrase([
        "Güzel bir kare.",
        "Emeğinize sağlık.",
        f"Sonuç olarak {choose_variant('etkili', seen)} bir fotoğraf.",
    ], memory)
    text = " ".join([start, shorten_sentence(extra), closer])
    return apply_repeat_breaker(text, "sevgin", seen)


def generate_editor_reviews(core: Dict[str, str]) -> Dict[str, str]:
    seen: set = set()
    reviews = {
        "selahattin": write_as_selahattin(core, seen, recent_memory_for("selahattin")),
        "guler": write_as_guler(core, seen, recent_memory_for("guler")),
        "sevgin": write_as_sevgin(core, seen, recent_memory_for("sevgin")),
    }
    for editor_key, review in reviews.items():
        remember_review(editor_key, review)
    return reviews


def build_debate(core: Dict[str, str], photo_type: str) -> List[Tuple[str, str]]:
    return [
        (EDITORS["selahattin"]["name"], random.choice([
            f"{core['composition']} Bu yüzden ilk karar noktası kadraj tarafında beliriyor.",
            f"{core['story']} Yine de sahnenin dili biraz daha seçici kurulabilirdi.",
        ])),
        (EDITORS["guler"]["name"], random.choice([
            f"Benim için asıl güç {core['relationship'].lower()} ve fotoğrafın {core['emotion']} duygusunu taşıması.",
            f"Sahnenin duygusal ağırlığı, teknik küçük kusurlardan daha baskın hissediliyor.",
        ])),
        (EDITORS["sevgin"]["name"], random.choice([
            "Genel etki güçlü. Özellikle yerleşim ve atmosfer birlikte çalışıyor.",
            f"Kısa bakışta bile {photo_type.lower()} karakteri net hissediliyor.",
        ])),
    ]


def default_analysis(img: Image.Image) -> Dict[str, Any]:
    photo_type, confidence, raw_scores = classify_photo_type_heuristic(img)
    tags = scene_tags_heuristic(img)
    scores = score_bundle(img)
    recipe = build_recipe(photo_type, img, scores)
    core = build_core_analysis(photo_type, img, scores, recipe)
    editor_reviews = generate_editor_reviews(core)
    debate = build_debate(core, photo_type)

    strong = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
    weak = sorted(scores.items(), key=lambda x: x[1])[:2]

    common_summary = [
        core["story"],
        core["composition"],
        f"En etkili düzenleme yönü: {recipe['tek_hamle']}.",
    ]

    return {
        "analysis_source": "heuristic",
        "photo_type": photo_type,
        "confidence": round(confidence, 1),
        "scene_tags": tags,
        "scores": scores,
        "strong_sides": [f"{k} güçlü görünüyor" for k, _ in strong],
        "risks": [f"{k} tarafında toparlama gerekiyor" for k, _ in weak],
        "summary": common_summary,
        "one_move": recipe["tek_hamle"],
        "recipe": recipe,
        "raw_type_scores": raw_scores,
        "core_analysis": core,
        "editor_reviews": editor_reviews,
        "debate": debate,
    }


def normalize_vision_payload(data: Dict[str, Any], fallback_img: Image.Image) -> Dict[str, Any]:
    fallback = default_analysis(fallback_img)
    merged = fallback.copy()
    merged.update({k: v for k, v in data.items() if v is not None})

    scores = merged.get("scores") or fallback["scores"]
    recipe = merged.get("recipe") or fallback["recipe"]
    photo_type = merged.get("photo_type") or fallback["photo_type"]

    core = merged.get("core_analysis")
    if not core:
        core = build_core_analysis(photo_type, fallback_img, scores, recipe)
        if isinstance(data.get("summary"), list) and len(data.get("summary")) >= 2:
            core["story"] = str(data["summary"][0])
            core["composition"] = str(data["summary"][1])
    merged["core_analysis"] = core
    merged["editor_reviews"] = generate_editor_reviews(core)
    merged["debate"] = build_debate(core, photo_type)
    return merged


def vision_prompt() -> str:
    return """
Sen ÇOFSAT PRO V3 isimli fotoğraf okuma motorusun.
Görev: Fotoğrafı analiz et ve SADECE geçerli JSON döndür.
Yorum dili: Türkçe.

JSON şeması:
{
  "analysis_source": "vision",
  "photo_type": "Sokak|Portre|Belgesel|Mimari|Manzara|Minimal|Gece|Diğer",
  "confidence": 0-100,
  "scene_tags": ["etiket1", "etiket2", "etiket3"],
  "scores": {
    "Hikâye": 0-100,
    "Kompozisyon": 0-100,
    "Işık": 0-100,
    "Tonlama": 0-100,
    "Özne Gücü": 0-100
  },
  "strong_sides": ["madde", "madde", "madde"],
  "risks": ["madde", "madde", "madde"],
  "summary": ["ortak yorum 1", "ortak yorum 2", "ortak yorum 3"],
  "one_move": "tek hamlede en etkili öneri",
  "recipe": {
    "temel": {
      "Pozlama": "+0.20",
      "Highlights": "-25",
      "Shadows": "+20",
      "Whites": "+5",
      "Blacks": "-10",
      "Clarity": "+6",
      "Texture": "+8",
      "Dehaze": "+4",
      "Vibrance": "+3",
      "Vignette": "-8"
    },
    "tek_hamle": "tek hamle önerisi",
    "kırpma": "kırpma önerisi",
    "lokal": "lokal düzenleme önerisi",
    "akış": ["adım1", "adım2", "adım3"]
  }
}

Kurallar:
- Klişe cümle üretme.
- Sahneye özgü konuş.
- JSON dışında hiçbir şey yazma.
""".strip()


def analyze_with_openai(img: Image.Image, model: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not OPENAI_AVAILABLE or not api_key:
        raise RuntimeError("OpenAI istemcisi veya OPENAI_API_KEY bulunamadı.")

    client = OpenAI(api_key=api_key)
    b64 = image_to_base64(img, fmt="JPEG")
    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": vision_prompt()},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
            ],
        }],
        temperature=0.2,
        max_output_tokens=2200,
    )
    text = getattr(response, "output_text", "") or ""
    data = safe_json_loads(text)
    if not data:
        raise RuntimeError("Vision model geçerli JSON döndürmedi.")
    return normalize_vision_payload(data, img)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## ÇOFSAT PRO V3")
    st.markdown("Çoklu editör kadrosu + görsel okuma katmanları + düzenleme reçetesi")

    analysis_engine = st.radio(
        "Analiz motoru",
        ["Akıllı Yerel Motor", "OpenAI Vision"],
        index=0 if not os.getenv("OPENAI_API_KEY") else 1,
    )

    model_name = st.text_input(
        "Vision model",
        value="gpt-5.4-mini",
        help="OpenAI Vision seçiliyse kullanılacak model adı.",
    )

    st.markdown("---")
    st.markdown("### Görsel katmanlar")
    show_focus = st.checkbox("Odak noktası", value=True)
    show_eye = st.checkbox("Göz akışı", value=True)
    show_distraction = st.checkbox("Dikkat dağıtan alanlar", value=True)
    show_heat = st.checkbox("Isı haritası", value=True)

    st.markdown("---")
    st.markdown("### Editör kadrosu")
    selected_editors = []
    if st.checkbox("Selahattin Kalaycı", value=True):
        selected_editors.append("selahattin")
    if st.checkbox("Güler Ataşer", value=True):
        selected_editors.append("guler")
    if st.checkbox("Sevgin Cingöz", value=True):
        selected_editors.append("sevgin")

    st.markdown("---")
    if st.button("Editör hafızasını sıfırla", use_container_width=True):
        st.session_state.editor_memory = {}
        st.success("Editör hafızası temizlendi.")

    st.markdown("---")
    st.markdown(
        "<div class='helper-note'>OpenAI Vision seçeneği için sistemde <b>OPENAI_API_KEY</b> tanımlı olmalı. Anahtar yoksa uygulama otomatik olarak yerel analiz motoruna döner.</div>",
        unsafe_allow_html=True,
    )


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="kicker">Çofsat Pro V3</div>
      <h1 class="title">Aynı analizi üç farklı editör sesiyle okuyan arayüz</h1>
      <div class="subtitle">Önce tek bir çekirdek analiz üretir, sonra aynı omurgayı Selahattin Kalaycı, Güler Ataşer ve Sevgin Cingöz diline dönüştürür.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Fotoğraf yükle", type=["jpg", "jpeg", "png", "webp"])

if not uploaded:
    st.info("Başlamak için bir fotoğraf yükleyin.")
    st.stop()

image = pil_from_upload(uploaded)
overlays = overlay_analysis(image)

col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown("### Fotoğraf")
    st.image(image, use_container_width=True)

with col_right:
    st.markdown("### Görsel okuma katmanları")
    selected_layers = []
    if show_focus:
        selected_layers.append("Odak Noktası")
    if show_eye:
        selected_layers.append("Göz Akışı")
    if show_distraction:
        selected_layers.append("Dikkat Dağıtan Alanlar")
    if show_heat:
        selected_layers.append("Görsel Ağırlık / Isı Haritası")

    if not selected_layers:
        st.info("En az bir katman seçin.")
    else:
        tabs = st.tabs(selected_layers)
        for tab, name in zip(tabs, selected_layers):
            with tab:
                st.image(overlays[name], use_container_width=True)

analyze = st.button("Analizi Başlat", type="primary", use_container_width=True)

if not analyze:
    st.stop()

with st.spinner("Fotoğraf okunuyor..."):
    if analysis_engine == "OpenAI Vision":
        try:
            analysis = analyze_with_openai(image, model_name)
        except Exception as e:
            st.warning(f"Vision model kullanılamadı. Yerel motora dönüldü. Neden: {e}")
            analysis = default_analysis(image)
    else:
        analysis = default_analysis(image)

scores = analysis.get("scores", {})
recipe = analysis.get("recipe", {})
scene_tags = analysis.get("scene_tags", [])
editor_reviews = analysis.get("editor_reviews", {})

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
metric_cols = st.columns(4, gap="medium")
metrics = [
    ("Fotoğraf Türü", analysis.get("photo_type", "Bilinmiyor")),
    ("Güven Skoru", f"%{analysis.get('confidence', 0)}"),
    ("Analiz Kaynağı", analysis.get("analysis_source", "-")),
    ("Tek Hamle", analysis.get("one_move", "-")),
]
for col, (label, val) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"<div class='metric-card'><div class='mini-label'>{label}</div><div style='font-weight:800;color:#111827'>{val}</div></div>",
            unsafe_allow_html=True,
        )

if scene_tags:
    st.markdown("### Sahne Etiketleri")
    st.markdown("".join([f"<span class='badge'>{t}</span>" for t in scene_tags]), unsafe_allow_html=True)

st.markdown("### Karar Paneli")
score_cols = st.columns(len(scores) if scores else 5)
for col, (name, value) in zip(score_cols, scores.items()):
    with col:
        st.markdown(
            f"<div class='metric-card'><div class='mini-label'>{name}</div><div class='big-number'>{value}</div></div>",
            unsafe_allow_html=True,
        )

body_left, body_right = st.columns([1, 1], gap="large")

with body_left:
    st.markdown("### Ortak Analiz")
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    for line in analysis.get("summary", []):
        st.write(f"• {line}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Güçlü Yanlar")
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    for item in analysis.get("strong_sides", []):
        st.write(f"✅ {item}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Riskler")
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    for item in analysis.get("risks", []):
        st.write(f"⚠️ {item}")
    st.markdown("</div>", unsafe_allow_html=True)

with body_right:
    st.markdown("### Fotoğrafa Özel Düzenleme Reçetesi")
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    temel = recipe.get("temel", {})
    if temel:
        st.markdown("**Temel Ayarlar**")
        st.json(temel)
    st.markdown(f"**Tek Hamle:** {recipe.get('tek_hamle', '-')}")
    st.markdown(f"**Kırpma:** {recipe.get('kırpma', '-')}")
    st.markdown(f"**Lokal Düzenleme:** {recipe.get('lokal', '-')}")
    steps = recipe.get("akış", [])
    if steps:
        st.markdown("**Uygulama Sırası**")
        for i, step in enumerate(steps, start=1):
            st.write(f"{i}. {step}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### ÇOFSAT Editör Kadrosu")
if not selected_editors:
    st.info("Kenardaki panelden en az bir editör seçin.")
else:
    for key in selected_editors:
        meta = EDITORS[key]
        review = editor_reviews.get(key, "Yorum üretilemedi.")
        badges_html = "".join([f"<span class='badge-soft'>{b}</span>" for b in meta["badges"]])
        st.markdown("<div class='editor-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='editor-name'>✍️ {meta['name']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='editor-sub'>{meta['subtitle']}</div>", unsafe_allow_html=True)
        st.markdown(badges_html, unsafe_allow_html=True)
        st.write(review)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Editörler Ne Diyor?")
st.markdown("<div class='result-card'>", unsafe_allow_html=True)
for name, line in analysis.get("debate", []):
    st.write(f"**{name}:** {line}")
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Editör hafızası ve tekrar kırıcı durumu"):
    memory_state = st.session_state.get("editor_memory", {})
    for key in selected_editors:
        meta = EDITORS[key]
        memory = memory_state.get(key, {})
        recent_reviews = memory.get("reviews", [])[-3:]
        recent_tokens = memory.get("tokens", [])
        token_rank = sorted(set(recent_tokens), key=lambda t: recent_tokens.count(t), reverse=True)[:8]
        st.markdown(f"**{meta['name']}**")
        st.write(f"Son yorum sayısı: {len(memory.get('reviews', []))}")
        if token_rank:
            st.write("Sık geçen kelimeler:", ", ".join([f"{tok} ({recent_tokens.count(tok)})" for tok in token_rank]))
        if recent_reviews:
            st.write("Son 3 yorum:")
            for item in recent_reviews:
                st.write(f"- {item}")
        st.markdown("---")

with st.expander("Çekirdek analiz verisi"):
    st.json(analysis.get("core_analysis", {}))

with st.expander("Ham analiz verisi"):
    st.json(analysis)

st.caption("Not: Bu sürümde çekirdek analiz tek kez üretilir. Editörler aynı omurgayı farklı yazım DNA'sıyla yeniden yorumlar. OpenAI Vision etkinse çekirdek analiz sahneye daha özgül hale gelir.")
