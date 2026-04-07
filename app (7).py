import os
import io
import re
import json
import base64
import random
import hashlib
from collections import Counter
from typing import Dict, Any, List, Tuple

import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageFilter, ImageDraw

OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

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
  --melon-dark: #dd7c46;
  --ink: #171717;
  --muted: #5b6472;
  --card: rgba(255,255,255,0.88);
  --border: rgba(242,140,82,0.22);
}
html, body, [class*="css"] { font-family: Inter, Arial, sans-serif; }
.stApp { background: radial-gradient(circle at top left, #fff8f2 0%, #fff 42%, #f6f7f9 100%); }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f28c52 0%, #ea854f 100%);
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: white !important; }
.block-container { padding-top: 1rem; padding-bottom: 1.25rem; }
.hero {
  background: linear-gradient(135deg, rgba(242,140,82,0.16), rgba(242,140,82,0.07));
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 20px 22px;
  margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.05);
}
.metric-card, .result-card, .editor-card, .subtle-card {
  background: var(--card);
  border: 1px solid rgba(255,255,255,0.78);
  border-radius: 20px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.05);
}
.metric-card { padding: 16px 18px; min-height: 116px; }
.result-card, .editor-card, .subtle-card { padding: 18px; margin-bottom: 14px; }
.mini-label { font-size: 0.84rem; color: var(--muted); margin-bottom: 4px; }
.big-number { font-size: 1.95rem; font-weight: 800; color: var(--ink); line-height: 1.05; }
.kicker { color: var(--melon-dark); font-weight: 800; letter-spacing: .03em; text-transform: uppercase; font-size: .8rem; }
.title { color: var(--ink); font-size: 2rem; font-weight: 900; margin: 0; }
.subtitle { color: #4b5563; margin-top: 8px; line-height: 1.6; font-size: 1rem; }
.badge, .badge-soft {
  display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700;
  margin-right: 6px; margin-bottom: 6px; font-size:.84rem;
}
.badge { background:rgba(242,140,82,0.13); color:#b45309; }
.badge-soft { background:rgba(17,24,39,0.06); color:#374151; }
.editor-name { font-weight: 900; color: #111827; font-size: 1.08rem; }
.editor-sub { color: var(--muted); font-size: .92rem; margin-top: 4px; margin-bottom: 10px; }
.readable {
  color: #1f2937; font-size: 1.02rem; line-height: 1.78; letter-spacing: 0.01em;
}
.readable p { margin: 0 0 10px 0; }
.section-title { font-size: 1.1rem; font-weight: 800; margin-bottom: 10px; color: #111827; }
.helper-note {
  border-left: 4px solid #fff; padding: 10px 12px; background: rgba(255,255,255,0.12);
  border-radius: 10px; color: white;
}
.sidebar-chip {
  display:inline-block; padding:5px 9px; border-radius:999px; margin: 0 6px 6px 0;
  background: rgba(255,255,255,0.16); color:white; font-weight:700; font-size:.8rem;
}
hr.soft {
  border: none; height: 1px;
  background: linear-gradient(90deg, rgba(242,140,82,0), rgba(242,140,82,.45), rgba(242,140,82,0));
  margin: 14px 0;
}
.stTabs [data-baseweb="tab-list"] button { font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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

MEMORY_SYNONYM_MAP = {
    "etkili": ["güçlü", "yerinde", "karşılık bulan", "iyi kurulan"],
    "güzel": ["yerinde", "temiz", "olgun", "dengeli"],
    "kompozisyon": ["kadraj düzeni", "yerleşim", "görsel yapı"],
    "fotoğraf": ["kare", "görüntü", "sahne"],
    "anlatı": ["hikâye", "duygu hattı", "görsel cümle"],
}

STOPWORDS = {
    "ve", "ile", "ama", "bu", "bir", "çok", "daha", "gibi", "için", "olan", "göre", "olarak",
    "fotoğraf", "kare", "sahne", "özne", "ana", "olan", "güçlü", "güzel", "etkili"
}


def init_state() -> None:
    if "editor_memory" not in st.session_state:
        st.session_state.editor_memory = {
            key: {"reviews": [], "words": Counter()} for key in EDITORS
        }


def reset_editor_memory() -> None:
    st.session_state.editor_memory = {
        key: {"reviews": [], "words": Counter()} for key in EDITORS
    }


def tokenize_text(text: str) -> List[str]:
    words = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", text.lower())
    return [w for w in words if len(w) > 3 and w not in STOPWORDS]


def apply_memory_variation(text: str, editor_key: str, enabled: bool = True) -> str:
    if not enabled:
        return text
    memory = st.session_state.editor_memory.get(editor_key, {"words": Counter()})
    frequent = {w for w, c in memory.get("words", Counter()).most_common(12) if c >= 2}
    updated = text
    for word, replacements in MEMORY_SYNONYM_MAP.items():
        if word in frequent:
            replacement = random.choice(replacements)
            updated = re.sub(rf"\b{word}\b", replacement, updated, flags=re.IGNORECASE)
    return updated


def store_editor_review(editor_key: str, text: str) -> None:
    memory = st.session_state.editor_memory.setdefault(editor_key, {"reviews": [], "words": Counter()})
    memory["reviews"].append(text)
    memory["reviews"] = memory["reviews"][-12:]
    memory["words"].update(tokenize_text(text))


def image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    if img.mode in ("RGBA", "LA") and fmt.upper() == "JPEG":
        img = img.convert("RGB")
    img.save(buffer, format=fmt, quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_from_bytes(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
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
    return float(ImageStat.Stat(img.convert("L")).mean[0])


def contrast_score(img: Image.Image) -> float:
    return float(min(100, ImageStat.Stat(img.convert("L")).stddev[0] * 1.6))


def saturation_score(img: Image.Image) -> float:
    rgb = img.convert("RGB").resize((160, 160))
    sats = []
    for r, g, b in list(rgb.getdata()):
        mx, mn = max(r, g, b), min(r, g, b)
        sats.append(0 if mx == 0 else (mx - mn) / mx)
    return float(sum(sats) / max(len(sats), 1) * 100)


def edge_density(img: Image.Image) -> float:
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(min(100, ImageStat.Stat(edges).mean[0] * 1.6))


def subject_box_guess(img: Image.Image) -> Tuple[int, int, int, int]:
    w, h = img.size
    return int(w * 0.22), int(h * 0.18), int(w * 0.78), int(h * 0.82)


def downsample_for_analysis(img: Image.Image, max_side: int = 1400) -> Image.Image:
    copy = img.copy()
    copy.thumbnail((max_side, max_side))
    return copy


def scene_tags_heuristic(img: Image.Image) -> List[str]:
    ratio = img.size[0] / max(img.size[1], 1)
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
    ratio = img.size[0] / max(img.size[1], 1)
    bright = avg_brightness(img)
    contrast = contrast_score(img)
    sat = saturation_score(img)
    edge = edge_density(img)
    scores = {"Sokak": 0.0, "Portre": 0.0, "Belgesel": 0.0, "Mimari": 0.0, "Manzara": 0.0, "Minimal": 0.0, "Gece": 0.0}
    if edge > 42:
        scores["Sokak"] += 24; scores["Belgesel"] += 20; scores["Mimari"] += 24
    if sat < 20:
        scores["Minimal"] += 20; scores["Belgesel"] += 8
    if ratio > 1.35:
        scores["Manzara"] += 26; scores["Mimari"] += 10
    if 0.8 < ratio < 1.25:
        scores["Portre"] += 16; scores["Sokak"] += 10
    if bright < 75:
        scores["Gece"] += 30; scores["Sokak"] += 8
    if contrast > 48:
        scores["Belgesel"] += 12; scores["Sokak"] += 8; scores["Gece"] += 4
    if edge < 28 and sat < 25:
        scores["Minimal"] += 22; scores["Portre"] += 8
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
    points = [(int(w * 0.12), int(h * 0.68)), (int(w * 0.35), int(h * 0.56)), (int(w * 0.52), int(h * 0.48)), (int(w * 0.68), int(h * 0.42)), (int(w * 0.82), int(h * 0.38))]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=(64, 191, 255, 230), width=8)
        x, y = points[i + 1]
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(64, 191, 255, 220))

    dis = base.copy()
    draw = ImageDraw.Draw(dis, "RGBA")
    regions = [(int(w * 0.02), int(h * 0.05), int(w * 0.22), int(h * 0.24)), (int(w * 0.78), int(h * 0.02), int(w * 0.98), int(h * 0.18)), (int(w * 0.80), int(h * 0.72), int(w * 0.98), int(h * 0.96))]
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
    return {"Hikâye": story, "Kompozisyon": composition, "Işık": light, "Tonlama": tonal, "Özne Gücü": focus}


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
        "Sokak": {"tek_hamle": "arka planın dikkat çalan parlak bölgelerini bastırıp ana özneyi lokal kontrastla ayır", "crop": "kenarlardaki gereksiz ağırlığı %5–8 kırp", "local": "öznenin yüz/beden hattına hafif dodge, çevreye yumuşak burn uygula"},
        "Portre": {"tek_hamle": "yüzü sahnenin en kontrollü parlak alanı yap", "crop": "göz hizasını üst üçte bire yaklaştır", "local": "göz, yüz ve el bölgelerinde yumuşak lokal aydınlatma kullan"},
        "Belgesel": {"tek_hamle": "hikâyeyi dağıtan ikinci parlak odağı bastır", "crop": "anlatıya hizmet etmeyen boşluğu azalt", "local": "ana olay çevresinde mikro kontrastı artır"},
        "Mimari": {"tek_hamle": "çizgisel yapıyı ton ayrımıyla daha net vurgula", "crop": "eğim hissi veren boş kenarları temizle", "local": "yapısal yüzeylere texture ve kontrollü clarity ver"},
        "Manzara": {"tek_hamle": "ön plan-orta plan-arka plan katmanlarını tonla ayır", "crop": "ufuk hattını daha kararlı konuma taşı", "local": "gökyüzü ve ön plan için ayrı maske kullan"},
        "Minimal": {"tek_hamle": "sahneyi iki ya da üç ana tona indirerek sadeliği güçlendir", "crop": "fazla bilgiyi cesurca kes", "local": "tek özne dışındaki alanlarda dikkat azalt"},
        "Gece": {"tek_hamle": "gürültüyü artırmadan özneyi karanlıktan ayır", "crop": "parazit oluşturan köşe ışıklarını temizle", "local": "ışık kaynaklarının çevresini kontrol ederek glow’u sınırlı tut"},
    }
    rule = type_rules.get(photo_type, type_rules["Sokak"])
    return {
        "temel": {"Pozlama": exposure, "Highlights": highlights, "Shadows": shadows, "Whites": whites, "Blacks": blacks, "Clarity": clarity, "Texture": texture, "Dehaze": dehaze, "Vibrance": vibrance, "Vignette": vignette},
        "tek_hamle": rule["tek_hamle"],
        "kırpma": rule["crop"],
        "lokal": rule["local"],
        "akış": ["Önce global pozlama ve highlight/shadow dengesini kur.", "Sonra ana özne ile arka planı lokal maske ile ayır.", "En sonda dikkat dağıtan parlaklıkları ve köşeleri sakinleştir."],
    }


def build_core_analysis(photo_type: str, img: Image.Image, scores: Dict[str, int], recipe: Dict[str, Any]) -> Dict[str, str]:
    bright = avg_brightness(img)
    contrast = contrast_score(img)
    sat = saturation_score(img)
    edge = edge_density(img)
    composition = random.choice([
        "Kadraj yerleşimi sahneyi taşımayı başarıyor" if scores["Kompozisyon"] >= 78 else "Kadraj kararında küçük bir sıkışma hissi var",
        "Kompozisyon ana karar noktasını fena kurmuyor" if scores["Kompozisyon"] >= 78 else "Yerleşim tam oturmadığı için sahnenin etkisi bölünüyor",
        "Yerleşim kurgusu özneye doğru çalışan bir yapı oluşturuyor" if scores["Kompozisyon"] >= 78 else "Kompozisyon tarafında biraz daha sade bir karar fotoğrafı güçlendirebilirdi",
    ])
    story = random.choice([
        "Fotoğrafın bir anlatı kurma isteği belirgin" if scores["Hikâye"] >= 76 else "Anlatı var ama tam kristalleşmiyor",
        "Sahne kendi hikâyesini izleyiciye geçirebiliyor" if scores["Hikâye"] >= 76 else "Hikâye duygusu kuruluyor, fakat merkez daha net seçilebilirdi",
        "Karede bir duygu ve anlatı hattı kurulmuş durumda" if scores["Hikâye"] >= 76 else "Fotoğraf bir şey söylüyor ama sesi yer yer dağılabiliyor",
    ])
    light = random.choice([
        "Işık özneyi destekleyen bir denge kuruyor" if scores["Işık"] >= 74 else "Işık tarafında özneyi daha belirgin ayıracak bir karar gerekebilirdi",
        "Işık kullanımı sahnenin atmosferini taşıyor" if scores["Işık"] >= 74 else "Parlak bölgeler ana anlatıyla yarışıyor",
        "Parlaklık ilişkisi fotoğrafın kararını destekliyor" if scores["Işık"] >= 74 else "Ton dengesinde biraz daha kontrollü bir ışık dağılımı iyi olurdu",
    ])
    tones = random.choice([
        "Ton geçişleri görüntünün ritmini taşıyor" if scores["Tonlama"] >= 74 else "Ton tarafında biraz daha sadeleştirme fotoğrafı yükseltebilirdi",
        "Tonlama sahnenin duygusuna karşılık veriyor" if scores["Tonlama"] >= 74 else "Orta ton yoğunluğu dikkat akışını dağıtıyor",
        "Işık ve gölge arasındaki ilişki okunabilir durumda" if scores["Tonlama"] >= 74 else "Tonlama kararları özneyi yeterince yalnız bırakmıyor",
    ])
    subject = random.choice([
        "Ana özne sahnenin yükünü taşımayı başarıyor" if scores["Özne Gücü"] >= 74 else "Ana özne biraz daha belirginleştirilebilirdi",
        "Özne yeterince görünür ve tutarlı duruyor" if scores["Özne Gücü"] >= 74 else "Bakışın ilk tutunduğu alan daha kararlı seçilmeliydi",
        "Bakışın toplandığı merkez büyük ölçüde belli" if scores["Özne Gücü"] >= 74 else "Özne çevresiyle rekabet ediyor",
    ])
    relation = random.choice(["Figürler veya öğeler arasında bir bağ hissi oluşuyor", "Sahnedeki unsurlar birbirini destekleyen bir ilişki kuruyor", "Görsel yapı ile duygu hattı arasında bir temas var"])
    emotion = random.choice(["yalnızlık ve direnç", "yakınlık ve aidiyet", "sessizlik ve bekleyiş", "hareket ile durağanlık arasındaki gerilim", "tanıklık duygusu"])
    texture = "Dokular ve yüzeyler sahnenin ikincil gücüne katkı sağlıyor" if edge > 42 else "Doku tarafı geri planda ama sahneyi boğmuyor"
    lines = "Çizgisel akış bakışı belirli bir yöne taşıyor" if contrast > 40 else "Çizgiler yeterince baskın olmasa da sahnenin düzenini destekliyor"
    atmosphere = "Atmosfer fotoğrafın duygusunu kuruyor" if bright < 110 or sat < 22 else "Atmosfer daha çok ışık ve yerleşim kararından besleniyor"
    reflection = "Yansıma ve yüzey ilişkileri sahneye ikinci bir katman ekliyor" if sat > 18 and bright > 95 else "Yansıma etkisi bu karede ana unsur değil"
    gesture = "Beden dili sahnenin duygusunu güçlendiriyor" if scores["Hikâye"] > 72 else "Beden dili daha okunur olsaydı anlatı kuvvetlenebilirdi"
    visual_flow = "Göz akışı fotoğraf içinde belirli bir rota izliyor" if scores["Kompozisyon"] > 72 else "Göz akışı yer yer dağılma eğilimi gösteriyor"
    return {
        "photo_type": photo_type, "composition": composition, "story": story, "light": light, "tones": tones, "subject": subject,
        "relationship": relation, "emotion": emotion, "texture": texture, "lines": lines, "atmosphere": atmosphere,
        "reflection": reflection, "gesture": gesture, "visual_flow": visual_flow, "one_move": recipe["tek_hamle"],
    }


def choose_variant(word: str, seen: set) -> str:
    options = SYNONYMS.get(word, [word])
    random.shuffle(options)
    for candidate in options:
        if candidate not in seen:
            seen.add(candidate)
            return candidate
    seen.add(word)
    return word


def shorten_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .") + "."


def write_as_selahattin(core: Dict[str, str], seen: set) -> str:
    opener = random.choice([shorten_sentence(core["composition"]), shorten_sentence(core["subject"]), shorten_sentence(core["light"])])
    structure = [shorten_sentence(core["visual_flow"] + " ve bu durum özneye doğru çalışan bir dil kuruyor"), shorten_sentence(core["story"])]
    cultural = random.choice([
        "Fotoğraf, görsel kararının ötesinde bir hafızaya da dokunuyor.",
        "Sahnenin taşıdığı hava yalnız bugünü değil, geçmişi de çağırıyor.",
        "Bu kare, yalnızca görüleni değil, arkasındaki kültürel sesi de hissettiriyor.",
    ])
    imagination = random.choice([
        "Zaman burada biraz yavaşlıyor; mekân yalnızca fon değil, anlatının sessiz ortağı gibi duruyor.",
        "Figür ya da ana öğe, sanki sahnenin içinden geçip başka bir zamana doğru ilerliyormuş hissi bırakıyor.",
        "Mekânın kendisi de fotoğrafın öznesine eşlik eden ikinci bir karakter gibi davranıyor.",
        "",
    ])
    closer = random.choice([
        f"Bu haliyle de {choose_variant('etkili', seen)} bir fotoğraf.",
        "Birkaç küçük karar daha netleştiğinde çok daha kalıcı bir karşılık bulabilir.",
        "Yine de fotoğrafın kurduğu dil izleyicide karşılık buluyor.",
    ])
    return " ".join([p for p in [opener, structure[0], structure[1], cultural, imagination, closer] if p])


def write_as_guler(core: Dict[str, str], seen: set) -> str:
    lead = random.choice([
        f"{core['relationship']}; bu durum fotoğrafın {core['emotion']} duygusunu güçlendiriyor.",
        f"Fotoğrafta {core['emotion']} hissi belirgin ve bu duygu sahnedeki öğeler arasındaki bağla büyüyor.",
        "Sahnedeki ilişki, fotoğrafın duygusal ağırlığını belirleyen en önemli unsur gibi duruyor.",
    ])
    middle = random.choice([
        f"{core['composition']}, {core['light'].lower()} ve {core['tones'].lower()}.",
        f"{core['texture']}, {core['lines'].lower()} ve {core['atmosphere'].lower()}.",
        f"{core['gesture']}, {core['composition'].lower()} ve {core['tones'].lower()}.",
    ])
    closer = random.choice([
        "Bütün bunlar bir araya gelince sahne hem duygusal hem görsel olarak karşılık buluyor.",
        f"Bu yüzden kare {choose_variant('etkili', seen)} bir bütünlük taşıyor.",
        "Sonuçta fotoğrafın hem duygusu hem de görsel yapısı birbirini destekliyor.",
    ])
    return " ".join([lead, middle, closer])


def write_as_sevgin(core: Dict[str, str], seen: set) -> str:
    start = random.choice(["Etkili kompozisyon.", "Güzel bir kadraj düzeni.", "Yerleşim kararı başarılı."])
    extra = random.choice([core["gesture"], core["tones"], core["atmosphere"], core["subject"]])
    closer = random.choice(["Güzel bir kare.", "Emeğinize sağlık.", f"Sonuç olarak {choose_variant('etkili', seen)} bir fotoğraf."])
    return " ".join([start, shorten_sentence(extra), closer])


def generate_editor_reviews(core: Dict[str, str], use_memory: bool = True) -> Dict[str, str]:
    seen: set = set()
    reviews = {
        "selahattin": write_as_selahattin(core, seen),
        "guler": write_as_guler(core, seen),
        "sevgin": write_as_sevgin(core, seen),
    }
    final_reviews = {}
    for key, text in reviews.items():
        varied = apply_memory_variation(text, key, enabled=use_memory)
        final_reviews[key] = varied
    return final_reviews


def build_debate(core: Dict[str, str], photo_type: str) -> List[Tuple[str, str]]:
    return [
        (EDITORS["selahattin"]["name"], random.choice([f"{core['composition']} Bu yüzden ilk karar noktası kadraj tarafında beliriyor.", f"{core['story']} Yine de sahnenin dili biraz daha seçici kurulabilirdi."])),
        (EDITORS["guler"]["name"], random.choice([f"Benim için asıl güç {core['relationship'].lower()} ve fotoğrafın {core['emotion']} duygusunu taşıması.", "Sahnenin duygusal ağırlığı, teknik küçük kusurlardan daha baskın hissediliyor."])),
        (EDITORS["sevgin"]["name"], random.choice(["Genel etki güçlü. Özellikle yerleşim ve atmosfer birlikte çalışıyor.", f"Kısa bakışta bile {photo_type.lower()} karakteri net hissediliyor."])),
    ]


def default_analysis(img: Image.Image, use_memory: bool = True) -> Dict[str, Any]:
    photo_type, confidence, raw_scores = classify_photo_type_heuristic(img)
    tags = scene_tags_heuristic(img)
    scores = score_bundle(img)
    recipe = build_recipe(photo_type, img, scores)
    core = build_core_analysis(photo_type, img, scores, recipe)
    editor_reviews = generate_editor_reviews(core, use_memory=use_memory)
    strong = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
    weak = sorted(scores.items(), key=lambda x: x[1])[:2]
    return {
        "analysis_source": "heuristic",
        "photo_type": photo_type,
        "confidence": round(confidence, 1),
        "scene_tags": tags,
        "scores": scores,
        "strong_sides": [f"{k} güçlü görünüyor" for k, _ in strong],
        "risks": [f"{k} tarafında toparlama gerekiyor" for k, _ in weak],
        "summary": [core["story"], core["composition"], f"En etkili düzenleme yönü: {recipe['tek_hamle']}."],
        "one_move": recipe["tek_hamle"],
        "recipe": recipe,
        "raw_type_scores": raw_scores,
        "core_analysis": core,
        "editor_reviews": editor_reviews,
        "debate": build_debate(core, photo_type),
    }


def normalize_vision_payload(data: Dict[str, Any], fallback_img: Image.Image, use_memory: bool = True) -> Dict[str, Any]:
    fallback = default_analysis(fallback_img, use_memory=use_memory)
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
    merged["editor_reviews"] = generate_editor_reviews(core, use_memory=use_memory)
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
  "scores": {"Hikâye": 0-100, "Kompozisyon": 0-100, "Işık": 0-100, "Tonlama": 0-100, "Özne Gücü": 0-100},
  "strong_sides": ["madde", "madde", "madde"],
  "risks": ["madde", "madde", "madde"],
  "summary": ["ortak yorum 1", "ortak yorum 2", "ortak yorum 3"],
  "one_move": "tek hamlede en etkili öneri",
  "recipe": {"temel": {"Pozlama": "+0.20", "Highlights": "-25", "Shadows": "+20", "Whites": "+5", "Blacks": "-10", "Clarity": "+6", "Texture": "+8", "Dehaze": "+4", "Vibrance": "+3", "Vignette": "-8"}, "tek_hamle": "tek hamle önerisi", "kırpma": "kırpma önerisi", "lokal": "lokal düzenleme önerisi", "akış": ["adım1", "adım2", "adım3"]}
}
Kurallar:
- Klişe cümle üretme.
- Sahneye özgü konuş.
- JSON dışında hiçbir şey yazma.
""".strip()


def analyze_with_openai(img: Image.Image, model: str, use_memory: bool = True) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not OPENAI_AVAILABLE or not api_key:
        raise RuntimeError("OpenAI istemcisi veya OPENAI_API_KEY bulunamadı.")
    client = OpenAI(api_key=api_key)
    b64 = image_to_base64(img, fmt="JPEG")
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": vision_prompt()}, {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}]}],
        temperature=0.2,
        max_output_tokens=2200,
    )
    text = getattr(response, "output_text", "") or ""
    data = safe_json_loads(text)
    if not data:
        raise RuntimeError("Vision model geçerli JSON döndürmedi.")
    return normalize_vision_payload(data, img, use_memory=use_memory)


@st.cache_data(show_spinner=False)
def cached_overlay_bytes(file_hash: str, image_bytes: bytes) -> Dict[str, bytes]:
    img = pil_from_bytes(image_bytes)
    overlays = overlay_analysis(img)
    out = {}
    for name, overlay in overlays.items():
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        out[name] = buf.getvalue()
    return out


@st.cache_data(show_spinner=False)
def cached_default_analysis(file_hash: str, image_bytes: bytes) -> Dict[str, Any]:
    img = downsample_for_analysis(pil_from_bytes(image_bytes))
    return default_analysis(img, use_memory=False)


def memory_stats(editor_key: str) -> List[Tuple[str, int]]:
    words = st.session_state.editor_memory.get(editor_key, {}).get("words", Counter())
    return words.most_common(6)


def render_editor_card(meta: Dict[str, Any], review: str) -> None:
    badges_html = "".join([f"<span class='badge-soft'>{b}</span>" for b in meta["badges"]])
    st.markdown("<div class='editor-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='editor-name'>✍️ {meta['name']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='editor-sub'>{meta['subtitle']}</div>", unsafe_allow_html=True)
    st.markdown(badges_html, unsafe_allow_html=True)
    st.markdown(f"<div class='readable'><p>{review}</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


init_state()

with st.sidebar:
    st.markdown("## ÇOFSAT PRO V3")
    st.markdown("Çekirdek analiz + editör kadrosu + reçete")
    analysis_engine = st.radio("Analiz motoru", ["Akıllı Yerel Motor", "OpenAI Vision"], index=0 if not os.getenv("OPENAI_API_KEY") else 1)
    model_name = st.text_input("Vision model", value="gpt-5.4-mini", help="OpenAI Vision seçiliyse kullanılacak model adı.")

    st.markdown("---")
    st.markdown("### Görsel katmanlar")
    show_focus = st.checkbox("Odak noktası", value=True)
    show_eye = st.checkbox("Göz akışı", value=True)
    show_distraction = st.checkbox("Dikkat dağıtan alanlar", value=True)
    show_heat = st.checkbox("Isı haritası", value=True)

    st.markdown("---")
    st.markdown("### ÇOFSAT Editörleri")
    mode_all = st.checkbox("Tüm editörleri göster", value=True)
    selected_editors = []
    if mode_all:
        selected_editors = list(EDITORS.keys())
        st.markdown("<span class='sidebar-chip'>Selahattin</span><span class='sidebar-chip'>Güler</span><span class='sidebar-chip'>Sevgin</span>", unsafe_allow_html=True)
    else:
        if st.checkbox("Selahattin Kalaycı", value=True):
            selected_editors.append("selahattin")
        if st.checkbox("Güler Ataşer", value=True):
            selected_editors.append("guler")
        if st.checkbox("Sevgin Cingöz", value=True):
            selected_editors.append("sevgin")

    use_memory = st.checkbox("Editör hafızasını kullan", value=True)
    show_memory_panel = st.checkbox("Hafıza durumunu göster", value=False)
    if st.button("Editör hafızasını sıfırla", use_container_width=True):
        reset_editor_memory()
        st.success("Editör hafızası sıfırlandı.")

    st.markdown("---")
    st.markdown("<div class='helper-note'>OpenAI Vision seçeneği için <b>OPENAI_API_KEY</b> tanımlı olmalı. Anahtar yoksa uygulama yerel motora döner. Hız için analiz düşük çözünürlüklü önizleme üzerinde yapılır; fotoğrafın görüntüsü tam boy gösterilir.</div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="kicker">Çofsat Pro V3</div>
  <h1 class="title">Bozulmadan büyüyen arayüz</h1>
  <div class="subtitle">Bu sürüm, önceki yapıyı korur. Çekirdek analiz, görsel okuma katmanları, reçete paneli ve editör sistemi aynı düzen içinde çalışır. Yazılar daha okunur, hız daha dengeli, editör erişimi ise sol kavuniçi sütunda toplanır.</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Fotoğraf yükle", type=["jpg", "jpeg", "png", "webp"])
if not uploaded:
    st.info("Başlamak için bir fotoğraf yükleyin.")
    st.stop()

image_bytes = uploaded.getvalue()
file_hash = hashlib.md5(image_bytes).hexdigest()
image = pil_from_bytes(image_bytes)
overlays_png = cached_overlay_bytes(file_hash, image_bytes)
overlays = {name: Image.open(io.BytesIO(data)) for name, data in overlays_png.items()}

col_left, col_right = st.columns([1.2, 1], gap="large")
with col_left:
    st.markdown("### Fotoğraf")
    st.image(image, use_container_width=True)
with col_right:
    st.markdown("### Görsel okuma katmanları")
    selected_layers = []
    if show_focus: selected_layers.append("Odak Noktası")
    if show_eye: selected_layers.append("Göz Akışı")
    if show_distraction: selected_layers.append("Dikkat Dağıtan Alanlar")
    if show_heat: selected_layers.append("Görsel Ağırlık / Isı Haritası")
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
    preview_img = downsample_for_analysis(image)
    if analysis_engine == "OpenAI Vision":
        try:
            analysis = analyze_with_openai(preview_img, model_name, use_memory=use_memory)
        except Exception as e:
            st.warning(f"Vision model kullanılamadı. Yerel motora dönüldü. Neden: {e}")
            analysis = cached_default_analysis(file_hash, image_bytes)
    else:
        analysis = cached_default_analysis(file_hash, image_bytes)

    # cache sonuçları hafızasız döner; bu aşamada hafızalı varyasyonu uygula
    if use_memory:
        analysis["editor_reviews"] = generate_editor_reviews(analysis["core_analysis"], use_memory=True)

scores = analysis.get("scores", {})
recipe = analysis.get("recipe", {})
scene_tags = analysis.get("scene_tags", [])
editor_reviews = analysis.get("editor_reviews", {})

if use_memory:
    for key, text in editor_reviews.items():
        store_editor_review(key, text)

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
        st.markdown(f"<div class='metric-card'><div class='mini-label'>{label}</div><div style='font-weight:800;color:#111827'>{val}</div></div>", unsafe_allow_html=True)

if scene_tags:
    st.markdown("### Sahne Etiketleri")
    st.markdown("".join([f"<span class='badge'>{t}</span>" for t in scene_tags]), unsafe_allow_html=True)

st.markdown("### Karar Paneli")
score_cols = st.columns(len(scores) if scores else 5)
for col, (name, value) in zip(score_cols, scores.items()):
    with col:
        st.markdown(f"<div class='metric-card'><div class='mini-label'>{name}</div><div class='big-number'>{value}</div></div>", unsafe_allow_html=True)

body_left, body_right = st.columns([1, 1], gap="large")
with body_left:
    st.markdown("### Ortak Analiz")
    st.markdown("<div class='result-card'><div class='readable'>", unsafe_allow_html=True)
    for line in analysis.get("summary", []):
        st.markdown(f"<p>• {line}</p>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("### Güçlü Yanlar")
    st.markdown("<div class='result-card'><div class='readable'>", unsafe_allow_html=True)
    for item in analysis.get("strong_sides", []):
        st.markdown(f"<p>✅ {item}</p>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("### Riskler")
    st.markdown("<div class='result-card'><div class='readable'>", unsafe_allow_html=True)
    for item in analysis.get("risks", []):
        st.markdown(f"<p>⚠️ {item}</p>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

with body_right:
    st.markdown("### Fotoğrafa Özel Düzenleme Reçetesi")
    st.markdown("<div class='result-card'><div class='readable'>", unsafe_allow_html=True)
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
            st.markdown(f"<p>{i}. {step}</p>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("### ÇOFSAT Editör Kadrosu")
if not selected_editors:
    st.info("Sol sütundan en az bir editör seçin.")
else:
    for key in selected_editors:
        render_editor_card(EDITORS[key], editor_reviews.get(key, "Yorum üretilemedi."))

st.markdown("### Editörler Ne Diyor?")
st.markdown("<div class='result-card'><div class='readable'>", unsafe_allow_html=True)
for name, line in analysis.get("debate", []):
    st.markdown(f"<p><strong>{name}:</strong> {line}</p>", unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)

if show_memory_panel:
    st.markdown("### Editör Hafızası")
    cols = st.columns(3)
    for col, key in zip(cols, ["selahattin", "guler", "sevgin"]):
        with col:
            stats = memory_stats(key)
            st.markdown("<div class='subtle-card'>", unsafe_allow_html=True)
            st.markdown(f"**{EDITORS[key]['name']}**")
            if stats:
                for word, count in stats:
                    st.markdown(f"- {word}: {count}")
            else:
                st.markdown("Henüz hafıza oluşmadı.")
            st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Çekirdek analiz verisi"):
    st.json(analysis.get("core_analysis", {}))
with st.expander("Ham analiz verisi"):
    st.json(analysis)

st.caption("Not: Bu sürümde temel arayüz korunur. Editör sistemi sol kavuniçi sütundan yönetilir. Yazılar okunurluk için büyütüldü, satır aralıkları açıldı ve analiz önizleme çözünürlüğü üzerinden hızlandırıldı.")
