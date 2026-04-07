import os
import io
import json
import base64
from typing import Dict, Any, List, Tuple

import streamlit as st
from PIL import Image, ImageOps, ImageStat, ImageFilter, ImageEnhance, ImageDraw

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
    page_title="ÇOFSAT PRO Vision",
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
  --card: rgba(255,255,255,0.78);
  --border: rgba(242,140,82,0.25);
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
.block-container { padding-top: 1.1rem; padding-bottom: 1.4rem; }
.hero {
  background: linear-gradient(135deg, rgba(242,140,82,0.18), rgba(242,140,82,0.08));
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 20px 22px;
  margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}
.metric-card, .glass, .result-card {
  background: var(--card);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.7);
  border-radius: 22px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}
.metric-card {
  padding: 16px 18px;
  min-height: 118px;
}
.result-card {
  padding: 18px;
  margin-bottom: 14px;
}
.mini-label {
  font-size: 0.84rem;
  color: #6b7280;
  margin-bottom: 4px;
}
.big-number {
  font-size: 2rem;
  font-weight: 800;
  color: var(--ink);
  line-height: 1.05;
}
.kicker {
  color: var(--melon);
  font-weight: 800;
  letter-spacing: .03em;
  text-transform: uppercase;
  font-size: .8rem;
}
.title {
  color: var(--ink);
  font-size: 2rem;
  font-weight: 900;
  margin: 0;
}
.subtitle {
  color: #4b5563;
  margin-top: 8px;
}
.badge {
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  background:rgba(242,140,82,0.13);
  color:#b45309;
  font-weight:700;
  margin-right: 6px;
  margin-bottom: 6px;
  font-size:.85rem;
}
.section-title {
  font-size: 1.15rem;
  font-weight: 800;
  margin-bottom: 10px;
  color: #111827;
}
.photo-frame {
  border-radius: 22px;
  overflow: hidden;
  border: 1px solid rgba(0,0,0,0.06);
  background: white;
}
.helper-note {
  border-left: 4px solid var(--melon);
  padding: 10px 12px;
  background: rgba(242,140,82,0.08);
  border-radius: 10px;
  color: #7c2d12;
}
hr.soft {
  border: none;
  height: 1px;
  background: linear-gradient(90deg, rgba(242,140,82,0), rgba(242,140,82,.4), rgba(242,140,82,0));
  margin: 14px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
    stat = ImageStat.Stat(gray)
    return float(stat.mean[0])


def contrast_score(img: Image.Image) -> float:
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    return float(min(100, stat.stddev[0] * 1.6))


def saturation_score(img: Image.Image) -> float:
    rgb = img.convert("RGB")
    pixels = list(rgb.resize((160, 160)).getdata())
    sats = []
    for r, g, b in pixels:
        mx, mn = max(r, g, b), min(r, g, b)
        sats.append(0 if mx == 0 else (mx - mn) / mx)
    return float(sum(sats) / len(sats) * 100)


def edge_density(img: Image.Image) -> float:
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edges)
    return float(min(100, stat.mean[0] * 1.6))


def subject_box_guess(img: Image.Image) -> Tuple[int, int, int, int]:
    # Heuristic subject box using central weighted crop
    w, h = img.size
    left = int(w * 0.22)
    top = int(h * 0.18)
    right = int(w * 0.78)
    bottom = int(h * 0.82)
    return left, top, right, bottom


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

    # focus overlay
    focus = base.copy()
    draw = ImageDraw.Draw(focus, "RGBA")
    box = subject_box_guess(img)
    draw.rounded_rectangle(box, radius=18, outline=(255, 188, 66, 255), width=6)
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    draw.ellipse((cx-16, cy-16, cx+16, cy+16), outline=(255, 80, 80, 255), width=5)

    # eye path overlay
    eye = base.copy()
    draw = ImageDraw.Draw(eye, "RGBA")
    points = [
        (int(w*0.12), int(h*0.68)),
        (int(w*0.35), int(h*0.56)),
        (int(w*0.52), int(h*0.48)),
        (int(w*0.68), int(h*0.42)),
        (int(w*0.82), int(h*0.38)),
    ]
    for i in range(len(points)-1):
        draw.line([points[i], points[i+1]], fill=(64, 191, 255, 230), width=8)
        x, y = points[i+1]
        draw.ellipse((x-10, y-10, x+10, y+10), fill=(64, 191, 255, 220))

    # distraction overlay
    dis = base.copy()
    draw = ImageDraw.Draw(dis, "RGBA")
    regions = [
        (int(w*0.02), int(h*0.05), int(w*0.22), int(h*0.24)),
        (int(w*0.78), int(h*0.02), int(w*0.98), int(h*0.18)),
        (int(w*0.80), int(h*0.72), int(w*0.98), int(h*0.96)),
    ]
    for r in regions:
        draw.rounded_rectangle(r, radius=14, outline=(255, 72, 72, 255), width=5, fill=(255, 72, 72, 40))

    # heatmap overlay
    heat = base.copy()
    draw = ImageDraw.Draw(heat, "RGBA")
    heat_regions = [
        (int(w*0.28), int(h*0.28), int(w*0.72), int(h*0.78), (255, 160, 0, 55)),
        (int(w*0.34), int(h*0.34), int(w*0.66), int(h*0.70), (255, 70, 0, 70)),
        (int(w*0.42), int(h*0.40), int(w*0.58), int(h*0.60), (255, 0, 0, 85)),
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
    composition = int(max(44, min(94, 58 + (edge * 0.15) + (10 if 0.8 < img.size[0] / max(img.size[1],1) < 1.8 else 0))))
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


def editor_summary(mode: str, photo_type: str, scores: Dict[str, int], recipe: Dict[str, Any]) -> List[str]:
    ordered = sorted(scores.items(), key=lambda x: x[1])
    weak_key, weak_val = ordered[0]
    strong_key, strong_val = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0]

    if mode == "Sert Editör":
        return [
            f"Fotoğrafın en güçlü yanı {strong_key.lower()} tarafında; ama {weak_key.lower()} hâlâ görüntünün ağırlığını tam taşıyamıyor.",
            f"{photo_type.lower()} karakteri var, fakat ana karar noktası yeterince acımasız seçilmediği için etki dağılma riski taşıyor.",
            f"Bu kareyi ileri taşıyacak hamle net: {recipe.get('tek_hamle', 'ana özneyi daha baskın hale getirmek')}."
        ]
    if mode == "Minimal":
        return [
            f"Güçlü taraf: {strong_key}.",
            f"Zayıf halka: {weak_key}.",
            f"Tek hamle: {recipe.get('tek_hamle', 'tonal ayrımı sadeleştir')}"
        ]
    return [
        f"Bu kare {photo_type.lower()} hissi taşıyor ve özellikle {strong_key.lower()} tarafında izleyiciyi içine çekebilen bir temel kuruyor.",
        f"Şu anda en çok gelişme alanı {weak_key.lower()} bölümünde; orası toparlandığında fotoğrafın dili çok daha netleşir.",
        f"En etkili düzenleme yönü: {recipe.get('tek_hamle', 'ana özneyi öne çıkaran lokal tonlama')}"
    ]


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


def default_analysis(img: Image.Image, editor_mode: str) -> Dict[str, Any]:
    photo_type, confidence, raw_scores = classify_photo_type_heuristic(img)
    tags = scene_tags_heuristic(img)
    scores = score_bundle(img)
    recipe = build_recipe(photo_type, img, scores)
    summary = editor_summary(editor_mode, photo_type, scores, recipe)

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
        "summary": summary,
        "one_move": recipe["tek_hamle"],
        "recipe": recipe,
        "raw_type_scores": raw_scores,
    }


def vision_prompt(editor_mode: str) -> str:
    return f"""
Sen ÇOFSAT PRO isimli fotoğraf okuma motorusun.
Görev: Fotoğrafı derinlemesine analiz et ve SADECE geçerli JSON döndür.
Yorum dili: Türkçe.
Editör modu: {editor_mode}.

İstediğim alanlar:
{{
  "analysis_source": "vision",
  "photo_type": "Sokak|Portre|Belgesel|Mimari|Manzara|Minimal|Gece|Diğer",
  "confidence": 0-100,
  "scene_tags": ["etiket1", "etiket2", "etiket3"],
  "scores": {{
    "Hikâye": 0-100,
    "Kompozisyon": 0-100,
    "Işık": 0-100,
    "Tonlama": 0-100,
    "Özne Gücü": 0-100
  }},
  "strong_sides": ["madde", "madde", "madde"],
  "risks": ["madde", "madde", "madde"],
  "summary": ["cümle1", "cümle2", "cümle3"],
  "one_move": "tek hamlede en etkili öneri",
  "recipe": {{
    "temel": {{
      "Pozlama": "örnek +0.20",
      "Highlights": "örnek -25",
      "Shadows": "örnek +20",
      "Whites": "örnek +5",
      "Blacks": "örnek -10",
      "Clarity": "örnek +6",
      "Texture": "örnek +8",
      "Dehaze": "örnek +4",
      "Vibrance": "örnek +3",
      "Vignette": "örnek -8"
    }},
    "tek_hamle": "tek hamle önerisi",
    "kırpma": "kırpma önerisi",
    "lokal": "lokal düzenleme önerisi",
    "akış": ["adım1", "adım2", "adım3"]
  }}
}}

Kurallar:
- Genel ve tekrar eden klişe yorum üretme.
- Sahneye özgü konuş.
- Kısa ama keskin ol.
- JSON dışında hiçbir şey yazma.
""".strip()


def analyze_with_openai(img: Image.Image, editor_mode: str, model: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not OPENAI_AVAILABLE or not api_key:
        raise RuntimeError("OpenAI istemcisi veya OPENAI_API_KEY bulunamadı.")

    client = OpenAI(api_key=api_key)
    b64 = image_to_base64(img, fmt="JPEG")

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": vision_prompt(editor_mode)},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ],
            }
        ],
        temperature=0.3,
        max_output_tokens=2200,
    )

    text = getattr(response, "output_text", "") or ""
    data = safe_json_loads(text)
    if not data:
        raise RuntimeError("Vision model geçerli JSON döndürmedi.")
    return data


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## ÇOFSAT PRO Vision")
    st.markdown("Gerçek vision model entegrasyonu + sahneye özel eleştiri + düzenleme reçetesi")

    editor_mode = st.radio(
        "Editör karakteri",
        ["Eğitmen", "Sert Editör", "Minimal"],
        index=0,
    )

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
    st.markdown(
        "<div class='helper-note'>OpenAI Vision seçeneği için sistemde <b>OPENAI_API_KEY</b> tanımlı olmalı. Yoksa uygulama otomatik olarak yerel analiz motoruna döner.</div>",
        unsafe_allow_html=True,
    )


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="kicker">Çofsat Pro</div>
      <h1 class="title">Dünyada olmayan fotoğraf okuma sistemi</h1>
      <div class="subtitle">Fotoğraf türü tanıma, sahneye özel eleştiri motoru, görsel okuma katmanları ve fotoğrafa özel düzenleme reçetesi tek arayüzde.</div>
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
    analysis: Dict[str, Any]
    if analysis_engine == "OpenAI Vision":
        try:
            analysis = analyze_with_openai(image, editor_mode, model_name)
        except Exception as e:
            st.warning(f"Vision model kullanılamadı. Yerel motora dönüldü. Neden: {e}")
            analysis = default_analysis(image, editor_mode)
    else:
        analysis = default_analysis(image, editor_mode)

scores = analysis.get("scores", {})
recipe = analysis.get("recipe", {})
scene_tags = analysis.get("scene_tags", [])

# Top metrics
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

# Score cards
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
    st.markdown("### Editör Yorumu")
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

with st.expander("Ham analiz verisi"):
    st.json(analysis)

st.caption("Not: OpenAI Vision etkin değilse uygulama güçlü bir yerel analiz motoruyla çalışır. Vision model bağlandığında sahne okuması daha özgül ve fotoğrafa daha bağlı hale gelir.")
