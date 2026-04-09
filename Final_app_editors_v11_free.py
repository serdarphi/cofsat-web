import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageStat, ImageFilter, ImageEnhance, ImageDraw

try:
    import cv2
except Exception:
    cv2 = None

import streamlit as st

st.set_page_config(page_title="ÇOFSAT Fotoğraf Ön Değerlendirme v11 Free", layout="wide", page_icon="📷")

MAX_SIZE = 1400
EDITOR_NAMES = [
    "Selahattin Kalaycı",
    "Güler Ataşer",
    "Sevgin Cingöz",
    "Mürşide Çilengir",
    "Gülcan Ceylan Çağın",
]

EDITOR_PERSONAS = {
    "Selahattin Kalaycı": {
        "title": "Düşünsel okuma",
        "tone": "sorgulayıcı, ağırlıklı, karar arayan",
        "focus": ["niyet", "kadraj kararı", "görsel gerilim", "neden bu an"],
    },
    "Güler Ataşer": {
        "title": "Atmosfer okuması",
        "tone": "duyusal, şiirsel, yumuşak ama süssüz",
        "focus": ["ışık", "hava", "ton geçişi", "dokusal his"],
    },
    "Sevgin Cingöz": {
        "title": "Yapısal okuma",
        "tone": "analitik, kısa, iskelet gören",
        "focus": ["yerleşim", "denge", "göz akışı", "gereksiz yük"],
    },
    "Mürşide Çilengir": {
        "title": "İnsani okuma",
        "tone": "içten, empatik, insan merkezli",
        "focus": ["insan ilişkisi", "mesafe", "beden dili", "duygu"],
    },
    "Gülcan Ceylan Çağın": {
        "title": "Editoryal okuma",
        "tone": "soğukkanlı, profesyonel, seçici ama yapıcı",
        "focus": ["yayın potansiyeli", "seçki gücü", "netlik", "toparlama payı"],
    },
}

CSS = """
<style>
:root {
    --bg: #111827;
    --card: #172033;
    --card2: #1f2937;
    --accent: #f39c5e;
    --accent2: #ffdfbf;
    --text: #f7efe7;
    --muted: #d4c2b3;
    --ok: #79d9a7;
    --warn: #ffd166;
}
html, body, [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #0d1422 0%, #111827 100%); }
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; }
.hero {
    background: linear-gradient(135deg, rgba(243,156,94,.22), rgba(255,223,191,.08));
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 22px; padding: 1.1rem 1.2rem; margin-bottom: 1rem;
}
.hero h1 { color: var(--text); margin: 0; font-size: 2rem; }
.hero p { color: var(--muted); margin: .35rem 0 0 0; line-height: 1.6; }
.card {
    background: rgba(23,32,51,.88); border: 1px solid rgba(255,255,255,.08);
    border-radius: 22px; padding: 1rem 1rem; margin-bottom: .9rem;
    box-shadow: 0 10px 30px rgba(0,0,0,.18);
}
.card h3, .card h4, .card p, .card li, .card div { color: var(--text); }
.small { color: var(--muted); font-size: .94rem; line-height: 1.6; }
.badge {
 display:inline-block; padding:.28rem .62rem; border-radius:999px; background:rgba(243,156,94,.17);
 color: var(--accent2); border:1px solid rgba(243,156,94,.25); font-size:.82rem; margin-right:.4rem;
}
.statgrid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.7rem; }
.stat {
    background: rgba(255,255,255,.03); border-radius: 16px; padding: .8rem;
    border: 1px solid rgba(255,255,255,.06);
}
.stat .k { color: var(--muted); font-size: .85rem; }
.stat .v { color: var(--text); font-size: 1.35rem; font-weight: 700; margin-top: .15rem; }
.editor-title { display:flex; align-items:center; justify-content:space-between; gap:.8rem; }
.editor-title .name { font-size:1.05rem; font-weight:700; color:var(--text); }
.editor-title .role { color:var(--accent2); font-size:.84rem; }
.note { background: rgba(255,255,255,.03); border-radius: 16px; padding: .85rem .95rem; line-height: 1.78; }
.sectionhead { color: var(--accent2); font-weight: 700; letter-spacing:.01em; margin-bottom:.35rem; }
</style>
"""


@dataclass
class Metrics:
    width: int
    height: int
    brightness: float
    contrast: float
    focus: float
    edge_density: float
    symmetry: float
    center_x: float
    center_y: float
    negative_space: float
    highlight_clip: float
    shadow_clip: float


def safe_resize(img: Image.Image, max_size: int = MAX_SIZE) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_size:
        return img
    scale = max_size / float(longest)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)


def gray_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"))


def edge_density(gray: np.ndarray) -> float:
    if cv2 is not None:
        edges = cv2.Canny(gray, 100, 200)
        return float((edges > 0).mean())
    gy, gx = np.gradient(gray.astype(np.float32))
    mag = np.sqrt(gx**2 + gy**2)
    return float((mag > np.percentile(mag, 80)).mean())


def focus_score(gray: np.ndarray) -> float:
    if cv2 is not None:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gy, gx = np.gradient(gray.astype(np.float32))
    return float(np.var(gx) + np.var(gy))


def symmetry_score(gray: np.ndarray) -> float:
    h, w = gray.shape
    left = gray[:, :w//2]
    right = gray[:, w-left.shape[1]:]
    right = np.fliplr(right)
    diff = np.abs(left.astype(np.float32) - right.astype(np.float32)).mean() / 255.0
    return max(0.0, 1.0 - float(diff))


def center_of_mass(gray: np.ndarray) -> Tuple[float, float]:
    inv = 255 - gray.astype(np.float32)
    total = inv.sum() + 1e-9
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((xs * inv).sum() / total) / w
    cy = float((ys * inv).sum() / total) / h
    return cx, cy


def negative_space_score(gray: np.ndarray) -> float:
    small = np.array(Image.fromarray(gray).resize((96, 96))).astype(np.float32)
    gy, gx = np.gradient(small)
    mag = np.sqrt(gx**2 + gy**2)
    return float((mag < np.percentile(mag, 35)).mean())


def extract_metrics(image_bytes: bytes) -> Metrics:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    img = safe_resize(img)
    gray = gray_np(img)
    stat = ImageStat.Stat(img.convert("L"))
    cx, cy = center_of_mass(gray)
    return Metrics(
        width=img.width,
        height=img.height,
        brightness=float(stat.mean[0]),
        contrast=float(stat.stddev[0]),
        focus=float(focus_score(gray)),
        edge_density=float(edge_density(gray)),
        symmetry=float(symmetry_score(gray)),
        center_x=cx,
        center_y=cy,
        negative_space=float(negative_space_score(gray)),
        highlight_clip=float((gray >= 250).mean()),
        shadow_clip=float((gray <= 5).mean()),
    )


def classify_mode(m: Metrics) -> Tuple[str, str]:
    if m.negative_space > 0.58 and m.symmetry > 0.72:
        return "Soyut", "Boşluk, denge ve biçim ilişkisi bu kareyi soyut okumaya yaklaştırıyor."
    if 0.18 < m.edge_density < 0.42 and m.contrast > 42:
        return "Sokak", "Katman ve sahne hareketi bu karede sokak hissini öne çıkarıyor."
    if m.center_x > 0.35 and m.center_x < 0.65 and m.focus > 90:
        return "Portre", "Odak ağırlığı tek bir merkeze toplanıyor; portre okuması daha doğal duruyor."
    return "Belgesel", "Sahne anlatısı ve bağlam hissi bu kareyi belgesel okumaya yaklaştırıyor."


def brightness_label(v: float) -> str:
    if v < 70:
        return "koyu"
    if v < 120:
        return "kontrollü"
    if v < 180:
        return "dengeli"
    return "parlak"


def contrast_label(v: float) -> str:
    if v < 28:
        return "yumuşak"
    if v < 48:
        return "orta"
    return "sert"


def focus_label(v: float) -> str:
    if v < 55:
        return "gevşek"
    if v < 120:
        return "yeterli"
    return "belirgin"


def subject_position(m: Metrics) -> str:
    if m.center_x < 0.38:
        horiz = "sola yakın"
    elif m.center_x > 0.62:
        horiz = "sağa yakın"
    else:
        horiz = "merkeze yakın"
    if m.center_y < 0.38:
        vert = "üst hatta"
    elif m.center_y > 0.62:
        vert = "alt hatta"
    else:
        vert = "orta hatta"
    return f"{horiz}, {vert}"


def density_label(v: float) -> str:
    if v < 0.16:
        return "seyrek"
    if v < 0.30:
        return "kontrollü"
    if v < 0.40:
        return "yoğun"
    return "çok yoğun"


def negative_space_label(v: float) -> str:
    if v < 0.18:
        return "sıkışık"
    if v < 0.34:
        return "ölçülü"
    if v < 0.58:
        return "rahat"
    return "ferah"


def balance_label(m: Metrics) -> str:
    if m.symmetry > 0.82:
        return "dengeli"
    if m.center_x < 0.35 or m.center_x > 0.65 or m.center_y < 0.35 or m.center_y > 0.65:
        return "gerilimli"
    return "oynak"


def crop_suggestion(m: Metrics) -> str:
    if m.edge_density > 0.38:
        return "kenarlardaki yükü hafifletecek sıkı bir kırpma"
    if m.highlight_clip > 0.04:
        return "parlak bölgeyi kısacak küçük bir kadraj ayarı"
    if m.negative_space < 0.18:
        return "özneye nefes açacak daha açık bir kadraj kararı"
    return "kadrajı büyük ölçüde koruyan küçük bir ton temizliği"


def rhythm_label(m: Metrics) -> str:
    if m.edge_density > 0.36 and m.contrast > 44:
        return "hızlı"
    if m.negative_space > 0.55:
        return "sakin"
    return "ölçülü"


def tonal_risk(m: Metrics) -> str:
    if m.highlight_clip > 0.04:
        return "parlak bölgeler dikkat çekme yarışına giriyor"
    if m.shadow_clip > 0.08:
        return "koyu bölgelerde bilgi kaybı hissi oluşuyor"
    return "ton tarafında büyük bir taşma görünmüyor"


def editorial_decision(score: float) -> str:
    if score >= 80:
        return "seçkiye yakın"
    if score >= 68:
        return "güçlü bir aday"
    if score >= 55:
        return "geliştirilirse seçkiye girebilir"
    return "henüz ham"


def emphasis_phrase(score: float) -> str:
    if score >= 80:
        return "kararını büyük ölçüde vermiş"
    if score >= 68:
        return "ayağı yere basan"
    if score >= 55:
        return "potansiyeli görünen"
    return "fikri olan ama sıkılaşması gereken"


def scene_facts(m: Metrics) -> Dict[str, str]:
    mode, reason = classify_mode(m)
    detail_1 = "ışık kararı"
    if m.highlight_clip > 0.03:
        detail_1 = "parlak alan baskısı"
    elif m.shadow_clip > 0.08:
        detail_1 = "koyu alan birikmesi"
    elif m.contrast > 48:
        detail_1 = "sert ton ayrımı"

    detail_2 = "göz akışı"
    if m.negative_space > 0.58:
        detail_2 = "boşluk kullanımı"
    elif m.edge_density > 0.34:
        detail_2 = "arka plan yoğunluğu"
    elif m.symmetry > 0.80:
        detail_2 = "denge hissi"

    issue = "belirgin sorun görünmüyor"
    if m.edge_density > 0.35:
        issue = "arka plan yükü ana vurguyu zorluyor"
    elif m.highlight_clip > 0.04:
        issue = "parlak bölgeler ana ilişkiyi bastırabiliyor"
    elif m.focus < 50:
        issue = "netlik kararı biraz gevşek kalıyor"
    elif m.negative_space < 0.20:
        issue = "karede nefes alanı az"

    strength = "görsel omurga dengeli"
    if m.focus > 110:
        strength = "ana vurgu net biçimde toplanıyor"
    elif m.symmetry > 0.78:
        strength = "yerleşim dengesi iyi çalışıyor"
    elif 0.22 < m.edge_density < 0.34:
        strength = "sahne yoğunluğu kontrollü"

    return {
        "mode": mode,
        "mode_reason": reason,
        "brightness": brightness_label(m.brightness),
        "contrast": contrast_label(m.contrast),
        "focus": focus_label(m.focus),
        "position": subject_position(m),
        "detail_1": detail_1,
        "detail_2": detail_2,
        "issue": issue,
        "strength": strength,
        "density": density_label(m.edge_density),
        "space": negative_space_label(m.negative_space),
        "balance": balance_label(m),
        "crop": crop_suggestion(m),
        "rhythm": rhythm_label(m),
        "tonal_risk": tonal_risk(m),
    }


def overall_score(m: Metrics) -> float:
    tech = min(1.0, math.log1p(m.focus) / 5.2)
    light = max(0.0, 1 - abs(m.brightness - 122) / 122)
    comp = 1 - min(1.0, (abs(m.center_x - 0.5) + abs(m.center_y - 0.5)) / 0.9)
    control = max(0.0, 1 - (m.highlight_clip + m.shadow_clip) * 3.5)
    clean = max(0.0, 1 - abs(m.edge_density - 0.24) / 0.30)
    return round(100 * (0.25 * tech + 0.18 * light + 0.22 * comp + 0.18 * control + 0.17 * clean), 1)


def level(score: float) -> str:
    if score < 45:
        return "Gelişmeye Açık"
    if score < 65:
        return "Orta"
    if score < 80:
        return "Güçlü"
    return "Çok Güçlü"


def first_reading(facts: Dict[str, str]) -> str:
    return (
        f"İlk bakışta kare {facts['position']} toplanan bir dikkat kuruyor. "
        f"Işık {facts['brightness']}, kontrast ise {facts['contrast']} bir hava veriyor. "
        f"Burada öne çıkan taraf {facts['strength']}; kırılma ise daha çok {facts['issue']} tarafında beliriyor."
    )


def summary_line(score: float, facts: Dict[str, str]) -> str:
    return (
        f"Bu çalışma {facts['mode'].lower()} çizgisine yakın duruyor; ana omurgası {facts['strength']}, "
        f"gelişim alanı ise {facts['issue']}. Genel seviye: {level(score)}."
    )


def build_editor_comment(editor_name: str, facts: Dict[str, str], score: float) -> str:
    verdict = editorial_decision(score)
    emphasis = emphasis_phrase(score)

    if editor_name == "Selahattin Kalaycı":
        return (
            f"Ben bu karede önce görüntünün niyetine bakıyorum. Dikkatin {facts['position']} toplanması rastlantı değil; bu yerleşim izleyiciyi doğrudan bir karar alanına itiyor. "
            f"Sorun şu ki {facts['issue']}; bu yüzden görüntü ilk cümlesini kuruyor ama son cümlesini tam kapatamıyor. Yine de {facts['strength']} tarafı çalışmayı boş bırakmıyor; tam tersine, burada {emphasis} bir omurga var. "
            f"{facts['crop'].capitalize()} yapılırsa bu kare yalnız görünmekle kalmaz, zihinde de daha sert yer eder."
        )
    if editor_name == "Güler Ataşer":
        return (
            f"Bu kare bana önce bir sıcaklık değil, bir hava veriyor. Işığın {facts['brightness']} kalması ve kontrastın {facts['contrast']} tutulması görüntünün ritmini {facts['rhythm']} yapıyor; bu iyi bir başlangıç. "
            f"Özellikle {facts['detail_1']} ile {facts['detail_2']} birlikte çalışınca sahnenin yüzeyinde ince bir titreşim oluşuyor. Fakat {facts['issue']} hissi bu akışın üzerinde küçük bir pürüz bırakıyor. "
            f"{facts['crop'].capitalize()} o pürüzü azaltırsa kare daha temiz bir nefes alır ve etkisini bağırmadan derinleştirir."
        )
    if editor_name == "Sevgin Cingöz":
        return (
            f"Yapısal veri açık: kare {facts['balance']} bir yerleşim kuruyor ve ağırlık {facts['position']} toplanıyor. Bu yerleşimin çalışan tarafı {facts['strength']}; yani gözün ilk tutunduğu iskelet dağılmıyor. "
            f"Asıl aksama {facts['issue']} noktasında. Ayrıca sahne yoğunluğunun {facts['density']} kalması, yük ile ihtiyaç arasındaki farkı büyütüyor. "
            f"Benim önerim net: {facts['crop']} ve ton tarafında küçük bir ayıklama. O zaman kompozisyon daha disiplinli, daha okunur çalışır."
        )
    if editor_name == "Mürşide Çilengir":
        return (
            f"Bu sahne bende önce bir temas duygusu bırakıyor. Ana merkezin {facts['position']} durması, görüntüye kuru bir kayıt hissi yerine insani bir yakınlık katıyor. Buradaki kıymet {facts['strength']}; fotoğrafın kalbi orada atıyor. "
            f"Ama {facts['issue']} yüzünden o kalp biraz sıkışıyor; özellikle alanın {facts['space']} kalması bunu hissettiriyor. "
            f"{facts['crop'].capitalize()} ya da küçük bir ton ferahlığı verilirse sahne izleyiciye daha açık ve daha içten ulaşır."
        )
    return (
        f"Editoryal açıdan baktığımda bu çalışma {verdict} görünüyor. Bunun nedeni yalnızca teknik toparlanma değil; {facts['strength']} dediğimiz omurga kareyi masaya getirecek kadar ayağa kalkmış durumda. "
        f"Ama seçki kararını zorlayan yer hâlâ {facts['issue']}. Üstelik {facts['tonal_risk']} hissi de bunu biraz büyütüyor. "
        f"Ben bu dosyayı şimdiden kapatmam; çünkü {facts['crop']} ile görüntü çok daha net bir yayın omurgasına kavuşabilir."
    )


def polish_comment(editor_name: str, text: str) -> str:
    replacements = {
        'kare': {'Selahattin Kalaycı': 'görüntü', 'Sevgin Cingöz': 'kompozisyon', 'Mürşide Çilengir': 'sahne', 'Gülcan Ceylan Çağın': 'çalışma'},
        'fotoğraf': {'Selahattin Kalaycı': 'görüntü', 'Sevgin Cingöz': 'kompozisyon', 'Mürşide Çilengir': 'sahne', 'Gülcan Ceylan Çağın': 'çalışma'},
    }
    out = text
    for word, mapping in replacements.items():
        if editor_name in mapping:
            out = out.replace(word, mapping[editor_name])
            out = out.replace(word.capitalize(), mapping[editor_name].capitalize())
    if editor_name == 'Gülcan Ceylan Çağın' and 'çalışan tarafı teslim' not in out.lower():
        out = 'Önce çalışan tarafı teslim etmek gerekir. ' + out
    return out


def heatmap_from_image(img: Image.Image) -> Image.Image:
    gray = gray_np(img)
    if cv2 is not None:
        sal = cv2.GaussianBlur((255 - gray).astype(np.float32), (0, 0), sigmaX=18)
    else:
        sal = np.array(Image.fromarray(255 - gray).filter(ImageFilter.GaussianBlur(radius=12))).astype(np.float32)
    sal = sal - sal.min()
    if sal.max() > 0:
        sal = sal / sal.max()
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    h, w = sal.shape
    step = max(8, min(w, h) // 24)
    for y in range(0, h, step):
        for x in range(0, w, step):
            v = float(sal[y:y+step, x:x+step].mean())
            if v < 0.35:
                continue
            alpha = int(120 * v)
            color = (255, int(130 + 80 * (1 - v)), 70, alpha)
            draw.rectangle((x, y, min(w, x+step), min(h, y+step)), fill=color)
    return Image.alpha_composite(base, overlay).convert("RGB")


def render_card_open(title: str):
    st.markdown(f"<div class='card'><div class='sectionhead'>{title}</div>", unsafe_allow_html=True)


def render_card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class='hero'>
            <h1>ÇOFSAT Fotoğraf Ön Değerlendirme · v11 Free</h1>
            <p>Bu sürüm ücretsiz, hızlı ve yerel çalışır. Ağır vision zinciri yerine hafif görsel ölçüm + güçlendirilmiş editör sesi kullanır. Amaç hızdan vazgeçmeden daha karakterli, daha doğal ve birbirinden belirgin biçimde ayrışan yorumlar üretmektir.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Sistem durumu")
        st.markdown("<span class='badge'>Ücretsiz mod</span><span class='badge'>Keskin editör sesi</span><span class='badge'>5 editör</span>", unsafe_allow_html=True)
        st.markdown("<div class='small'>Bu sürüm OpenAI istemez. Sahneden teknik veri çıkarır ve 5 editörü bu verinin üstünde konuşturur. Amaç hız ve tutarlılık.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.05, 1.15], gap="large")
    with col1:
        uploaded = st.file_uploader("Fotoğraf yükle", type=["jpg", "jpeg", "png", "webp"])
        mode_choice = st.selectbox("Okuma modu", ["Otomatik", "Sokak", "Portre", "Belgesel", "Soyut"], index=0)
        show_heatmap = st.toggle("Dikkat akışı ısı haritasını göster", value=True)

        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
            img = safe_resize(img)
            st.image(img, caption="Yüklenen fotoğraf", use_container_width=True)
            if show_heatmap:
                st.image(heatmap_from_image(img), caption="Dikkat akışı / yoğunluk ısı haritası", use_container_width=True)

    with col2:
        if uploaded is None:
            render_card_open("Hazır olduğunda")
            st.markdown("<div class='small'>Sol taraftan bir fotoğraf yüklediğinde sistem hızlı bir ön okuma yapacak, 5 editör aynı kareyi farklı dille yorumlayacak.</div>", unsafe_allow_html=True)
            render_card_close()
            return

        image_bytes = uploaded.getvalue()
        m = extract_metrics(image_bytes)
        facts = scene_facts(m)
        if mode_choice != "Otomatik":
            facts["mode"] = mode_choice
            facts["mode_reason"] = f"Kullanıcı bu kareyi {mode_choice.lower()} çizgisinde okumayı seçti."
        score = overall_score(m)

        render_card_open("Kısa genel okuma")
        st.markdown(f"<div class='small'>{first_reading(facts)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small' style='margin-top:.55rem'>{summary_line(score, facts)}</div>", unsafe_allow_html=True)
        render_card_close()

        render_card_open("Skor kartı")
        st.markdown(
            f"<div class='statgrid'>"
            f"<div class='stat'><div class='k'>Genel skor</div><div class='v'>{score}</div></div>"
            f"<div class='stat'><div class='k'>Seviye</div><div class='v'>{level(score)}</div></div>"
            f"<div class='stat'><div class='k'>Önerilen tür</div><div class='v'>{facts['mode']}</div></div>"
            f"<div class='stat'><div class='k'>Netlik</div><div class='v'>{facts['focus']}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='small' style='margin-top:.65rem'>{facts['mode_reason']}</div>", unsafe_allow_html=True)
        render_card_close()

        render_card_open("Somut veriler")
        st.markdown(
            f"<div class='small'>Işık: <b>{facts['brightness']}</b> · Kontrast: <b>{facts['contrast']}</b> · Ana ağırlık: <b>{facts['position']}</b><br>"
            f"Çalışan taraf: <b>{facts['strength']}</b><br>"
            f"Gelişim alanı: <b>{facts['issue']}</b><br>"
            f"Dikkat çeken iki veri: <b>{facts['detail_1']}</b> ve <b>{facts['detail_2']}</b></div>",
            unsafe_allow_html=True,
        )
        render_card_close()

        render_card_open("5 editör yorumu")
        for name in EDITOR_NAMES:
            persona = EDITOR_PERSONAS[name]
            st.markdown(
                f"<div class='note' style='margin-bottom:.75rem'>"
                f"<div class='editor-title'><div><div class='name'>{name}</div><div class='role'>{persona['title']} · {persona['tone']}</div></div></div>"
                f"<div style='margin-top:.6rem'>{build_editor_comment(name, facts, score)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        render_card_close()

        render_card_open("Çekim ve düzenleme notları")
        notes = [
            f"Ana omurgayı koruyup yalnızca {facts['issue']} tarafını hafifletmek bu kareyi belirgin biçimde güçlendirir.",
            f"Eğer yeniden çekim şansı varsa, ana ağırlığı {facts['position']} çizgisinden çok hafif ayrıştırmak daha net bir hiyerarşi kurabilir.",
            f"Tonlama aşamasında {facts['detail_1']} üzerine dikkatli gitmek yeterli; büyük bir estetik müdahaleden çok küçük bir temizlik daha doğru olur.",
        ]
        for n in notes:
            st.markdown(f"- {n}")
        render_card_close()


if __name__ == "__main__":
    main()
