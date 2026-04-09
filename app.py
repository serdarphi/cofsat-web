import os
import io
import math
import json
import base64
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from html import escape
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageStat, ImageDraw, ImageFilter, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

import requests

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq
    except Exception:
        AutoModelForVision2Seq = None
except Exception:
    AutoProcessor = None
    AutoModelForVision2Seq = None

import streamlit as st
# OPENAI key is read from .streamlit/secrets.toml or environment.
import streamlit.components.v1 as components

st.set_page_config(
    page_title="ÇOFSAT Fotoğraf Ön Değerlendirme",
    layout="wide",
    page_icon="📷",
)
# v16: global ürün polish


PHI = 1.61803398875
MAX_ANALYSIS_SIZE = 1600


OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
QWEN_VISION_MODEL = os.getenv("QWEN_VISION_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")


def get_openai_api_key() -> str:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "").strip()


def openai_vision_available() -> bool:
    return bool(get_openai_api_key())


def qwen_vision_requested() -> bool:
    return os.getenv("COFSAT_ENABLE_QWEN", "0").strip().lower() in {"1", "true", "yes", "on"}


def qwen_vision_runtime_available() -> bool:
    return qwen_vision_requested() and AutoProcessor is not None and AutoModelForVision2Seq is not None and torch is not None


def active_ai_provider_label() -> str:
    if openai_vision_available():
        return "OpenAI Vision"
    if qwen_vision_runtime_available():
        return "Qwen Vision Light"
    return "ÇOFSAT Motoru"

def _safe_provider_error_message(error_text: str) -> str:
    error_text = (error_text or "").strip()
    if not error_text:
        return "Derin AI okuması şu an kullanılamıyor."
    lowered = error_text.lower()
    if "429" in lowered or "rate" in lowered or "too many requests" in lowered or "çok fazla istek" in lowered:
        return "Derin AI okuması şu an yoğunluk nedeniyle kullanılamıyor."
    if "401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
        return "Derin AI okuması için API anahtarı doğrulanamadı."
    return "Derin AI okuması şu an kullanılamıyor."

def _current_analysis_key(image_bytes: bytes, mode: str, editor_mode: str) -> str:
    return hashlib.md5(image_bytes + f"|{mode}|{editor_mode}".encode("utf-8")).hexdigest()

@st.cache_resource(show_spinner=False)
def load_qwen_vision_model():
    if not qwen_vision_runtime_available():
        raise RuntimeError("Qwen runtime hazır değil. transformers/torch kurulumu eksik olabilir.")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(QWEN_VISION_MODEL, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        QWEN_VISION_MODEL,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    return processor, model


def _decode_qwen_output(processor, generated_ids) -> str:
    try:
        if hasattr(processor, "batch_decode"):
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            text = processor.decode(generated_ids[0], skip_special_tokens=True)
    except Exception:
        text = str(generated_ids)
    return (text or "").strip()


@st.cache_data(show_spinner=False)
def qwen_vision_critique_cached(image_bytes: bytes, mode: str, editor_mode: str) -> Dict:
    if not qwen_vision_runtime_available():
        return {}

    heuristic = _build_heuristic_context_for_llm(image_bytes, mode, editor_mode)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prompt = f"""
Sen ÇOFSAT için çalışan, güçlü görsel okuma yapan ve birbirinden net biçimde ayrışan 5 editör sesi üreten fotoğraf yorum motorususun.
Her editör; John Berger, Susan Sontag ve Roland Barthes çizgilerinden türetilmiş genel düşünme biçimlerinden beslenebilir ama bu isimleri asla anmaz, doğrudan taklit yapmaz.

Bu fotoğrafı ÇOFSAT mantığıyla oku.
Yorumlarda:
- her editör kendi persona kaydına sadık kalsın,
- her editör yorumunda fotoğraftan en az 2 somut ayrıntı geçsin,
- figürün yeri, ışığın durumu, mekânın dokusu ve sahne içi gerilim mutlaka hesaba katılsın,
- teknik kadar anlam, duygu, ilişki ve kadraj içi kırılma konuşulsun,
- editörler birbirine benzemesin,
- her editör kendi kelime dağarcığına ve cümle ritmine sadık kalsın,
- editörler gerekirse birbirine zıt görüş bildirebilsin,
- Berger çizgisi: bakış rejimi, bağlam ve neden böyle görüldüğü,
- Sontag çizgisi: temsil, seçme eylemi, etik/mesafe ve tanıklık,
- Barthes çizgisi: küçük ayrıntının kişisel yarası, punctum etkisi,
- bu çizgiler doğrudan alıntı veya isim vererek değil, yorumların düşünme biçiminde hissedilsin,
- her editör 4-5 cümle kursun,
- cümle açılışları tekrar etmesin,
- genel ve şablon cümlelerden kaçın,
- "fotoğraf güzel", "güçlü yanı", "biraz daha" gibi jenerik kalıpları kullanma.

Heuristik bağlam:
{json.dumps(heuristic, ensure_ascii=False)}

Sadece geçerli JSON döndür:
{{
  "scene_summary": "2-3 cümlelik genel okuma",
  "concrete_details": ["fotoğrafa özgü 5 somut ayrıntı"],
  "meaning_layers": ["anlam katmanı 1", "anlam katmanı 2", "anlam katmanı 3"],
  "editor_comments": {{
    "Selahattin Kalaycı": "4-5 cümle",
    "Güler Ataşer": "4-5 cümle",
    "Sevgin Cingöz": "4-5 cümle",
    "Mürşide Çilengir": "4-5 cümle",
    "Gülcan Ceylan Çağın": "4-5 cümle"
  }},
  "global_summary": "kısa genel değerlendirme",
  "one_line_caption": "fotoğrafın özünü veren kısa cümle"
}}
"""

    try:
        processor, model = load_qwen_vision_model()

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt.strip()},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_prompt], images=[img], return_tensors="pt")
        except Exception:
            inputs = processor(images=img, text=prompt.strip(), return_tensors="pt")

        device = getattr(model, "device", None)
        if device is not None:
            try:
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            except Exception:
                pass

        generated_ids = model.generate(**inputs, max_new_tokens=700)
        raw_text = _decode_qwen_output(processor, generated_ids)
    except Exception as exc:
        return {"error": str(exc)}

    parsed = _extract_json_object(raw_text)
    if isinstance(parsed, dict) and parsed:
        parsed["_model"] = QWEN_VISION_MODEL
        parsed["_provider"] = "qwen_local"
        return parsed
    return {"raw_text": raw_text[:5000], "_model": QWEN_VISION_MODEL, "_provider": "qwen_local"}


def _extract_json_object(raw_text: str) -> Dict:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {}
    try:
        return json.loads(raw_text)
    except Exception:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw_text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return {}
    return {}


def _image_bytes_to_data_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_heuristic_context_for_llm(image_bytes: bytes, mode: str, editor_mode: str) -> Dict:
    metrics = ImageMetrics(**extract_metrics_cached(image_bytes))
    profile = SceneProfile(**extract_scene_profile_cached(image_bytes))
    return {
        "selected_mode": mode,
        "editor_mode": editor_mode,
        "scene_type": profile.scene_type,
        "subject_hint": profile.subject_hint,
        "visual_mood": profile.visual_mood,
        "light_character": profile.light_character,
        "complexity_level": profile.complexity_level,
        "balance_character": profile.balance_character,
        "primary_region": profile.primary_region,
        "secondary_region": profile.secondary_region,
        "distraction_region": profile.distraction_region,
        "subject_position": profile.subject_position,
        "light_type_detail": profile.light_type_detail,
        "environment_type": profile.environment_type,
        "human_action": profile.human_action,
        "dominant_shape": profile.dominant_shape,
        "visual_tension_level": profile.visual_tension_level,
        "historical_texture_hint": profile.historical_texture_hint,
        "concrete_details": [profile.concrete_detail_1, profile.concrete_detail_2, profile.concrete_detail_3],
        "editor_personas": EDITOR_PERSONAS,
        "editor_voice_engines": EDITOR_VOICE_ENGINES,
        "critic_style_dna": CRITIC_STYLE_DNA,
        "editor_critic_blends": {name: _critic_blend_payload(name) for name in EDITOR_NAMES},
        "face_count": profile.face_count,
        "human_presence_score": round(profile.human_presence_score, 3),
        "brightness_mean": round(metrics.brightness_mean, 2),
        "contrast_std": round(metrics.contrast_std, 2),
        "edge_density": round(metrics.edge_density, 4),
        "negative_space_score": round(metrics.negative_space_score, 4),
        "dynamic_tension_score": round(metrics.dynamic_tension_score, 4),
        "highlight_clip_ratio": round(metrics.highlight_clip_ratio, 4),
        "shadow_clip_ratio": round(metrics.shadow_clip_ratio, 4),
    }



SELAHATTIN_STYLE_DNA = {
    "opening_moves": [
        "görsel iskeleti hızlıca kur: kontrast, yön çizgisi, perspektif, ritim veya yerleşim",
        "ardından 'ama asıl...' dönüşüyle sahnenin gerçek hikâyesine geç",
        "somut ayrıntılardan atmosfer, zaman duygusu veya tarih hissi çıkar"
    ],
    "language_markers": [
        "yön kontrastı", "perspektif derinliği", "dikey ritim", "ışık-gölge dağılımı",
        "zamansız", "sessizlik hissi", "tarihsel katman", "gizemli", "melankolik"
    ],
    "tone": "sıcak, kendinden emin, görsel ayrıntıyı hızla okuyup ardından şiirsel ama anlaşılır bir derinliğe geçen",
    "avoid": [
        "aşırı felsefi soyutlama", "çok teknik rapor dili", "aynı öneriyi herkes gibi verme"
    ],
    "signature": [
        "önce fotoğrafın görünen yapısını teslim et",
        "sonra atmosferi ve asıl hikâyeyi aç",
        "gerekirse tek bir küçük kusuru nazikçe işaret et",
        "kapanışta kısa ve sıcak bir takdir tonu bırak"
    ]
}

@st.cache_data(show_spinner=False)
def openai_editor_comment_cached(image_bytes: bytes, editor_name: str, mode: str, editor_mode: str) -> Dict:
    api_key = get_openai_api_key()
    if not api_key:
        return {}

    data_url = _image_bytes_to_data_url(image_bytes)

    editor_rules = {
        "Selahattin Kalaycı": {
            "focus": "görüntünün iskeleti, kontrastı, yön ilişkileri ve perspektifinden başlayıp 'ama asıl' dönüşüyle sahnenin niyetini, atmosferini ve zamansız/tarihsel katmanını açmak",
            "avoid": "kuru teknik rapor, aşırı soyut felsefi dil, diğer editörlerle aynı öneriyi vermek",
        },
        "Güler Ataşer": {
            "focus": "yalnızca ışık, ton, yüzey, hava, doku ve görüntünün teni",
            "avoid": "editoryal hüküm, kompozisyon açıklaması, hikâye dersi",
        },
        "Sevgin Cingöz": {
            "focus": "yalnızca kompozisyon, yerleşim, denge, göz akışı, ikinci hareket ve yapısal kırılma",
            "avoid": "duygusal romantizasyon, atmosfer dili, editoryal karar",
        },
        "Mürşide Çilengir": {
            "focus": "yalnızca insan ilişkisi, beden dili, yakınlık, kırılganlık ve duygusal çekirdek",
            "avoid": "teknik rapor dili, ışık jargonu, kompozisyon dersi",
        },
        "Gülcan Ceylan Çağın": {
            "focus": "yalnızca editoryal seçim, yayın potansiyeli, seçkiye girip girmeme eşiği ve fazlalıkların ayıklanması",
            "avoid": "şiirsel uzatma, teknik rapor, diğer editörlerin alanı",
        },
    }
    rule = editor_rules.get(editor_name, {"focus": "fotoğrafı özgün bir bakışla yorumla", "avoid": "genel geçer cümleler"})

    extra_dna = ""
    if editor_name == "Selahattin Kalaycı":
        extra_dna = f"""
Selahattin Kalaycı yazılım DNA'sı:
- Açılışı çoğu zaman görüntünün omurgasından yap: kontrast, dikey-yatay ilişki, perspektif derinliği, yön çizgileri, ışık-gölge dağılımı.
- Ardından 'ama asıl...' türü bir kırılmayla gerçek hikâyeye geç.
- Somut ayrıntılardan melankoli, sessizlik, zamansızlık, mahalle/tarih duygusu veya gizem hissi üret.
- Dilin sıcak, akıcı ve fotoğraf kulübü üslubuna yakın olsun; anlaşılır kal, fazla teorikleşme.
- Bir küçük kusur söyleyebilirsin ama sertleşme; kısa bir takdir tonu taşı.
- Şu söz alanlarını doğal biçimde kullanabilirsin: {', '.join(SELAHATTIN_STYLE_DNA['language_markers'])}.
"""

    system_prompt = f"""
Sen yalnızca {editor_name} olarak konuşan tek bir fotoğraf editörüsün.
Başka editör yok. Önceden üretilmiş hiçbir analizi kullanma. Görüntüyü sıfırdan kendin oku.
Yalnızca şu alana odaklan: {rule['focus']}.
Şunlardan kaçın: {rule['avoid']}.
{extra_dna}
Kurallar:
- Fotoğrafı sıfırdan analiz et; heuristik bağlam, önceki analiz, hazır yorum veya ortak özet kullanma.
- En az 4 cümle yaz.
- En az 2 somut görsel ayrıntı an.
- Aynı cümleyi, aynı kalıbı ve aynı öneriyi tekrar etme.
- Belirsizsen uydurma yapma; bunu dürüstçe belirt.
- Genel laflar kurma: “güçlü bir atmosfer”, “izleyiciyle bağ”, “hikâye hissi” gibi boş kalıplardan kaçın.
- Sadece {editor_name} gibi konuş.
"""

    user_prompt = f"""
Bu fotoğrafı {editor_name} olarak, {mode} bağlamını biliyor olsan bile hazır analizlerden bağımsız biçimde kendin oku.
Editör tonu: {editor_mode}.

Yalnızca düz metin döndür.
"""

    payload = {
        "model": OPENAI_VISION_MODEL,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt.strip()}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt.strip()},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        "max_output_tokens": 500,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return {"error": str(exc)}

    raw_text = ""
    if isinstance(data, dict):
        raw_text = data.get("output_text", "") or ""
        if not raw_text:
            outputs = data.get("output", []) or []
            parts: List[str] = []
            for item in outputs:
                for content in item.get("content", []) or []:
                    if content.get("type") in {"output_text", "text"}:
                        text_value = content.get("text")
                        if isinstance(text_value, str):
                            parts.append(text_value)
                if parts:
                    break
            raw_text = "\n".join(parts).strip()

    return {
        "editor": editor_name,
        "comment": (raw_text or "").strip(),
        "_model": OPENAI_VISION_MODEL,
    }


@st.cache_data(show_spinner=False)
def openai_scene_read_cached(image_bytes: bytes, mode: str, editor_mode: str) -> Dict:
    api_key = get_openai_api_key()
    if not api_key:
        return {}

    data_url = _image_bytes_to_data_url(image_bytes)
    system_prompt = """
Sen ÇOFSAT için çalışan bir görsel okuma motorusun.
Fotoğrafı sıfırdan kendin oku. Önceden yapılmış hiçbir analizi, heuristik bağlamı veya yorum özetini kullanma.
Somut ve kısa ol. Uydurma yapma.
"""
    user_prompt = f"""
Bu fotoğrafı sıfırdan analiz et.
Kısa bir JSON nesnesi döndür:
{{
  "scene_summary": "2 cümle",
  "concrete_details": ["4 somut ayrıntı"],
  "meaning_layers": ["3 kısa anlam katmanı"],
  "global_summary": "1 kısa paragraf",
  "one_line_caption": "1 cümle"
}}
"""
    payload = {
        "model": OPENAI_VISION_MODEL,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt.strip()}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt.strip()},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
        "max_output_tokens": 700,
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return {"error": str(exc)}

    raw_text = ""
    if isinstance(data, dict):
        raw_text = data.get("output_text", "") or ""
        if not raw_text:
            outputs = data.get("output", []) or []
            parts: List[str] = []
            for item in outputs:
                for content in item.get("content", []) or []:
                    if content.get("type") in {"output_text", "text"}:
                        text_value = content.get("text")
                        if isinstance(text_value, str):
                            parts.append(text_value)
                if parts:
                    break
            raw_text = "\n".join(parts).strip()

    parsed = _extract_json_object(raw_text)
    if isinstance(parsed, dict) and parsed:
        parsed["_model"] = OPENAI_VISION_MODEL
        return parsed
    return {"raw_text": raw_text[:5000], "_model": OPENAI_VISION_MODEL}


@st.cache_data(show_spinner=False)
def openai_vision_critique_cached(image_bytes: bytes, mode: str, editor_mode: str) -> Dict:
    api_key = get_openai_api_key()
    if not api_key:
        return {}

    scene_payload = openai_scene_read_cached(image_bytes, mode, editor_mode) or {}
    if isinstance(scene_payload, dict) and scene_payload.get("error"):
        return scene_payload

    editor_comments: Dict[str, str] = {}
    model_name = OPENAI_VISION_MODEL
    for editor_name in EDITOR_NAMES:
        payload = openai_editor_comment_cached(image_bytes, editor_name, mode, editor_mode) or {}
        if isinstance(payload, dict) and payload.get("error"):
            return payload
        comment = str(payload.get("comment", "")).strip()
        if comment:
            editor_comments[editor_name] = comment
        if payload.get("_model"):
            model_name = str(payload.get("_model"))

    result = {
        "editor_comments": editor_comments,
        "_model": model_name,
    }
    for key in ["scene_summary", "concrete_details", "meaning_layers", "global_summary", "one_line_caption"]:
        value = scene_payload.get(key) if isinstance(scene_payload, dict) else None
        if value:
            result[key] = value
    return result

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

EDITOR_NAMES = [
    "Selahattin Kalaycı",
    "Güler Ataşer",
    "Sevgin Cingöz",
    "Mürşide Çilengir",
    "Gülcan Ceylan Çağın",
]


EDITOR_PERSONAS = {
    "Selahattin Kalaycı": {
        "focus": ["anlam", "niyet", "hikâye", "kadraj içi gerilim"],
        "tone": "düşünsel, sorgulayıcı, yer yer felsefi",
        "entry_style": "soru ile açar",
        "decision_style": "fotoğrafın niçin var olduğunu arar",
        "must_do": ["somut ayrıntıdan metafor kur", "son cümlede düşünsel bir genişleme yap"],
        "avoid": ["düz teknik özet", "fazla kısa hüküm"],
    },
    "Güler Ataşer": {
        "focus": ["ışık", "atmosfer", "dokusal incelik", "ton geçişi"],
        "tone": "şiirsel, yumuşak, duyusal",
        "entry_style": "hissettiği havadan başlar",
        "decision_style": "ışıktan ve havadan anlam çıkarır",
        "must_do": ["en az bir dokusal ayrıntı an", "ışığın duygusal etkisini söyle"],
        "avoid": ["sert hüküm", "mekanik analiz dili"],
    },
    "Sevgin Cingöz": {
        "focus": ["kompozisyon", "denge", "figürün yeri", "göz akışı"],
        "tone": "analitik, net, düz konuşan",
        "entry_style": "doğrudan tespitle açar",
        "decision_style": "yerleşim ve akış üzerinden karar verir",
        "must_do": ["sağ-sol/ön-arka ilişkisini an", "somut yapısal öneri ver"],
        "avoid": ["duyguyu fazla romantize etmek", "muğlak cümle"],
    },
    "Mürşide Çilengir": {
        "focus": ["insan", "yakınlık", "duygu", "sessiz gerilim"],
        "tone": "empatik, insancıl, içten",
        "entry_style": "insani etkiyle açar",
        "decision_style": "fotoğrafın kalbini insan ilişkilerinde arar",
        "must_do": ["beden dili ya da mesafe hissini an", "eleştiriyi şefkatle kur"],
        "avoid": ["soğuk teknik dil", "keskin yargı"],
    },
    "Gülcan Ceylan Çağın": {
        "focus": ["editoryal değer", "yayınlanabilirlik", "seçki gücü", "karar netliği"],
        "tone": "profesyonel, dengeli, seçici ve yapıcı",
        "entry_style": "önce çalışan tarafı teslim edip sonra karar verir",
        "decision_style": "yayına girer / geliştirilirse girer eşiğini dengeli tartar",
        "must_do": ["önce güçlü tarafı teslim et", "seçki açısından açık ama yapıcı hüküm ver", "fazlalığı somut biçimde işaret ederken çözüm de öner"],
        "avoid": ["sürekli reddedici dil", "tek cümlede kesin olumsuz hüküm", "aşırı romantik ton"],
    },
}

EDITOR_VOICE_ENGINES = {
    "Selahattin Kalaycı": {
        "lexicon": ["niyet", "eşik", "sessiz gerilim", "iç cümle", "gizli ağırlık", "düşünsel alan"],
        "openings": [
            "Bu kare gerçekten neyi saklayarak konuşuyor?",
            "İnsan bu fotoğrafa bakınca önce şu soruya çarpıyor:",
            "Ben burada önce görüneni değil, kadrajın kendi niyetini sorguluyorum:"
        ],
        "connectors": ["çünkü", "bu yüzden", "tam da burada", "bana kalırsa"],
        "closings": [
            "Küçük bir netleşmeyle bu fotoğraf yalnızca görünmeyecek, zihinde de kalacak.",
            "Bir karar daha sertleşirse kare düşünceden görüntüye değil, görüntüden düşünceye yürür.",
            "Asıl değer, neyi gösterdiğinde değil, neden böyle sustuğunda yatıyor."
        ],
        "cadence": "uzun-cümle / soru / düşünsel kapanış",
    },
    "Güler Ataşer": {
        "lexicon": ["hava", "dokusal nefes", "ten", "süzülen ışık", "yumuşak yankı", "iç ısı"],
        "openings": [
            "Bu fotoğraf bana önce bir hava veriyor:",
            "Beni içine alan ilk şey sahnenin soluğu oluyor:",
            "Burada görüntüden önce atmosfer konuşuyor:"
        ],
        "connectors": ["sanki", "usulca", "orada", "birden değil"],
        "closings": [
            "Bu etkiyi fazla zorlamadan korumak, fotoğrafın lehine çalışır.",
            "Kare bağırmıyor; iyi yanı da tam olarak burada.",
            "Bir iki küçük temizlikle bu şiirsellik daha derine iner."
        ],
        "cadence": "yumuşak giriş / duyusal orta bölüm / sakin kapanış",
    },
    "Sevgin Cingöz": {
        "lexicon": ["yerleşim", "eksen", "göz akışı", "taşıyıcı hat", "denge kırığı", "kompozisyon kararı"],
        "openings": [
            "Yapısal veri açık:",
            "Bu kare önce yerleşimiyle karar veriyor:",
            "Kompozisyon açısından bakınca ilk tespit net:"
        ],
        "connectors": ["özellikle", "dolayısıyla", "bu nedenle", "burada"],
        "closings": [
            "Sorun ilham değil; kompozisyon kararlarını biraz daha sıkı vermek.",
            "Temel iskelet var, şimdi onu disiplinle netleştirmek gerekiyor.",
            "Doğru toparlanırsa göz akışı çok daha berrak çalışır."
        ],
        "cadence": "kısa giriş / net tespit / doğrudan öneri",
    },
    "Mürşide Çilengir": {
        "lexicon": ["yakınlık", "kırılganlık", "içtenlik", "insani çekirdek", "sessiz temas", "duygu taşıyıcısı"],
        "openings": [
            "Bu kare bende önce insani bir iz bırakıyor:",
            "Gözden önce içimde kalan şey şu oluyor:",
            "Fotoğrafın kalbi bana burada açılıyor:"
        ],
        "connectors": ["sanki", "orada", "bence", "bir bakıma"],
        "closings": [
            "Bu sessiz etkiyi biraz daha görünür bırakmak yeterli olabilir.",
            "İnsani çekirdek yerinde; gerisi ona biraz daha alan açmak.",
            "Şefkatli bir netlikle toparlanırsa izleyicide daha uzun kalır."
        ],
        "cadence": "duygusal giriş / sıcak gözlem / şefkatli öneri",
    },
    "Gülcan Ceylan Çağın": {
        "lexicon": ["yayın omurgası", "seçki dengesi", "editoryal netlik", "gereksiz yük", "yayın potansiyeli", "toparlama payı"],
        "openings": [
            "Editoryal açıdan bakınca önce çalışan tarafı teslim etmek gerekir:",
            "Seçki masasında bu karenin elini güçlendiren şey şu:",
            "Ben bu kareye yayın ihtimali üzerinden bakıyorum ve ilk gördüğüm şey şu:"
        ],
        "connectors": ["özellikle", "bu yüzden", "öte yandan", "aynı anda"],
        "closings": [
            "Biraz daha toparlanırsa bu kare seçki içinde rahatlıkla kendine yer açabilir.",
            "Ben burada kapıyı kapatmıyorum; küçük bir editoryal toplama ile görüntü ciddi biçimde güçlenir.",
            "Karar alanı açık: çalışan taraf korunup gereksiz yük ayıklanırsa yayın ihtimali belirginleşir."
        ],
        "cadence": "güçlü tarafı teslim / kanıt / dengeli editoryal karar",
    },
}


CRITIC_STYLE_DNA = {
    "John Berger": {
        "core": "görmek ile anlamlandırmak arasındaki ilişki; görüntünün nasıl bir bakış rejimi kurduğu",
        "language": "açık, derin, düşünsel ama erişilebilir",
        "moves": [
            "görünen şey ile ona bakma biçimi arasındaki farkı vurgula",
            "fotoğrafın neden böyle kurulduğunu sorgula",
            "görüntünün toplumsal ve mekânsal bağlamını sezdir"
        ],
        "avoid": ["aşırı akademik jargona kaçmak", "soyutluğu görüntüden koparmak"],
    },
    "Susan Sontag": {
        "core": "fotoğrafın temsil gücü, etik mesafesi, tanıklık ve seçme eylemi",
        "language": "keskin, berrak, eleştirel ama kontrollü",
        "moves": [
            "fotoğrafın neyi görünür kıldığını ve neyi dışarıda bıraktığını sorgula",
            "estetik karar ile tanıklık/mesafe ilişkisini tart",
            "görüntünün etkisinin seçim ve eleme sonucu olduğunu hissettir"
        ],
        "avoid": ["ajitatif moralizm", "ham slogan cümleleri"],
    },
    "Roland Barthes": {
        "core": "görüntüde küçük bir ayrıntının izleyiciyi yaralayan kişisel etkisi; studium ve punctum gerilimi",
        "language": "duyarlı, ince, kişisel ama ölçülü",
        "moves": [
            "küçük bir ayrıntının sahnenin tamamını nasıl deldiğini söyle",
            "fotoğrafın duygusal çekirdeğini tek bir ayrıntı üzerinden kur",
            "görünen ile hissedilen arasındaki ince mesafeyi aç"
        ],
        "avoid": ["aşırı melodram", "tamamen kapalı soyut ifadeler"],
    },
}

EDITOR_CRITIC_BLEND = {
    "Selahattin Kalaycı": ["John Berger", "Roland Barthes"],
    "Güler Ataşer": ["Roland Barthes"],
    "Sevgin Cingöz": ["John Berger", "Susan Sontag"],
    "Mürşide Çilengir": ["Roland Barthes", "John Berger"],
    "Gülcan Ceylan Çağın": ["Susan Sontag", "John Berger"],
}

EDITOR_SIGNATURES = {
    "Selahattin Kalaycı": {
        "subject_word": "görüntü",
        "signature_phrases": ["bana kalırsa", "asıl mesele", "düşünsel eşik"],
    },
    "Güler Ataşer": {
        "subject_word": "kare",
        "signature_phrases": ["usulca", "havayı taşıyor", "dokusal nefes"],
    },
    "Sevgin Cingöz": {
        "subject_word": "kompozisyon",
        "signature_phrases": ["net olarak", "yapısal karar", "taşıyıcı hat"],
    },
    "Mürşide Çilengir": {
        "subject_word": "sahne",
        "signature_phrases": ["içten bir yerden", "insani çekirdek", "sessiz temas"],
    },
    "Gülcan Ceylan Çağın": {
        "subject_word": "çalışma",
        "signature_phrases": ["editoryal açıdan", "yayın potansiyeli", "toparlama payı"],
    },
}

EDITOR_SHARED_WORD_REMAP = {
    "Selahattin Kalaycı": {"fotoğraf": "görüntü", "kare": "görüntü", "ana vurgu": "düşünsel ağırlık"},
    "Güler Ataşer": {"fotoğraf": "kare", "görüntü": "kare", "ana vurgu": "ışık izi"},
    "Sevgin Cingöz": {"fotoğraf": "kompozisyon", "kare": "kompozisyon", "ana vurgu": "taşıyıcı merkez"},
    "Mürşide Çilengir": {"fotoğraf": "sahne", "kare": "sahne", "ana vurgu": "insani çekirdek"},
    "Gülcan Ceylan Çağın": {"fotoğraf": "çalışma", "kare": "çalışma", "ana vurgu": "yayın omurgası"},
}


def _apply_editor_signature(editor_name: str, text: str) -> str:
    text = (text or "").strip()
    remap = EDITOR_SHARED_WORD_REMAP.get(editor_name, {})
    for old, new in remap.items():
        text = text.replace(old, new)
        text = text.replace(old.capitalize(), new.capitalize())
    signature = EDITOR_SIGNATURES.get(editor_name, {})
    phrases = signature.get("signature_phrases", []) if isinstance(signature, dict) else []
    for phrase in phrases[:2]:
        if phrase and phrase not in text:
            if text.endswith('.'):
                text = text[:-1] + f"; {phrase}."
            else:
                text += f"; {phrase}."
            break
    return text


def _soften_gulcan_comment(text: str) -> str:
    replacements = {
        "şu haliyle seçkiye almam": "şu haliyle biraz daha toparlama ister",
        "yayın eşiği için ana vurgu daha sert ayrılmalı": "yayın eşiği için ana vurgu biraz daha net ayrışmalı",
        "mesele duygudan çok": "meseleyi yalnız duyguda değil, aynı zamanda",
        "karar sertliği": "karar netliği",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if "güçlü tarafı" not in text and "çalışan taraf" not in text:
        text = "Çalışan tarafı teslim etmek gerekir: " + text[0].lower() + text[1:] if text else text
    return text


def _critic_blend_payload(editor_name: str) -> List[Dict[str, str]]:
    names = EDITOR_CRITIC_BLEND.get(editor_name, [])
    payload = []
    for name in names:
        dna = CRITIC_STYLE_DNA.get(name, {})
        if dna:
            payload.append({
                "critic": name,
                "core": str(dna.get("core", "")),
                "language": str(dna.get("language", "")),
            })
    return payload


def _critic_sentence(editor_name: str, bits: Dict, profile, result: "CritiqueResult") -> str:
    detail_1 = bits.get("detail_1", "küçük ayrıntı")
    detail_2 = bits.get("detail_2", "ikinci vurgu")
    subject_position = getattr(profile, "subject_position", bits.get("primary", "merkez")) if profile is not None else bits.get("primary", "merkez")
    environment_type = getattr(profile, "environment_type", "gündelik çevre") if profile is not None else "gündelik çevre"
    historical_texture_hint = getattr(profile, "historical_texture_hint", "mekân dokusu anlatıya katılıyor") if profile is not None else "mekân dokusu anlatıya katılıyor"
    human_action = getattr(profile, "human_action", "hafif bir hareket") if profile is not None else "hafif bir hareket"
    emotional = float((result.rubric_scores or {}).get("duygusal_yogunluk", 0))
    editorial = float((result.rubric_scores or {}).get("editoryal_deger", 0))

    if editor_name == "Selahattin Kalaycı":
        return (
            f"{detail_1.capitalize()} yalnızca sahnenin içindeki bir unsur gibi değil; "
            f"ona nasıl bakmamız gerektiğini de kuruyor ve {subject_position} duran ağırlık bu yüzden düşünsel bir eşik yaratıyor."
        )
    if editor_name == "Güler Ataşer":
        return (
            f"Benim için punctum duygusu biraz {detail_2} tarafında doğuyor; "
            f"{historical_texture_hint} ile birleşince görüntünün teninde kalan ince sızı oradan yayılıyor."
        )
    if editor_name == "Sevgin Cingöz":
        return (
            f"Bu kareyi yalnızca güzel ya da zayıf diye ayırmak yetmez; "
            f"{environment_type} içinde {human_action} hissinin hangi seçme ve eleme kararlarıyla görünür olduğu yapısal olarak okunabiliyor."
        )
    if editor_name == "Mürşide Çilengir":
        punctum = detail_1 if emotional >= 60 else detail_2
        return (
            f"Bana dokunan şey tam olarak {punctum}; "
            f"o küçük ayrıntı yüzünden sahnedeki insan yakınlığı sadece bilgi olmaktan çıkıp içte kalan bir iz hâline geliyor."
        )
    verdict_line = "Estetik karar ile tanıklık duygusu aynı çizgide buluşmuş." if editorial >= 68 else "Estetik karar ile tanıklık duygusu birbirine yaklaşmış ama hâlâ biraz toparlama payı var."
    return (
        f"{verdict_line} {detail_1.capitalize()} bu çalışmanın yayın potansiyelini açıyor; "
        f"{detail_2} ve {historical_texture_hint} arasındaki seçimi biraz daha netleştirmek editoryal eşiği görünür kılar."
    )



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
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
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



# Removed earlier duplicate definition of pick_strengths

# Removed earlier duplicate definition of pick_development_areas

# Removed earlier duplicate definition of build_key_strength

# Removed earlier duplicate definition of build_key_issue

# Removed earlier duplicate definition of build_one_move_improvement

# Removed earlier duplicate definition of build_reading_prompts

# Removed earlier duplicate definition of build_shooting_notes

# Removed earlier duplicate definition of build_editing_notes
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



# Removed earlier duplicate definition of build_editor_summary
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



# Removed earlier duplicate definition of critique_image
def get_resized_rgb(image_bytes: bytes) -> Image.Image:
    return safe_resize(ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB"))


def optimize_uploaded_bytes(image_bytes: bytes, max_size: int = 1600, jpeg_quality: int = 85) -> bytes:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    img = safe_resize(img, max_size=max_size)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return buffer.getvalue()


def human_file_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0


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
    norm = normalize_array(attention)
    # yumuşatılmış yoğunluk haritası
    smooth = np.array(Image.fromarray((norm * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=22))).astype(np.float32) / 255.0

    h, w = smooth.shape
    heat = np.zeros((h, w, 4), dtype=np.uint8)

    # profesyonel mavi->camgöbeği->sarı->turuncu->kırmızı geçişi
    t = np.clip(smooth, 0.0, 1.0)
    r = np.where(t < 0.35, 40 + t / 0.35 * 120, np.where(t < 0.65, 160 + (t - 0.35) / 0.30 * 95, 255))
    g = np.where(t < 0.35, 95 + t / 0.35 * 120, np.where(t < 0.65, 215 - (t - 0.35) / 0.30 * 80, 135 - np.clip((t - 0.65) / 0.35, 0, 1) * 85))
    b = np.where(t < 0.35, 180 + t / 0.35 * 40, np.where(t < 0.65, 220 - (t - 0.35) / 0.30 * 170, 50 - np.clip((t - 0.65) / 0.35, 0, 1) * 50))
    alpha = np.clip((t ** 1.15) * 210, 0, 210)

    heat[..., 0] = r.astype(np.uint8)
    heat[..., 1] = g.astype(np.uint8)
    heat[..., 2] = np.clip(b, 0, 255).astype(np.uint8)
    heat[..., 3] = alpha.astype(np.uint8)

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
        draw.line((x, 0, x, h), fill=(112, 203, 255, 250), width=7)
    for y in (y1, y2):
        draw.line((0, y, w, y), fill=(112, 203, 255, 250), width=7)

    for x in (x1, x2):
        for y in (y1, y2):
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=(112, 203, 255, 245))

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
        draw.line(line, fill=(255, 154, 82, 242 if i < 2 else 214), width=7 if i < 2 else 4)

    return Image.alpha_composite(out, overlay).convert("RGB")



def draw_golden_spiral(img: Image.Image, focus_points: Optional[List[Tuple[int, int]]] = None) -> Image.Image:
    out = img.copy().convert("RGBA")
    w, h = out.size
    scale = 2
    big_size = (w * scale, h * scale)
    overlay = Image.new("RGBA", big_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    min_dim = min(w, h)

    if focus_points:
        cx = sum(p[0] for p in focus_points) / max(1, len(focus_points))
        cy = sum(p[1] for p in focus_points) / max(1, len(focus_points))
    else:
        cx, cy = w * 0.62, h * 0.38

    cx = max(min_dim * 0.12, min(w - min_dim * 0.12, cx))
    cy = max(min_dim * 0.12, min(h - min_dim * 0.12, cy))

    b = 2 * math.log(PHI) / math.pi
    a = max(8.0, min_dim * 0.018)

    if cx >= w / 2 and cy <= h / 2:
        rotation = -math.pi * 0.15
    elif cx < w / 2 and cy <= h / 2:
        rotation = math.pi * 0.35
    elif cx < w / 2 and cy > h / 2:
        rotation = math.pi * 0.85
    else:
        rotation = -math.pi * 0.65

    points = []
    max_theta = math.pi * 5.2
    steps = 820
    for i in range(steps + 1):
        theta = (i / steps) * max_theta
        r = a * math.exp(b * theta)
        x = (cx + r * math.cos(theta + rotation)) * scale
        y = (cy + r * math.sin(theta + rotation)) * scale
        points.append((x, y))

    draw.line(points, fill=(163, 241, 109, 250), width=12, joint="curve")
    draw.line(points, fill=(246, 250, 232, 165), width=4, joint="curve")

    ring_r = max(8, int(min_dim * 0.018)) * scale
    cxb, cyb = cx * scale, cy * scale
    draw.ellipse((cxb - ring_r, cyb - ring_r, cxb + ring_r, cyb + ring_r), outline=(246, 250, 232, 215), width=5)
    draw.ellipse((cxb - 6, cyb - 6, cxb + 6, cyb + 6), fill=(163, 241, 109, 235))

    overlay = overlay.resize(out.size, Image.Resampling.LANCZOS)
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
            --bg: #101214;
            --card: rgba(255,255,255,0.055);
            --card-border: rgba(255,255,255,0.10);
            --text: #f5efe6;
            --muted: #cdbfae;
            --accent: #d1a15f;
            --accent-2: #b36c44;
            --danger: #ff7f7f;
            --ok: #61e6a5;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(100,185,210,0.08), transparent 22%),
                radial-gradient(circle at top right, rgba(214,161,95,0.12), transparent 20%),
                linear-gradient(180deg, #0f1113 0%, #171a1d 100%);
        }
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
        h1, h2, h3, h4, h5, h6, p, li, span, label, div {color: var(--text);}

        section[data-testid="stSidebar"] {
            background: #b56d45 !important;
            border-right: 1px solid rgba(83, 41, 9, 0.16) !important;
        }
        section[data-testid="stSidebar"] > div {
            background: #b56d45 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #1f1510 !important;
        }
        section[data-testid="stSidebar"] .sidebar-card {
            background: rgba(248, 236, 224, 0.82) !important;
            border: 1px solid rgba(83, 41, 9, 0.14) !important;
            box-shadow: 0 10px 26px rgba(82, 39, 11, 0.12) !important;
            backdrop-filter: blur(8px);
        }
        section[data-testid="stSidebar"] .mini-note,
        section[data-testid="stSidebar"] .muted-note,
        section[data-testid="stSidebar"] .stat-caption,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {
            color: #2b1d14 !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: #f4e7db !important;
            color: #1f1510 !important;
            border: 1px solid rgba(83,41,9,0.18) !important;
            box-shadow: 0 8px 18px rgba(91, 47, 12, .10) !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] *,
        section[data-testid="stSidebar"] div[data-baseweb="select"] *,
        section[data-testid="stSidebar"] [data-baseweb="select"] input,
        section[data-testid="stSidebar"] [data-baseweb="select"] span,
        section[data-testid="stSidebar"] [data-baseweb="select"] svg,
        section[data-testid="stSidebar"] [data-baseweb="select"] path {
            color: #1f1510 !important;
            fill: #1b140e !important;
            stroke: #1b140e !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        div[data-baseweb="popover"] {
            z-index: 999999 !important;
        }
        div[data-baseweb="popover"] *,
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] div,
        div[data-baseweb="popover"] span,
        div[data-baseweb="popover"] [role="listbox"],
        div[data-baseweb="popover"] [role="option"] {
            background: #f4e7db !important;
            color: #1f1510 !important;
            fill: #1b140e !important;
            stroke: #1b140e !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        div[data-baseweb="popover"] [role="option"]:hover,
        div[data-baseweb="popover"] li:hover {
            background: rgba(200,110,58,.22) !important;
            color: #1f1510 !important;
        }
        div[data-baseweb="popover"] [role="option"][aria-selected="true"],
        div[data-baseweb="popover"] [aria-selected="true"] {
            background: rgba(200,110,58,.30) !important;
            color: #1f1510 !important;
            font-weight: 700 !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: rgba(255,244,234,0.96) !important;
            color: #111111 !important;
            border: 1px solid rgba(0,0,0,0.14) !important;
            box-shadow: 0 8px 18px rgba(0,0,0,.08) !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] *,
        section[data-testid="stSidebar"] div[data-baseweb="select"] *,
        section[data-testid="stSidebar"] [data-baseweb="select"] input,
        section[data-testid="stSidebar"] [data-baseweb="select"] span,
        section[data-testid="stSidebar"] [data-baseweb="select"] svg,
        section[data-testid="stSidebar"] [data-baseweb="select"] path {
            color: #111111 !important;
            fill: #111111 !important;
            stroke: #111111 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        div[data-baseweb="popover"] {
            z-index: 999999 !important;
        }
        div[data-baseweb="popover"] *,
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] div,
        div[data-baseweb="popover"] span,
        div[data-baseweb="popover"] [role="listbox"],
        div[data-baseweb="popover"] [role="option"] {
            background: #fff6ee !important;
            color: #111111 !important;
            fill: #111111 !important;
            stroke: #111111 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        div[data-baseweb="popover"] [role="option"]:hover,
        div[data-baseweb="popover"] li:hover {
            background: rgba(200,110,58,.22) !important;
            color: #111111 !important;
        }
        div[data-baseweb="popover"] [role="option"][aria-selected="true"],
        div[data-baseweb="popover"] [aria-selected="true"] {
            background: rgba(200,110,58,.30) !important;
            color: #111111 !important;
            font-weight: 700 !important;
        }
        [data-testid="stFileUploader"] section {
            border: 1px dashed rgba(255,255,255,.22) !important;
            border-radius: 22px !important;
            background: linear-gradient(135deg, rgba(255,255,255,.05), rgba(255,255,255,.02)) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 20px 60px rgba(0,0,0,.18) !important;
            padding: 1rem 1rem !important;
        }
        [data-testid="stFileUploader"] section button {
            background: linear-gradient(135deg, #c67843, #b86a38) !important;
            color: #111111 !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 24px rgba(198,120,67,.28) !important;
        }
        .stDownloadButton button,
        .stButton button,
        button[kind="secondary"],
        button[kind="primary"] {
            background: linear-gradient(135deg, #c67843, #b86a38) !important;
            color: #111111 !important;
            border: 1px solid rgba(35,22,14,0.18) !important;
            border-radius: 999px !important;
            font-weight: 800 !important;
            box-shadow: 0 10px 24px rgba(198,120,67,.28) !important;
            opacity: 1 !important;
        }
        .stDownloadButton button:hover,
        .stButton button:hover,
        button[kind="secondary"]:hover,
        button[kind="primary"]:hover {
            background: linear-gradient(135deg, #d2864d, #c67843) !important;
            color: #0f0f0f !important;
            box-shadow: 0 14px 28px rgba(198,120,67,.34) !important;
        }
        .stDownloadButton button p,
        .stButton button p,
        .stDownloadButton button span,
        .stButton button span {
            color: #111111 !important;
            font-weight: 800 !important;
        }
        [data-testid="stFileUploader"] small {
            color: var(--muted) !important;
        }
        .stat-card, .summary-card, .panel-card, .score-box, .upload-panel {
            transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
        }
        .stat-card:hover, .summary-card:hover, .panel-card:hover, .score-box:hover {
            transform: translateY(-2px);
            box-shadow: 0 24px 56px rgba(0,0,0,.24);
            border-color: rgba(255,255,255,.18);
        }
        .hero {
            position: relative;
            overflow: hidden;
        }
        .hero:before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 0% 0%, rgba(104,213,255,.14), transparent 28%),
                        radial-gradient(circle at 100% 0%, rgba(248,214,109,.14), transparent 25%);
            pointer-events: none;
        }
        .hero > * {
            position: relative;
            z-index: 1;
        }
        .section-title {
            letter-spacing: .01em;
        }
        .panel-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: .55rem;
        }

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
        .compact-info-card {
            min-height: 118px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            overflow: visible;
        }
        .compact-info-title {
            font-size: .83rem;
            color: var(--muted);
            margin-bottom: .5rem;
            line-height: 1.2;
            white-space: normal;
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        .compact-info-value {
            font-size: .96rem;
            font-weight: 800;
            line-height: 1.18;
            white-space: normal;
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        .compact-info-caption {
            margin-top: .45rem;
            font-size: .74rem;
            color: var(--muted);
            line-height: 1.25;
            white-space: normal;
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        section[data-testid="stSidebar"] .sidebar-card .ghost-badge {
            background: rgba(255,255,255,0.52) !important;
            color: #2b1d14 !important;
            border-color: rgba(83,41,9,.14) !important;
        }
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
    st.markdown("<div class='section-title'>Değerlendirme Puanları</div>", unsafe_allow_html=True)
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


def render_sidebar(selected_mode: str, selected_editor_mode: str, selected_editor_name: str) -> None:
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
        st.markdown(f"<div class='mini-note'>Yukarıdaki seçili editörün yorumu görünür.</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Manifesto sorusu**")
        st.info(CULTURE["temel_soru"])

        st.markdown("**Kısa kullanım**")
        st.markdown(
            "- Fotoğrafı yükle\n- Tür, ton ve editörü seç\n- Editör yorumlarını karşılaştır\n- Isı Haritası, göz akışı ve altın oran katmanlarını incele\n- Çekim ve düzenleme notlarını uygula"
        )
        st.markdown("<div style='margin-top:1rem;padding-top:.9rem;border-top:1px solid rgba(73,36,8,.18);font-size:.82rem;font-weight:700;letter-spacing:.03em;color:#3a1b08;opacity:.95;'>Serdar Bayram™</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _editor_scene_bits(result: CritiqueResult) -> Dict[str, str]:
    profile = (result.metrics or {}).get("scene_profile", {}) if isinstance(result.metrics, dict) else {}
    subject = profile.get("subject_hint", "ana özne")
    primary = profile.get("primary_region", "merkez")
    secondary = profile.get("secondary_region", "çevre")
    distraction = profile.get("distraction_region", "yan alan")
    mood = profile.get("visual_mood", "sessiz")
    light = profile.get("light_character", "yumuşak")
    complexity = profile.get("complexity_level", "orta yoğunlukta")
    scene = profile.get("scene_type", result.suggested_mode or "Sokak")
    detail_1 = profile.get("concrete_detail_1", "")
    detail_2 = profile.get("concrete_detail_2", "")
    detail_3 = profile.get("concrete_detail_3", "")
    detail_signature = profile.get("detail_signature", "")
    return {
        "subject": subject,
        "primary": primary,
        "secondary": secondary,
        "distraction": distraction,
        "mood": mood,
        "light": light,
        "complexity": complexity,
        "scene": scene,
        "detail_1": detail_1,
        "detail_2": detail_2,
        "detail_3": detail_3,
        "detail_signature": detail_signature,
    }


def _pick_by_seed(options: List[str], seed_text: str) -> str:
    if not options:
        return ""
    idx = abs(hash(seed_text)) % len(options)
    return options[idx]


def _region_to_place(region: str) -> str:
    mapping = {
        "merkez": "tam merkezde",
        "sol merkez": "sol tarafta",
        "sağ merkez": "sağ tarafta",
        "üst merkez": "üst hatta",
        "alt merkez": "alt hatta",
        "üst sol": "üst solda",
        "üst sağ": "üst sağda",
        "alt sol": "alt solda",
        "alt sağ": "alt sağda",
    }
    return mapping.get(region, region)


def _light_sentence(metrics: Dict, light: str, primary: str) -> str:
    left = float(metrics.get("left_brightness", 0))
    right = float(metrics.get("right_brightness", 0))
    top = float(metrics.get("top_brightness", 0))
    bottom = float(metrics.get("bottom_brightness", 0))
    hi = float(metrics.get("highlight_clip_ratio", 0))

    if abs(top - bottom) > max(abs(left - right), 10):
        if top > bottom:
            return f"Işık daha çok üstten iniyor ve {primary} hattını belirginleştiriyor."
        return f"Işık alttan daha koyu bir basınç kuruyor; üst bölüm daha çok nefes alıyor."
    if abs(left - right) > 10:
        if left > right:
            return f"Işık soldan daha açık geliyor; bakışı doğal olarak o tarafa topluyor."
        return f"Işık sağdan açılıyor; gözün ilk yönelimi biraz da bu yüzden oraya gidiyor."
    if hi > 0.03:
        return "Parlak alanlar yer yer sertleşse de ışık sahnenin omurgasını kuruyor."
    if "sert" in light:
        return "Işık biraz sert ama bu sertlik fotoğrafa diri bir yapı veriyor."
    return "Işık yumuşak akıyor; sahnenin duygusunu bastırmadan taşıyor."


def _flow_sentence(primary: str, secondary: str, distraction: str, metrics: Dict) -> str:
    tension = float(metrics.get("dynamic_tension_score", 0))
    symmetry = float(metrics.get("symmetry_score", 0))
    if symmetry > 0.72:
        return f"Göz {primary} ile {secondary} arasında sakin dolaşıyor; düzen duygusu güçlü."
    if tension > 0.58:
        return f"Göz önce {primary}e oturuyor, sonra {secondary}e kayıyor; akış canlı ama tam sakin değil."
    return f"Göz {primary}ten açılıp {secondary}e uzanıyor; ama {distraction} zaman zaman bu hattı bölüyor."


def _subject_sentence(subject: str, primary: str, face_count: int) -> str:
    place = _region_to_place(primary)
    if face_count >= 1:
        return f"{place.capitalize()} görünen yüz, fotoğrafın en sahici kapısı gibi çalışıyor."
    if "figür" in subject:
        return f"{place.capitalize()} duran figür, kadrajın taşıdığı ana yükü üstleniyor."
    if "boşluk" in subject or "biçim" in subject:
        return f"{place.capitalize()} kurulan biçim ve boşluk ilişkisi, kareyi ilk bakışta ayakta tutuyor."
    return f"{place.capitalize()} duran ana ağırlık merkezi, fotoğrafın ilk cümlesini kuruyor."







def _build_scene_relation_phrase(result: CritiqueResult, bits: Dict[str, str]) -> str:
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    profile = (metrics or {}).get("scene_profile", {}) if isinstance(metrics, dict) else {}

    primary = bits.get("primary", "merkez")
    secondary = bits.get("secondary", "çevre")
    distraction = bits.get("distraction", "çevre")
    subject = (bits.get("subject") or "ana unsur").lower()
    light = bits.get("light", "")
    complexity = bits.get("complexity", "orta")
    scene = bits.get("scene", "Genel")
    mood = bits.get("mood", "sakin")

    face_count = int(profile.get("face_count", 0) or 0)
    edge_density = float(metrics.get("edge_density", 0) or 0)
    negative_space = float(metrics.get("negative_space_score", 0) or 0)
    tension = float(metrics.get("dynamic_tension_score", 0) or 0)
    symmetry = float(metrics.get("symmetry_score", 0) or 0)

    def p(region: str) -> str:
        return _region_to_place(region)

    subject_role = "ana unsur"
    if face_count >= 1:
        subject_role = "ana figür"
    elif "figür" in subject or "insan" in subject:
        subject_role = "ana figür"
    elif "boşluk" in subject:
        subject_role = "boşluk duygusu"
    elif "biçim" in subject or "form" in subject:
        subject_role = "biçim duygusu"
    elif "ışık" in subject:
        subject_role = "ışık vurgusu"

    relations = []

    # Core spatial relationships
    relations.extend([
        f"{p(primary)} tarafta toplanan ağırlık ile {p(secondary)} açılan alan arasında belirgin bir ilişki kurulmuş",
        f"ana vurgu ile çevredeki ikinci katman birbirini itmeden aynı karede yaşayabiliyor",
        f"öndeki asıl ağırlık ile geride kalan alan arasında okunur bir denge var",
        f"kadrajdaki ana vurgu, çevresindeki alanı bastırmadan onunla birlikte çalışıyor",
    ])

    # Subject-role based but generic
    if subject_role == "ana figür":
        relations.extend([
            f"ana figür ile çevredeki hayat arasında canlı bir temas hissi oluşuyor",
            f"figürün topladığı dikkat ile geride kalan bağlam birbirine yaslanıyor",
            f"insan varlığı ile çevrenin ağırlığı aynı cümlede buluşuyor",
        ])
    elif subject_role == "boşluk duygusu":
        relations.extend([
            f"boş bırakılan alan ile ana vurgu arasında sakin ama etkili bir gerilim var",
            f"geniş boşluk, ana etkinin daha yalnız ve daha görünür kalmasını sağlıyor",
            f"boşluk duygusu ile küçük vurgu birbirini büyüten bir ilişki kuruyor",
        ])
    elif subject_role == "biçim duygusu":
        relations.extend([
            f"tekrarlayan biçimler ile ana vurgu arasında düzenli bir akış oluşmuş",
            f"çizgi ve kütle ilişkisi, sahnenin ana etkisini sessizce büyütüyor",
            f"formlar ile asıl vurgu birbirini desteklediği için görüntü daha tutarlı duruyor",
        ])
    elif subject_role == "ışık vurgusu":
        relations.extend([
            f"ışığın toplandığı bölge ile koyu kalan alanlar arasında iyi bir karşılık var",
            f"aydınlık ve koyuluk arasındaki ilişki, sahnenin ana duygusunu taşıyor",
            f"ışığın vurduğu yer ile geri çekilen alanlar arasında temiz bir denge kurulmuş",
        ])
    else:
        relations.extend([
            f"ana unsur ile çevresindeki yapı birbirine karşı durmak yerine birbirini taşıyor",
            f"asıl vurgu ile yan alanlar arasında kontrollü bir gerilim oluşuyor",
            f"fotoğrafın merkezindeki etki ile çevredeki yapı aynı duyguda buluşuyor",
        ])

    # Density / clutter
    if complexity == "yoğun":
        relations.extend([
            f"ana etki ile {p(distraction)} tarafındaki görsel ses arasında sürekli bir çekişme var",
            f"öndeki vurgu ile çevredeki yoğunluk yan yana duruyor; bu da kareye hareket veriyor",
            f"kalabalık yapı ile ana vurgu birbirini zorlasa da sahneyi canlı tutuyor",
        ])
    elif complexity == "sade":
        relations.extend([
            f"çevrede bırakılan sakin alan, ana vurgunun daha rahat duyulmasını sağlıyor",
            f"fazla ses olmaması, ana etkinin görüntü içinde daha temiz kalmasına yardım ediyor",
            f"kadrajın nefesli kalması, asıl ilişkinin daha doğrudan hissedilmesini sağlıyor",
        ])

    # Light relationships
    if "sert" in light:
        relations.extend([
            f"ışığın sertliği, ana vurgu ile arka plan arasındaki ayrımı daha görünür kılıyor",
            f"parlaklık ile gölge arasındaki karşıtlık, sahnedeki ilişkiyi daha belirgin hale getiriyor",
            f"ışığın keskin davranışı, görüntüdeki esas ağırlığı daha hızlı ortaya çıkarıyor",
        ])
    else:
        relations.extend([
            f"yumuşak ışık, ana vurgu ile çevresindeki tonları aynı duygu içinde topluyor",
            f"ışığın sakin akışı, öndeki etki ile geride kalan alanı birbirine bağlıyor",
            f"yumuşak ton geçişleri, sahnenin ilişkilerini daha kırmadan duyuruyor",
        ])

    # Composition / tension
    if negative_space > 0.58:
        relations.extend([
            f"geniş boşluk ile küçük vurgu arasındaki ilişki, kareyi daha düşündürücü hale getiriyor",
            f"az ögeyle kurulan denge, sahnenin duygusunu daha açık bırakıyor",
        ])
    if tension > 0.62:
        relations.extend([
            f"yerleşimdeki hafif dengesizlik, fotoğrafa diri bir gerilim katıyor",
            f"ana ağırlığın tam oturmaması bile görüntüyü daha canlı kılıyor",
        ])
    if symmetry > 0.72:
        relations.extend([
            f"dengeli yerleşim ile ana vurgu arasındaki ilişki fotoğrafa sakin bir düzen veriyor",
            f"yerleşimin toplu durması, sahnenin etkisini daha derli toplu hissettiriyor",
        ])
    if edge_density > 0.14:
        relations.extend([
            f"çizgisel yoğunluk ile ana vurgu yan yana gelince kare daha hareketli okunuyor",
            f"detayların fazlalığı ile asıl etki arasında canlı ama riskli bir temas oluşuyor",
        ])

    # Scene-specific meaning without naming fixed objects
    if scene == "Portre":
        relations.extend([
            f"özne ile arkasında kalan tonlar birbirini boğmadan aynı duyguda buluşuyor",
            f"figürün taşıdığı duygu ile çevrenin sessizliği aynı çizgide ilerliyor",
        ])
    elif scene == "Sokak":
        relations.extend([
            f"ön taraftaki vurgu ile arkada akan hayat arasında sahici bir temas var",
            f"gündelik akış ile tekil vurgu aynı karede kısa ama güçlü bir bağ kuruyor",
        ])
    elif scene == "Belgesel":
        relations.extend([
            f"insan izi ile sahnenin bağlamı birbirinden kopmadan birlikte çalışıyor",
            f"görünen şey ile hissettirilen arka hikâye aynı fotoğrafın içinde buluşuyor",
        ])
    elif scene == "Soyut":
        relations.extend([
            f"biçim, ton ve boşluk bir araya gelerek doğrudan açıklanmayan bir etki kuruyor",
            f"görüntüdeki asıl ilişki nesnelerden çok yüzey ve ritim üzerinden çalışıyor",
        ])

    # Mood-sensitive relation
    if mood in {"gerilimli", "yoğun"}:
        relations.extend([
            f"sahnedeki sıkışma hissi ile ana vurgu birbirini beslediği için görüntü akılda kalıyor",
            f"duygudaki baskı, kompozisyondaki ilişkilere de yansıyor ve kareyi diri tutuyor",
        ])
    else:
        relations.extend([
            f"sahnenin sakin hali, ana ilişkiyi daha sessiz ama daha kalıcı hale getiriyor",
            f"duygunun fazla yükselmemesi, küçük ilişkilerin daha rahat fark edilmesini sağlıyor",
        ])

    seed = f"scenerel-{subject_role}-{primary}-{secondary}-{distraction}-{scene}-{complexity}-{light}-{mood}-{face_count}-{edge_density:.3f}-{negative_space:.3f}-{tension:.3f}"
    return _pick_by_seed(relations, seed)


def _finish_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text if text[-1] in ".!?" else text + "."


def _voice_pick(editor_name: str, field: str, seed: str) -> str:
    engine = EDITOR_VOICE_ENGINES.get(editor_name, {})
    options = engine.get(field, []) if isinstance(engine, dict) else []
    return _pick_by_seed([str(x) for x in options if str(x).strip()], f"{editor_name}-{field}-{seed}")


def _voice_lexicon(editor_name: str, index: int) -> str:
    engine = EDITOR_VOICE_ENGINES.get(editor_name, {})
    words = engine.get("lexicon", []) if isinstance(engine, dict) else []
    if not words:
        return ""
    return str(words[index % len(words)])


def _normalize_phrase(text: str) -> str:
    return " ".join(str(text or "").replace("..", ".").split()).strip()


def _trim_sentence(text: str) -> str:
    text = _normalize_phrase(text)
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _scene_observation_bank(profile: Optional["SceneProfile"]) -> List[str]:
    if profile is None:
        return []
    details = [
        f"Ana ağırlık {profile.subject_position} duruyor.",
        f"Göz ilk olarak {profile.primary_region} bölgesine oturuyor.",
        f"İkinci dikkat hattı {profile.secondary_region} tarafında açılıyor.",
        f"Işık {profile.light_type_detail} bir karakter taşıyor.",
        f"Mekân {profile.environment_type} duygusu veriyor.",
        f"Sahnede hissedilen hareket {profile.human_action} tarafında yoğunlaşıyor.",
        f"Kadrajın baskın geometrisi {profile.dominant_shape} çizgisinde kuruluyor.",
        f"Genel gerilim düzeyi {profile.visual_tension_level} hissediliyor.",
        f"Dokusal iz olarak {profile.historical_texture_hint} öne çıkıyor.",
        _trim_sentence(profile.concrete_detail_1),
        _trim_sentence(profile.concrete_detail_2),
        _trim_sentence(profile.concrete_detail_3),
    ]
    clean = []
    seen = set()
    for d in details:
        d = _normalize_phrase(d)
        if len(d) < 8:
            continue
        key = d.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(_trim_sentence(d))
    return clean


def _pick_observations_for_editor(editor_name: str, profile: Optional["SceneProfile"]) -> List[str]:
    obs = _scene_observation_bank(profile)
    if not obs:
        return []
    preferred = {
        "Selahattin Kalaycı": [0, 8, 3, 5, 9],
        "Güler Ataşer": [3, 8, 4, 10, 11],
        "Sevgin Cingöz": [0, 1, 2, 6, 7],
        "Mürşide Çilengir": [5, 9, 10, 0, 4],
        "Gülcan Ceylan Çağın": [0, 2, 4, 6, 8],
    }.get(editor_name, [0, 1, 2, 3])
    picked = []
    for i in preferred:
        if i < len(obs):
            phrase = obs[i]
            if phrase not in picked:
                picked.append(phrase)
        if len(picked) >= 3:
            break
    return picked or obs[:3]


def _score_signal(scores: Dict[str, float], key: str, good: str, mid: str, weak: str) -> str:
    value = float(scores.get(key, 0))
    if value >= 72:
        return good
    if value >= 58:
        return mid
    return weak


def _editor_focus_sentence(editor_name: str, profile: Optional["SceneProfile"], scores: Dict[str, float]) -> str:
    profile = profile or SceneProfile(
        scene_type="belirsiz", subject_hint="ana özne", visual_mood="nötr", light_character="yumuşak",
        complexity_level="orta", balance_character="yarı dengeli", primary_region="merkez", secondary_region="çevre",
        distraction_region="yan alan", face_count=0, human_presence_score=0.0, main_subject_confidence=0.0,
        concrete_detail_1="", concrete_detail_2="", concrete_detail_3="", detail_signature="",
        subject_position="merkezde", light_type_detail="yumuşak", environment_type="gündelik çevre",
        human_action="durağan", dominant_shape="yumuşak akış", visual_tension_level="orta", historical_texture_hint="dokusal iz"
    )
    if editor_name == "Selahattin Kalaycı":
        return f"Ben burada yalnızca görüneni değil, {profile.subject_hint} etrafında kurulan niyeti de okuyorum."
    if editor_name == "Güler Ataşer":
        return f"Bu karenin etkisi bende önce {profile.light_type_detail} ışıkla oluşan havadan geçiyor."
    if editor_name == "Sevgin Cingöz":
        return f"Kompozisyon tarafında belirleyici olan şey, {profile.primary_region} ile {profile.secondary_region} arasındaki yerleşim kararı."
    if editor_name == "Mürşide Çilengir":
        return f"Fotoğrafın bana geçtiği yer teknik tarafı değil, {profile.subject_hint} çevresinde biriken insani yakınlık."
    return f"Seçki açısından bakınca bu karenin ilk artısı, {profile.primary_region} çevresinde kurduğu net görsel merkez."


def _editor_issue_sentence(editor_name: str, profile: Optional["SceneProfile"], scores: Dict[str, float]) -> str:
    profile = profile or None
    if editor_name == "Gülcan Ceylan Çağın":
        return _score_signal(
            scores,
            "editoryal_deger",
            "Bence bu kare seçki içinde yer açabilecek bir çekirdek taşıyor; sadece son bir editoryal sıkılaşma onu daha görünür yapar.",
            "Şu haliyle seçkiye yaklaşan bir tarafı var; fakat kararın daha net duyulması için küçük bir sadeleşme iyi gelir.",
            f"Benim mesafem şu noktada başlıyor: {getattr(profile, 'distraction_region', 'yan alan')} tarafındaki fazlalık geri çekilirse karar çok daha ikna edici olur."
        )
    if editor_name == "Sevgin Cingöz":
        return _score_signal(
            scores,
            "kompozisyon",
            "İskelet yerinde; yalnız akışı biraz daha disipline etmek fotoğrafı belirgin biçimde yükseltir.",
            "Yapı çalışıyor ama yer yer gevşiyor; özellikle gözün döndüğü ikinci hattı biraz netleştirmek gerekir.",
            f"Sorun ilham eksikliği değil; {getattr(profile, 'secondary_region', 'ikinci hat')} ile {getattr(profile, 'distraction_region', 'yan alan')} arasındaki denge daha sıkı kurulmalı."
        )
    if editor_name == "Güler Ataşer":
        return _score_signal(
            scores,
            "duygusal_yogunluk",
            "Bunu çok zorlamadan korursan fotoğrafın şiirselliği kendi başına kalır.",
            "Etki güzel ama biraz daha temizlik istediği için bazı yerlerde hava dağılıyor.",
            f"Özellikle {getattr(profile, 'distraction_region', 'yan tarafta')} biriken ses hafifçe geri itilirse sahnenin nefesi daha saf duyulur."
        )
    if editor_name == "Mürşide Çilengir":
        return _score_signal(
            scores,
            "duygusal_yogunluk",
            "İnsani temas kurulmuş; şimdi bunu biraz daha açık ve temiz bırakmak yeterli.",
            "Duygu geliyor ama tam yerleşmiyor; küçük bir sadeleşme sahnenin kalbini daha görünür kılar.",
            f"Ben daha çok {getattr(profile, 'subject_hint', 'ana özne')} çevresindeki kırılgan alanın önüne geçen gereksiz görsel sesi azaltırdım."
        )
    return _score_signal(
        scores,
        "anlati_gucu",
        "Fotoğraf düşündürüyor; şimdi yalnızca iç cümlesini biraz daha berraklaştırması gerekiyor.",
        "Kare bir şey söylüyor ama bunu daha açık bir iç ritme kavuşturabilir.",
        f"Benim itirazım, {getattr(profile, 'subject_hint', 'ana özne')} çevresindeki anlamın henüz tam açıklık kazanmamış olması."
    )


def _compose_v8_editor_comment(editor_name: str, result: CritiqueResult) -> str:
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    profile_data = metrics.get("scene_profile", {}) if isinstance(metrics, dict) else {}
    profile = SceneProfile(**profile_data) if profile_data else None
    scores = result.rubric_scores or {}

    if profile is None:
        return build_editor_comment(editor_name, result)

    observations = _pick_observations_for_editor(editor_name, profile)
    obs1 = _trim_sentence(observations[0] if len(observations) > 0 else profile.concrete_detail_1 or f"Ana ağırlık {profile.subject_position} duruyor")
    obs2 = _trim_sentence(observations[1] if len(observations) > 1 else profile.concrete_detail_2 or f"Göz ilk olarak {profile.primary_region} bölgesine gidiyor")
    obs3 = _trim_sentence(observations[2] if len(observations) > 2 else profile.concrete_detail_3 or f"Işık {profile.light_type_detail} bir karakter taşıyor")

    person_phrase = (
        f"Kadrajda {profile.face_count} belirgin insan yüzü okunuyor" if profile.face_count >= 2 else
        "Kadrajda tek belirgin insan varlığı öne çıkıyor" if profile.face_count == 1 else
        f"İnsan yerine {profile.subject_hint} öne çıkıyor"
    )
    issue_sentence = _editor_issue_sentence(editor_name, profile, scores)
    opening = _voice_pick(editor_name, "openings", f"v95-open-{editor_name}-{profile.detail_signature}")
    closing = _voice_pick(editor_name, "closings", f"v95-close-{editor_name}-{profile.detail_signature}")

    if editor_name == "Selahattin Kalaycı":
        parts = [
            opening,
            obs1,
            f"{obs2[:-1]} ve bu yerleşim bana fotoğrafın neden özellikle {profile.subject_position} sıkıştığını sorduruyor.",
            f"{person_phrase}; {profile.environment_type} ile kurduğu temas görüntüyü yalnız bir kayıt olmaktan çıkarıyor.",
            issue_sentence,
            closing,
        ]
    elif editor_name == "Güler Ataşer":
        parts = [
            opening,
            obs3,
            f"{obs1[:-1]}; bu yüzden sahnenin havası ışıkla birlikte sertleşmeden duyuluyor.",
            f"{profile.historical_texture_hint.capitalize()} ve {profile.secondary_region} tarafındaki ikinci hat kareye ince bir doku nefesi veriyor.",
            issue_sentence,
            closing,
        ]
    elif editor_name == "Sevgin Cingöz":
        parts = [
            opening,
            obs1,
            obs2,
            f"Kompozisyonun asıl kararı, {profile.primary_region} ile {profile.secondary_region} arasında kurulan taşıyıcı hatta dayanıyor; {profile.distraction_region} tarafı ise kolayca fazlalığa dönebiliyor.",
            issue_sentence,
            closing,
        ]
    elif editor_name == "Mürşide Çilengir":
        parts = [
            opening,
            f"{person_phrase}; bu yüzden fotoğrafın kalbi teknik taraftan çok varlığın kadrajda taşıdığı sessizlikte atıyor.",
            obs3,
            f"{obs2[:-1]} ve bu küçük yön değişimi sahneye insani bir kırılganlık ekliyor.",
            issue_sentence,
            closing,
        ]
    else:
        parts = [
            opening,
            f"Önce çalışan tarafı teslim edeyim: {obs1[0].lower() + obs1[1:]}",
            obs2,
            f"Editoryal açıdan ana merkez {profile.primary_region} hattında kurulmuş durumda; fakat {profile.distraction_region} tarafındaki enerji ayıklanırsa karar çok daha temiz duyulur.",
            issue_sentence,
            closing,
        ]

    final = []
    seen = set()
    for p in parts:
        p = _trim_sentence(p)
        low = p.lower()
        if len(p) < 14 or low in seen:
            continue
        seen.add(low)
        final.append(p)
    return " ".join(final[:6]).strip()


# Removed earlier duplicate definition of _deoverlap_editor_comments
def _compose_v5_editor_comment(editor_name: str, result: CritiqueResult) -> str:
    bits = _editor_scene_bits(result)
    profile_data = (result.metrics or {}).get("scene_profile", {}) if isinstance(result.metrics, dict) else {}
    profile = SceneProfile(**profile_data) if profile_data else None
    scores = result.rubric_scores or {}

    primary = bits["primary"]
    secondary = bits["secondary"]
    distraction = bits["distraction"]
    detail_1 = bits.get("detail_1", "ana görsel vurgu")
    detail_2 = bits.get("detail_2", "ikinci dikkat hattı")
    detail_3 = bits.get("detail_3", "yan bölgelerdeki enerji")
    mood = bits.get("mood", "sessiz")
    relation = _build_scene_relation_phrase(result, bits)
    score = float(result.total_score or 0)

    subject_position = getattr(profile, "subject_position", primary)
    light_type_detail = getattr(profile, "light_type_detail", bits.get("light", "yumuşak ışık"))
    environment_type = getattr(profile, "environment_type", "gündelik çevre")
    human_action = getattr(profile, "human_action", "sakin bir görsel hareket")
    dominant_shape = getattr(profile, "dominant_shape", "yumuşak akış")
    visual_tension = getattr(profile, "visual_tension_level", "orta")
    historical_texture_hint = getattr(profile, "historical_texture_hint", "mekân dokusu anlatıya geri plandan eşlik ediyor")

    hierarchy = float(scores.get("odak_ve_hiyerarsi", 0))
    simplicity = float(scores.get("sadelik", 0))
    narrative = float(scores.get("anlati_gucu", 0))
    editorial = float(scores.get("editoryal_deger", 0))
    emotional = float(scores.get("duygusal_yogunluk", 0))

    if hierarchy < 58:
        issue_core = f"Bakış önce {_region_to_place(primary)} toplanıyor ama orada daha kararlı bir merkez arıyor"
    elif simplicity < 58:
        issue_core = f"{_region_to_place(distraction).capitalize()} tarafındaki görsel ses geri çekildiğinde ana cümle daha temiz duyulur"
    elif narrative < 60:
        issue_core = "Fotoğraf bir şey söylüyor ama bunu daha belirgin bir iç cümleye dönüştürme şansı açık"
    elif editorial < 62:
        issue_core = "Karede çekirdek var; fakat seçki içinde daha net durması için son kararların sıkılaşması gerekiyor"
    else:
        issue_core = "Büyük bir kırılma yok; ihtiyaç küçük ama yerinde bir netleşme"

    seed = f"v5-{editor_name}-{primary}-{secondary}-{distraction}-{detail_1}-{detail_2}-{detail_3}-{score:.1f}"
    opening = _voice_pick(editor_name, "openings", seed)
    connector = _voice_pick(editor_name, "connectors", seed + "-connector") or "çünkü"
    closing = _voice_pick(editor_name, "closings", seed + "-closing")
    lex0 = _voice_lexicon(editor_name, 0)
    lex1 = _voice_lexicon(editor_name, 1)
    lex2 = _voice_lexicon(editor_name, 2)
    critic_line = _critic_sentence(editor_name, bits, profile, result)

    if editor_name == "Selahattin Kalaycı":
        sentences = [
            f"{opening} {detail_1.capitalize()} tam da bu yüzden yalnızca bir ayrıntı gibi durmuyor",
            f"{detail_2.capitalize()}; {connector} {_region_to_place(subject_position)} biriken {lex0}, fotoğrafın görünenden büyük bir {lex1} taşıdığını düşündürüyor",
            f"{light_type_detail.capitalize()} ışık ile {historical_texture_hint} yan yana gelince, {environment_type} bir kayıt olmaktan çıkıp daha düşünsel bir alana açılıyor",
            critic_line,
            f"Benim asıl itirazım şu: {issue_core}; özellikle {_region_to_place(distraction)} kalan enerji bu iç cümleyi yer yer dağıtıyor",
            closing,
        ]
    elif editor_name == "Güler Ataşer":
        sentences = [
            f"{opening} {detail_1} sahnenin {lex0} gibi yayılıyor",
            f"{light_type_detail.capitalize()} ışık, {_region_to_place(subject_position)} duran ağırlığı sertleştirmeden görünür kılıyor; {detail_2} da buna ince bir {lex1} ekliyor",
            f"{environment_type.capitalize()} içinde {historical_texture_hint}; bu yüzden fotoğrafın yüzeyinde hafif bir {lex2} kalıyor",
            critic_line,
            f"Beni zorlayan yer şu: {issue_core}; bakışı yoran küçük yükler temizlenirse bu atmosfer çok daha derin çalışır",
            closing,
        ]
    elif editor_name == "Sevgin Cingöz":
        sentences = [
            f"{opening} ana ağırlık {_region_to_place(subject_position)} ve {primary} hattında kurulmuş durumda",
            f"{detail_2.capitalize()}; {connector} göz akışı {primary} ile {secondary} arasında bütünüyle kopmuyor ve {dominant_shape} bir {lex0} oluşuyor",
            f"{light_type_detail.capitalize()} ışık ile {visual_tension} gerilim birlikte çalıştığında kompozisyon tamamen dağılmıyor; {detail_3} ikinci bir taşıyıcı hat gibi davranıyor",
            critic_line,
            f"Net sorun şu: {issue_core}; özellikle {_region_to_place(distraction)} biriken fazlalık ayıklanırsa kompozisyon kararı daha berrak görünür",
            closing,
        ]
    elif editor_name == "Mürşide Çilengir":
        sentences = [
            f"{opening} {detail_1} bana doğrudan bir {lex0} hissi veriyor",
            f"{human_action.capitalize()} duygusu ve {detail_2}, görüntünün yalnızca görünmesini değil insana değmesini de sağlıyor; orada küçük ama sahici bir {lex1} var",
            f"{environment_type.capitalize()} içinde {historical_texture_hint}; bence bu geri plandaki doku, fotoğrafın {lex2} tarafını güçlendiriyor",
            critic_line,
            f"Ben yine de şunu hissediyorum: {issue_core}; biraz daha açıklık kazandığında bu sessiz temas izleyicide daha uzun kalır",
            closing,
        ]
    else:
        if editorial >= 72 and hierarchy >= 65:
            verdict = "editoryal açıdan güçlü ve rahatlıkla değerlendirilebilir"
        elif editorial >= 58:
            verdict = "iyi bir yayın potansiyeli taşıyor ama biraz daha toparlanma istiyor"
        else:
            verdict = "çekirdeği değerli; seçki için birkaç kararın netleşmesi gerekiyor"
        positive_core = f"{detail_1.capitalize()} bu çalışmanın {lex0} tarafını gerçekten kuruyor"
        sentences = [
            f"{opening} {verdict}",
            positive_core,
            f"{detail_2.capitalize()} ile {_region_to_place(distraction)} kalan gereksiz yük aynı anda çalıştığı için seçki dengesi tam yerine oturmuyor; yine de bu gerilim çözülebilir",
            f"{light_type_detail.capitalize()} ışık, {environment_type} hissi ve {historical_texture_hint} çalışmaya bir {lex1} veriyor; yayın eşiği için ana vurgu biraz daha net ayrışmalı",
            critic_line,
            f"Benim editoryal notum şu: {issue_core}; burada mesele sadece duygu değil, aynı zamanda {lex2} ve eleme disiplinini biraz daha berraklaştırmak",
            closing,
        ]

    cleaned = []
    for sentence in sentences:
        tone_mode = "Yapıcı" if editor_name in {"Mürşide Çilengir", "Gülcan Ceylan Çağın"} else "Dürüst"
        s = _finish_sentence(tone_text(sentence, tone_mode))
        if editor_name == "Gülcan Ceylan Çağın":
            s = _soften_gulcan_comment(s)
        s = _apply_editor_signature(editor_name, s)
        if s and s not in cleaned:
            cleaned.append(s)
    return " ".join(cleaned[:5])


def build_editor_comment_legacy(editor_name: str, result: CritiqueResult) -> str:
    bits = _editor_scene_bits(result)
    profile_data = (result.metrics or {}).get("scene_profile", {}) if isinstance(result.metrics, dict) else {}
    profile = SceneProfile(**profile_data) if profile_data else None
    scores = result.rubric_scores or {}

    primary = bits["primary"]
    secondary = bits["secondary"]
    distraction = bits["distraction"]
    mood = bits["mood"]
    light = bits["light"]
    complexity = bits["complexity"]
    detail_1 = bits.get("detail_1", "ana vurgu kendini gösteriyor")
    detail_2 = bits.get("detail_2", "ikinci bir dikkat hattı kadrajı destekliyor")
    detail_3 = bits.get("detail_3", "yan bölgelerde ayrı bir enerji birikiyor")
    detail_signature = bits.get("detail_signature", "")
    score = float(result.total_score or 0)

    subject_position = getattr(profile, "subject_position", primary)
    light_type_detail = getattr(profile, "light_type_detail", light)
    environment_type = getattr(profile, "environment_type", "gündelik çevre")
    human_action = getattr(profile, "human_action", "sakin bir görsel hareket")
    dominant_shape = getattr(profile, "dominant_shape", "yumuşak akış")
    visual_tension = getattr(profile, "visual_tension_level", "orta")
    historical_texture_hint = getattr(profile, "historical_texture_hint", "mekân dokusu geri planda ama tamamen silinmiş değil")

    def pick(options: List[str], salt: str) -> str:
        return _pick_by_seed(
            options,
            f"{editor_name}-{salt}-{primary}-{secondary}-{distraction}-{mood}-{light}-{complexity}-{detail_signature}-{score:.1f}",
        )

    def place(region: str) -> str:
        return _region_to_place(region)

    def capdot(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        return s[0].upper() + s[1:] + ("" if s.endswith(".") else ".")

    relation = capdot(_build_scene_relation_phrase(result, bits))

    hierarchy = float(scores.get("odak_ve_hiyerarsi", 0))
    simplicity = float(scores.get("sadelik", 0))
    narrative = float(scores.get("anlati_gucu", 0))
    editorial = float(scores.get("editoryal_deger", 0))
    emotional = float(scores.get("duygusal_yogunluk", 0))

    if hierarchy < 58:
        issue_core = f"Bakış önce {place(primary)} tarafına gidiyor ama orada daha kararlı bir merkez arıyor."
    elif simplicity < 58:
        issue_core = f"{place(distraction).capitalize()} tarafındaki görsel ses geri çekildiğinde ana cümle daha temiz duyulur."
    elif narrative < 60:
        issue_core = "Fotoğraf bir şey söylüyor ama bunu daha belirgin bir cümleye dönüştürme fırsatı hâlâ açık."
    elif editorial < 62:
        issue_core = "Karede çekirdek var; fakat seçki içinde daha net durması için son kararların sıkılaşması gerekiyor."
    else:
        issue_core = "Büyük bir kırılma yok; ihtiyaç küçük ama yerinde bir netleşme."

    if editor_name == "Selahattin Kalaycı":
        sentences = [
            pick([
                f"Bu kare neden ilk sözünü {subject_position} duran ağırlıkla söylüyor?",
                f"İnsan bu fotoğrafa bakınca önce şunu soruyor: {detail_1} burada niçin bu kadar belirleyici?",
            ], "sel-open"),
            pick([
                f"{detail_2.capitalize()}; bu yüzden görüntü yalnızca görünen şeyi değil, gizli niyeti de açıyor.",
                f"{relation} Özellikle {historical_texture_hint} duygusu fotoğrafı sıradan kayıttan çıkarıyor.",
            ], "sel-obs"),
            pick([
                f"{environment_type.capitalize()} içinde beliren {human_action}, sahneyi olay olmaktan çok düşünceye yaklaştırıyor.",
                f"{light_type_detail.capitalize()} ışık ile {dominant_shape} akış birleşince, kadrajın iç gerilimi görünenden daha derin hissediliyor.",
            ], "sel-meaning"),
            pick([issue_core, f"Benim itirazım teknikten çok şu noktaya: {place(distraction)} tarafındaki enerji, fotoğrafın asıl düşüncesini zaman zaman bastırıyor."], "sel-issue"),
            pick([
                "Ben bu karede değeri, neyi gösterdiğinden çok neden böyle kurulduğunu düşündürmesinde buluyorum.",
                "Bir parça daha netleşirse bu fotoğraf bakılıp geçilen değil, zihinde kalan bir kareye döner.",
            ], "sel-close"),
        ]
        return " ".join(sentences[:5])

    if editor_name == "Güler Ataşer":
        sentences = [
            pick([
                f"Beni içeri alan ilk şey {detail_1}; orada usulca açılan bir hava var.",
                f"Fotoğrafın bana ilk dokunan yeri {detail_2}; çünkü sahnenin soluğu orada duyuluyor.",
            ], "gul-open"),
            pick([
                f"{light_type_detail.capitalize()} ışık, {subject_position} duran ağırlığı sertleştirmeden belirginleştiriyor.",
                f"{detail_3.capitalize()}; bu yüzden görüntü kurulmuş olmaktan çok yaşanmış görünüyor.",
            ], "gul-obs"),
            pick([
                f"{environment_type.capitalize()} ve {historical_texture_hint}, karenin tenine sinmiş gibi duruyor.",
                f"{relation} Ben bu fotoğrafın duygusunu biraz da bu geri çekilmiş dokunun içinde buluyorum.",
            ], "gul-meaning"),
            pick([issue_core, "Bakışı yoran küçük yükler azaldığında bu şiirsellik daha derinden çalışır."], "gul-issue"),
            pick([
                "İçinde gerçek bir hava var; onu fazla bastırmadan korumak bu karenin en doğru yolu olur.",
                "Bu fotoğraf bağırmıyor ama iyi fotoğrafların bildiği o sessiz etkiyi taşıyor.",
            ], "gul-close"),
        ]
        return " ".join(sentences[:5])

    if editor_name == "Sevgin Cingöz":
        sentences = [
            pick([
                f"Yapısal olarak ilk veri net: ana ağırlık {subject_position} duruyor.",
                f"Bu kareyi taşıyan ilk karar, ana etkinin {subject_position} yerleşmiş olması.",
            ], "sev-open"),
            pick([
                f"{detail_2.capitalize()}; bu yüzden göz {place(primary)} ile {place(secondary)} arasında kontrollü dolaşıyor.",
                f"{dominant_shape.capitalize()} akış, bakışı tek noktada kilitlemek yerine kadraj boyunca yönlendiriyor.",
            ], "sev-obs"),
            pick([
                f"{light_type_detail.capitalize()} ışık ve {visual_tension} gerilim birbirini desteklediği için kompozisyon tamamen dağılmıyor.",
                f"{relation} Özellikle {detail_3}, ikinci katmanı kuran yapısal eleman gibi çalışıyor.",
            ], "sev-meaning"),
            pick([issue_core, f"Özellikle {place(distraction)} tarafındaki ağırlık toparlanırsa göz akışı daha berrak hale gelir."], "sev-issue"),
            pick([
                "Temel kurgu sağlam; bu yüzden burada mesele ilham değil, disiplinli bir son düzenleme.",
                "Ben bu karede çözülmez problem görmüyorum; yalnızca daha net verilmesi gereken kompozisyon kararları görüyorum.",
            ], "sev-close"),
        ]
        return " ".join(sentences[:5])

    if editor_name == "Mürşide Çilengir":
        sentences = [
            pick([
                f"Bu kare bende önce {detail_1} hissini bırakıyor.",
                f"İlk anda gözümden çok içimde kalan şey {detail_3} oluyor.",
            ], "mur-open"),
            pick([
                f"{human_action.capitalize()} hissi, fotoğrafın insani tarafını sessizce büyütüyor.",
                f"{detail_2.capitalize()}; bu yüzden görüntü yalnızca görünmüyor, insana değiyor da.",
            ], "mur-obs"),
            pick([
                f"{environment_type.capitalize()} içinde kalan {historical_texture_hint}, bu kırılganlığı daha inandırıcı yapıyor.",
                f"{relation} Ben bu karenin kalbini {subject_position} duran o hassas merkezde buluyorum.",
            ], "mur-meaning"),
            pick([issue_core, "Bir parça daha açıklık kazandığında bu sessiz etki izleyicide daha uzun kalır."], "mur-issue"),
            pick([
                "Ben yine de bu karenin bağırmadan konuşmasını çok kıymetli buluyorum.",
                "Fotoğrafın insani çekirdeği yerinde; gerisi onu biraz daha görünür bırakma meselesi.",
            ], "mur-close"),
        ]
        return " ".join(sentences[:5])

    if editor_name == "Gülcan Ceylan Çağın":
        verdict = "Şu haliyle seçkiye girebilir." if editorial >= 70 and hierarchy >= 65 else "Şu haliyle seçki sınırında." if editorial >= 58 else "Şu haliyle seçkiye almam."
        sentences = [
            pick([
                f"Benim ilk editoryal kararım net: {verdict}",
                f"Seçki mantığıyla bakınca ilk hükmüm şu: {verdict}",
            ], "gulcan-open"),
            pick([
                f"{detail_1.capitalize()} ve {light_type_detail} ışık, karenin hızlı okunmasını sağlıyor.",
                f"Ana ağırlığın {subject_position} durması, fotoğrafın ilk temasta elini güçlendiriyor.",
            ], "gulcan-obs"),
            pick([
                f"{environment_type.capitalize()} ile {historical_texture_hint}, bu kareye çıplak estetikten öte bir yayın zemini veriyor.",
                f"{dominant_shape.capitalize()} akış ve {visual_tension} gerilim düzeyi, seçkide tutunma ihtimalini artıran unsurlar.",
            ], "gulcan-meaning"),
            pick([issue_core, f"Özellikle {place(distraction)} tarafındaki fazlalık temizlenirse hükmüm daha hızlı olumluya döner."], "gulcan-issue"),
            pick([
                "Ben burada ciddiye alınacak bir çekirdek görüyorum; ama çekirdek ile bitmiş iş arasındaki farkı da açıkça görüyorum.",
                "Bu fotoğrafın yayın kararı, son temizlik ve karar sertliğiyle belirlenecek.",
            ], "gulcan-close"),
        ]
        return " ".join(sentences[:5])

    return "Bu karede bakışın tutunduğu bir yer var; küçük bir toparlama ile etkisi daha da büyüyebilir."

def build_editor_comment(editor_name: str, result: CritiqueResult) -> str:
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    vision = metrics.get("vision_analysis", {}) if isinstance(metrics, dict) else {}
    if isinstance(vision, dict):
        editor_comments = vision.get("editor_comments", {})
        if isinstance(editor_comments, dict):
            comment = str(editor_comments.get(editor_name, "")).strip()
            if comment:
                return comment
        scene_summary = str(vision.get("scene_summary", "")).strip()
        global_summary = str(vision.get("global_summary", "")).strip()
        if scene_summary and global_summary:
            return f"{scene_summary} {global_summary}".strip()
        if scene_summary:
            return scene_summary
    return _compose_grounded_editor_comment(editor_name, result)



def render_editor_snapshot_in_sidebar(selected_editor_name: str, result: CritiqueResult, ai_report: Optional[Dict] = None, placeholder=None) -> None:
    ai_comments = {}
    if isinstance(ai_report, dict) and isinstance(ai_report.get("editor_comments"), dict):
        ai_comments = ai_report.get("editor_comments") or {}

    value = ai_comments.get(selected_editor_name)
    if isinstance(value, str) and value.strip():
        comment = value.strip()
    else:
        comment = build_editor_comment(selected_editor_name, result)

    target = placeholder if placeholder is not None else st.sidebar
    with target.container():
        st.markdown("<div class='sidebar-card' style='margin-top:.7rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='mini-note' style='font-weight:700; margin-bottom:.45rem;'>Yukarıdaki seçili editörün yorumu görünür.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='mini-note' style='white-space: normal; line-height:1.75; color:#3a1b08;'>{escape(comment)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _human_count_phrase(profile: Optional["SceneProfile"]) -> str:
    if profile is None:
        return "insan varlığı belirsiz"
    if getattr(profile, "face_count", 0) <= 0:
        return "insan yerine sahnenin ana ağırlığı öne çıkıyor"
    if profile.face_count == 1:
        return "kadrajda tek belirgin insan figürü öne çıkıyor"
    if profile.face_count <= 3:
        return "kadrajda birkaç belirgin insan figürü okunuyor"
    return "kadraj kalabalık bir insan varlığı hissi taşıyor"


def _safe_detail(value: str, fallback: str) -> str:
    value = (value or '').strip()
    if not value:
        return fallback
    return value[0].upper() + value[1:] + ('' if value.endswith('.') else '.')


def _profile_distraction_text(profile: Optional["SceneProfile"]) -> str:
    if profile is None:
        return "belirgin dikkat kırılması yok"
    region = getattr(profile, "distraction_region", "yan bölgeler")
    if region in {"merkez", "orta"}:
        return "merkezde toplanan fazla enerji ana vurguyu zorluyor"
    return f"{region} tarafında biriken ikinci enerji ana cümleyi dağıtabiliyor"

def build_overlay_caption(profile: Optional["SceneProfile"]) -> str:
    if profile is None:
        return "Sarı halkalar ana odağın ve ikincil tutunma noktalarının toplandığı bölgeleri, kırmızı kutular ise ana cümleyi zorlayabilecek ikincil yükleri gösterir. İnce sarı çizgi bakışın muhtemel yönünü tarif eder."
    primary = getattr(profile, "primary_region", "merkez")
    secondary = getattr(profile, "secondary_region", "ikincil hat")
    distraction = getattr(profile, "distraction_region", "yan alan")
    return (
        f"Sarı halkalar ana görsel çağrının {primary} bölgesinde kurulduğunu, ikinci tutunmanın {secondary} tarafına uzadığını gösterir. "
        f"Kırmızı kutular ise {distraction} yönünde biriken ikincil yükü işaret eder; ince sarı çizgi bakışın muhtemel rotasını tarif eder."
    )


def build_heatmap_caption(profile: Optional["SceneProfile"]) -> str:
    if profile is None:
        return "Kırmızıya yaklaşan yoğunluk bakışın en hızlı toplandığı alanları; sarı ve camgöbeği geçişler görsel enerjinin sahneye nasıl yayıldığını gösterir."
    primary = getattr(profile, "primary_region", "merkez")
    secondary = getattr(profile, "secondary_region", "yan hat")
    distraction = getattr(profile, "distraction_region", "çevre")
    return (
        f"En sıcak yoğunluk {primary} bölgesindeki ana çağrıyı gösterir. Sarı ve camgöbeği geçişler enerjinin {secondary} hattına nasıl dağıldığını, "
        f"daha serin alanlar ise {distraction} yönünde kalan görsel art alanı tarif eder."
    )



# Removed earlier duplicate definition of _compose_grounded_editor_comment
def build_ai_reading_report(image_bytes: bytes, mode: str, editor_mode: str, result: CritiqueResult, use_deep_ai: bool = False) -> Dict:
    profile = SceneProfile(**extract_scene_profile_cached(image_bytes))
    scene_payload = build_scene_description_payload(ImageMetrics(**extract_metrics_cached(image_bytes)), profile)
    fallback = {
        "scene_summary": (
            f"Bu kareye bakınca göz önce {profile.primary_region} bölgesine oturuyor. "
            f"Ana vurgu {profile.subject_position} duruyor ve ışık {profile.light_type_detail} bir etki kuruyor. "
            f"Mekân {profile.environment_type} hissi verirken genel hava {profile.visual_mood} bir çizgide ilerliyor."
        ),
        "concrete_details": [
            f"Ana görsel ağırlık {profile.subject_position} ve {profile.primary_region} hattında toplanıyor.",
            f"İkinci dikkat hattı {profile.secondary_region} tarafında beliriyor.",
            f"Dikkati bölebilecek enerji en çok {profile.distraction_region} bölgesinde birikiyor.",
            f"Işık yapısı {profile.light_type_detail} karakter gösteriyor.",
            f"Mekân okuması {profile.environment_type} duygusu veriyor; {profile.historical_texture_hint}.",
        ],
        "meaning_layers": [
            f"Fotoğrafın ana cümlesi {profile.subject_hint} etrafında kuruluyor.",
            f"Görüntünün duygusu {profile.visual_mood} bir etki bırakıyor ve gerilim düzeyi {profile.visual_tension_level} hissediliyor.",
            f"Kadraj içindeki ilişki, {profile.human_action} ile çevresindeki alanın dengesi üzerinden okunuyor.",
        ],
        "scene_description": scene_payload,
        "editor_comments": _deoverlap_editor_comments({name: _compose_grounded_editor_comment(name, result) for name in EDITOR_NAMES}, profile),
        "global_summary": result.editor_summary,
        "one_line_caption": f"{profile.light_type_detail.capitalize()} ışığın içinde {profile.subject_hint} {profile.subject_position} hattında okunuyor.",
        "_source": "fallback",
        "_status": "fallback_active",
        "_status_label": "ÇOFSAT Motoru aktif",
        "_user_message": "Yorumlar şu an ÇOFSAT Motoru ile üretiliyor.",
        "_vision_ok": False,
        "_provider": "local",
    }

    # Bu sürümde OpenAI/Qwen birinci motordur; yalnızca kullanılamazsa ÇOFSAT fallback devreye girer.
    if not use_deep_ai:
        fallback["_status"] = "local_only"
        fallback["_status_label"] = "ÇOFSAT Motoru aktif"
        fallback["_user_message"] = "Yorumlar şu an ÇOFSAT Motoru ile üretiliyor."
        return fallback

    if openai_vision_available():
        payload = openai_vision_critique_cached(image_bytes, mode, editor_mode) or {}
        if isinstance(payload, dict) and payload and not payload.get("error"):
            merged = fallback.copy()
            for key in ["scene_summary", "concrete_details", "meaning_layers", "global_summary", "one_line_caption"]:
                value = payload.get(key)
                if value:
                    merged[key] = value
            if isinstance(payload.get("editor_comments"), dict) and payload["editor_comments"]:
                merged["editor_comments"] = _deoverlap_editor_comments(payload["editor_comments"], profile)
            merged["_source"] = "vision"
            merged["_provider"] = "openai_vision"
            merged["_status"] = "vision_active"
            merged["_status_label"] = "OpenAI Vision aktif"
            merged["_user_message"] = "Yorumlar şu an OpenAI Vision ile üretiliyor."
            merged["_vision_ok"] = True
            if payload.get("_model"):
                merged["_model"] = payload["_model"]
            return merged
        if isinstance(payload, dict) and payload.get("error"):
            fallback["_vision_error"] = _safe_provider_error_message(str(payload.get("error")))

    if qwen_vision_runtime_available():
        payload = qwen_vision_critique_cached(image_bytes, mode, editor_mode) or {}
        if isinstance(payload, dict) and payload and not payload.get("error"):
            merged = fallback.copy()
            for key in ["scene_summary", "concrete_details", "meaning_layers", "global_summary", "one_line_caption"]:
                value = payload.get(key)
                if value:
                    merged[key] = value
            if isinstance(payload.get("editor_comments"), dict) and payload["editor_comments"]:
                merged["editor_comments"] = _deoverlap_editor_comments(payload["editor_comments"], profile)
            merged["_source"] = "qwen"
            merged["_provider"] = "qwen_local"
            merged["_status"] = "qwen_active"
            merged["_status_label"] = "Qwen Vision Light Light aktif"
            merged["_user_message"] = "Yorumlar şu an daha hafif Qwen vision modeli üzerinden üretiliyor."
            merged["_vision_ok"] = True
            if payload.get("_model"):
                merged["_model"] = payload["_model"]
            return merged
        if isinstance(payload, dict) and payload.get("error"):
            fallback["_qwen_error"] = _safe_provider_error_message(str(payload.get("error")))

    # no providers active
    if openai_vision_available():
        fallback["_status"] = "vision_failed_fallback"
        fallback["_status_label"] = "ÇOFSAT Motoru ile devam ediliyor"
        fallback["_user_message"] = "Derin AI okuması şu an kullanılamıyor. Yorumlar ÇOFSAT motoru ile üretiliyor."
    elif qwen_vision_runtime_available():
        fallback["_status"] = "qwen_failed_fallback"
        fallback["_status_label"] = "ÇOFSAT Motoru ile devam ediliyor"
        fallback["_user_message"] = "Derin AI okuması şu an kullanılamıyor. Yorumlar ÇOFSAT motoru ile üretiliyor."
    else:
        fallback["_status"] = "no_provider"
        fallback["_status_label"] = "ÇOFSAT Motoru aktif"
        fallback["_user_message"] = "Yorumlar şu an ÇOFSAT Motoru ile üretiliyor."

    return fallback

def _normalize_list_for_report(value, fallback=None) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return fallback or []



def render_vision_debug_panel(report: Dict) -> None:
    model = str(report.get("_model") or "Heuristics + editor style engine")
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Sistem Durumu</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        render_compact_info_card("ÇOFSAT Motoru", "Hazır")
    with c2:
        render_compact_info_card("Durum", "Aktif", "Yerel ÇOFSAT motoru")
    with c3:
        render_compact_info_card("Model", model)

    st.success("ÇOFSAT motoru aktif. Editör yorumları ve özetler bu analizden üretiliyor.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_ai_reading_report(report: Dict) -> None:
    caption = str(report.get("one_line_caption") or "").strip()
    scene_summary = str(report.get("scene_summary") or "").strip()
    global_summary = str(report.get("global_summary") or "").strip()
    details = _normalize_list_for_report(report.get("concrete_details"))
    meanings = _normalize_list_for_report(report.get("meaning_layers"))
    source = report.get("_source", "fallback")
    model = report.get("_model", "")
    status_label = str(report.get("_status_label") or "").strip()
    user_message = str(report.get("_user_message") or "").strip()
    vision_error = str(report.get("_vision_error") or "").strip()
    model_badge = f" · {model}" if model else ""
    source_label = "ÇOFSAT Motoru okuması"

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>AI Okuma Raporu</div>", unsafe_allow_html=True)

    if source == "vision":
        st.success(f"{status_label}{model_badge} — {user_message}")
    else:
        msg = f"{status_label} — {user_message}".strip()
        if vision_error:
            msg = f"{msg} {vision_error}"
        st.info(msg)

    st.markdown(
        f"<div class='mini-note' style='margin-bottom:.8rem;'>{escape(source_label + model_badge)}</div>",
        unsafe_allow_html=True,
    )

    if caption:
        st.markdown(
            f"<div class='summary-card' style='margin-bottom:.85rem;'><div class='editor-title'>Tek cümlede fotoğraf</div><div class='mini-note' style='font-size:1.02rem; line-height:1.8;'>{escape(caption)}</div></div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns([1.1, 0.9], gap="large")
    with c1:
        if scene_summary:
            st.markdown("<div class='panel-title'>Genel okuma</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='mini-note' style='white-space: normal; line-height:1.8; font-size:1rem;'>{escape(scene_summary)}</div>", unsafe_allow_html=True)
        if global_summary:
            st.markdown("<div class='panel-title' style='margin-top:1rem;'>Kısa sonuç</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='mini-note' style='white-space: normal; line-height:1.8; font-size:1rem;'>{escape(global_summary)}</div>", unsafe_allow_html=True)
    with c2:
        if details:
            render_bullets("Somut ayrıntılar", details[:6], "🔎")
        if meanings:
            render_bullets("Anlam katmanları", meanings[:4], "🧠")

    st.markdown("</div>", unsafe_allow_html=True)


def render_editor_comments(result: CritiqueResult, selected_editor_name: str, ai_report: Optional[Dict] = None) -> None:
    ai_comments = {}
    if isinstance(ai_report, dict) and isinstance(ai_report.get("editor_comments"), dict):
        ai_comments = ai_report.get("editor_comments") or {}

    def comment_for(name: str) -> str:
        value = ai_comments.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return build_editor_comment(name, result)

    ordered_names = [selected_editor_name] + [n for n in EDITOR_NAMES if n != selected_editor_name]
    tabs = st.tabs(ordered_names)
    for tab, name in zip(tabs, ordered_names):
        with tab:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='mini-note' style='white-space: normal; line-height:1.8; font-size:1.02rem;'>{escape(comment_for(name))}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


def render_compact_info_card(title: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="summary-card compact-info-card">
            <div class="compact-info-title">{title}</div>
            <div class="compact-info-value">{value}</div>
            {f'<div class="compact-info-caption">{caption}</div>' if caption else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )




def render_vision_status_panel(ai_report: Dict) -> None:
    status = str(ai_report.get("_status") or "unknown")
    label = str(ai_report.get("_status_label") or "Bilinmiyor")
    message = str(ai_report.get("_user_message") or "")
    model = str(ai_report.get("_model") or "—")
    provider = str(ai_report.get("_provider") or "local")
    openai_error = str(ai_report.get("_vision_error") or "")
    qwen_error = str(ai_report.get("_qwen_error") or "")

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Sistem Durumu</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        render_compact_info_card("Derin AI", "Hazır" if openai_vision_available() else "Kapalı")
    with c2:
        render_compact_info_card("ÇOFSAT Motoru", "Hazır")
    with c3:
        render_compact_info_card("Çalışma modu", {"openai_vision":"OpenAI Vision","qwen_local":"Qwen Vision Light","local":"ÇOFSAT Motoru"}.get(provider, provider))
    with c4:
        render_compact_info_card("Model", model)

    if status in {"vision_active", "qwen_active"}:
        st.success(message or "AI provider aktif.")
    else:
        st.info(message or "Yorumlar şu an ÇOFSAT Motoru ile üretiliyor.")

    if openai_error:
        st.caption(vision_error)
    if qwen_error:
        st.error(f"Qwen Vision Light hatası: {qwen_error}")

    st.markdown("</div>", unsafe_allow_html=True)



def build_text_report(result: CritiqueResult, ai_report: Dict, dynamic_shooting_notes: List[str], dynamic_editing_notes: List[str], dynamic_strengths: List[str], dynamic_first_reading: str, dynamic_structural_reading: str, dynamic_editorial_result: str) -> str:
    editor_comments = ai_report.get("editor_comments", {}) if isinstance(ai_report, dict) else {}
    lines = [
        "ÇOFSAT Fotoğraf Ön Değerlendirme",
        "=" * 34,
        f"Skor: {result.total_score:.1f}/100",
        f"Seviye: {result.overall_level}",
        f"Tür önerisi: {result.suggested_mode}",
        "",
        "Ana güçlü taraf:",
        str(result.key_strength),
        "",
        "Ana sorun:",
        str(result.key_issue),
        "",
        "İlk okuma:",
        dynamic_first_reading,
        "",
        "Yapısal okuma:",
        dynamic_structural_reading,
        "",
        "Festival / seçki kararı:",
        dynamic_editorial_result,
        "",
        "Çekim notları:",
        *[f"- {x}" for x in dynamic_shooting_notes],
        "",
        "Düzenleme notları:",
        *[f"- {x}" for x in dynamic_editing_notes],
        "",
        "Editör yorumları:",
    ]
    for name in EDITOR_NAMES:
        comment = str(editor_comments.get(name) or "").strip()
        if comment:
            lines += [f"[{name}]", comment, ""]
    return "\n".join(lines).strip()
def main() -> None:
    inject_css()

    logo_file = find_logo_file()

    with st.sidebar:
        selected_mode = st.selectbox("Tür", list(MODE_PROFILES.keys()), index=0)
        selected_editor_mode = st.selectbox("Ton", list(EDITOR_MODES.keys()), index=0)
        selected_editor_name = st.radio("Editör", EDITOR_NAMES, index=0)
        sidebar_editor_placeholder = st.empty()

    render_sidebar(selected_mode, selected_editor_mode, selected_editor_name)

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
                    odak, akış, anlatı ve görsel dengeyi tek bakışta okunur hale getiren premium değerlendirme deneyimi.
                </p>
                <div class="hero-badges">
                    <span class="hero-badge">Aktif tür: {selected_mode}</span>
                    <span class="hero-badge">Ton: {selected_editor_mode}</span>
                                        <span class="ghost-badge">Isı Haritası + Göz Akışı + Altın Oran</span>
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

    st.caption("Yüklenen fotoğraflar analiz için otomatik optimize edilir; görünüm kalitesi korunurken analiz daha hızlı ve daha verimli çalışır.")

    if uploaded_file is None:
        st.markdown(
            """
            <div class="panel-card">
                <div class="section-title">Hazır olduğunda analiz burada görünecek</div>
                <div class="mini-note">
                    Sonuç ekranında genel skor, editör yorumları, Isı Haritası, göz akışı, odak katmanı, altın oran şemaları ve çekim/düzenleme notları yer alır.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    image_bytes = uploaded_file.getvalue()
    analysis_bytes = optimize_uploaded_bytes(image_bytes)
    analysis_key = _current_analysis_key(analysis_bytes, selected_mode, selected_editor_mode)
    if st.session_state.get("deep_ai_analysis_key") != analysis_key:
        st.session_state.setdefault("deep_ai_analysis_key", None)

    use_deep_ai = bool(openai_vision_available() or qwen_vision_runtime_available())
    st.markdown(
        f"<div class='mini-note' style='margin-top:.35rem;'>Yüklenen dosya: {human_file_size(len(image_bytes))} · Analiz sürümü: {human_file_size(len(analysis_bytes))}</div>",
        unsafe_allow_html=True,
    )


    loading_placeholder = st.empty()
    loading_placeholder.markdown(
        """
        <div style="display:flex; align-items:center; gap:.75rem; margin:.4rem 0 1rem 0;">
            <div style="width:14px;height:14px;border-radius:999px;background:#ff8c3a;box-shadow:0 0 0 rgba(255,140,58,.6);animation:pulseDot 1.2s infinite;"></div>
            <div style="padding:.75rem 1rem;border:1px solid rgba(255,255,255,.08);border-radius:999px;background:rgba(255,255,255,.03);font-weight:600;">Fotoğraf analiz ediliyor...</div>
        </div>
        <style>
        @keyframes pulseDot {0% {transform:scale(0.9); box-shadow:0 0 0 0 rgba(255,140,58,.55);} 70% {transform:scale(1); box-shadow:0 0 0 12px rgba(255,140,58,0);} 100% {transform:scale(0.9); box-shadow:0 0 0 0 rgba(255,140,58,0);} }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner(""):
        image = get_resized_rgb(image_bytes)
        result = critique_image(analysis_bytes, selected_mode, selected_editor_mode)

        attention = build_attention_map(image)
        main_points = top_regions(attention, n=3, window=max(35, min(image.size) // 12))
        distraction_points = distraction_regions(attention, main_points, n=2)

        overlay_img = draw_analysis_overlay(image, main_points, distraction_points)
        heatmap_img = build_heatmap_image(image, attention)

        phi_grid_img = draw_phi_grid(image)
        diagonal_img = draw_golden_diagonals(image)
        spiral_img = draw_golden_spiral(image, main_points)

        best_scheme, scheme_reason = describe_golden_ratio_fit(main_points, image.size[0], image.size[1])
        ai_report = build_ai_reading_report(analysis_bytes, selected_mode, selected_editor_mode, result, use_deep_ai=use_deep_ai)
        dynamic_shooting_notes, dynamic_editing_notes = derive_dynamic_action_notes(ai_report, result, selected_editor_mode)
        dynamic_strengths, dynamic_first_reading, dynamic_structural_reading, dynamic_editorial_result = derive_dynamic_summary_sections(ai_report, result)

    loading_placeholder.empty()

    render_editor_snapshot_in_sidebar(selected_editor_name, result, ai_report, sidebar_editor_placeholder)

    st.markdown("<div id='analysis-results'></div>", unsafe_allow_html=True)
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const scrollToResults = () => {
            const anchor = doc.getElementById('analysis-results');
            if (anchor) {
                anchor.scrollIntoView({behavior: 'smooth', block: 'start'});
            }
        };
        const flashSidebarComment = () => {
            const el = doc.getElementById('sidebar-editor-comment');
            if (!el) return;
            el.style.transition = 'box-shadow .25s ease, transform .25s ease';
            el.style.boxShadow = '0 0 0 3px rgba(255,255,255,0.28), 0 14px 28px rgba(114,58,13,0.18)';
            el.style.transform = 'translateY(-2px)';
            setTimeout(() => {
                el.style.boxShadow = '';
                el.style.transform = '';
            }, 900);
        };
        setTimeout(scrollToResults, 120);
        setTimeout(flashSidebarComment, 240);
        </script>
        """,
        height=0,
    )

    top_left, top_right = st.columns([1.08, 0.92], gap="large")

    with top_left:
        st.image(image, caption=f"Yüklenen fotoğraf · {uploaded_file.name}", use_container_width=True)

    with top_right:
        st.markdown("<div class='score-box'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Genel sonuç</div>", unsafe_allow_html=True)
        render_score_card("Skor", f"{result.total_score:.1f}/100", result.overall_level)
        row1_c1, row1_c2 = st.columns(2, gap="small")
        with row1_c1:
            render_compact_info_card("Seviye", result.overall_level, result.overall_tag)
        with row1_c2:
            render_compact_info_card("Tür önerisi", result.suggested_mode, result.suggested_mode_reason)
        render_compact_info_card("Aktif ton", selected_editor_mode)
        st.progress(result.total_score / 100.0)
        st.info((ai_report.get('global_summary') if isinstance(ai_report, dict) and ai_report.get('global_summary') else 'Fotoğrafın genel okuması hazırlanıyor.'))
        render_pill_row(result.tags)
        report_text = build_text_report(result, ai_report, dynamic_shooting_notes, dynamic_editing_notes, dynamic_strengths, dynamic_first_reading, dynamic_structural_reading, dynamic_editorial_result)
        st.download_button(
            "Metni indir",
            data=report_text.encode("utf-8"),
            file_name="cofsat_rapor.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_text_report",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Editörler bu kareye ne diyor?</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-note' style='margin-bottom:.7rem;'>Bu alan fotoğraf için üretilen editör yorumlarını gösterir.</div>", unsafe_allow_html=True)
    _report_editor_comments = ai_report.get("editor_comments", {}) if isinstance(ai_report, dict) else {}
    quick_tabs = st.tabs(EDITOR_NAMES)
    for tab, name in zip(quick_tabs, EDITOR_NAMES):
        with tab:
            _quick_comment = _report_editor_comments.get(name) if isinstance(_report_editor_comments, dict) else None
            if not isinstance(_quick_comment, str) or not _quick_comment.strip():
                _quick_comment = build_editor_comment(name, result)
            st.markdown(f"<div class='mini-note' style='white-space: normal; line-height:1.72; font-size:1rem;'>{escape(str(_quick_comment).strip())}</div>", unsafe_allow_html=True)
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
        ["Görsel Katmanlar", "Editör Okuması", "Puanlama", "Altın Oran", "Teknik"]
    )

    with tab1:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div class='panel-title'>Odak merkezi · İkincil yük · Göz rotası</div>", unsafe_allow_html=True)
            st.image(overlay_img, use_container_width=True)
            st.caption(build_overlay_caption(SceneProfile(**result.metrics.get('scene_profile', {})) if isinstance(result.metrics, dict) and result.metrics.get('scene_profile') else None))
            make_download_button(overlay_img, "Katmanı indir", "cofsat_overlay.png", "dl_overlay")
        with c2:
            st.markdown("<div class='panel-title'>Isı Haritası</div>", unsafe_allow_html=True)
            st.image(heatmap_img, use_container_width=True)
            st.caption(build_heatmap_caption(SceneProfile(**result.metrics.get('scene_profile', {})) if isinstance(result.metrics, dict) and result.metrics.get('scene_profile') else None))
            make_download_button(heatmap_img, "Isı Haritasını indir", "cofsat_isi_haritasi.png", "dl_heatmap")

    with tab2:
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            render_bullets("Güçlü yönler", dynamic_strengths, "✅")
            render_bullets("Gelişim alanları", result.development_areas, "⚠️")
        with c2:
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>İlk okuma</div>", unsafe_allow_html=True)
            st.write(dynamic_first_reading)
            st.markdown("<div class='section-title'>Yapısal okuma</div>", unsafe_allow_html=True)
            st.write(dynamic_structural_reading)
            st.markdown("<div class='section-title'>Festival / seçki kararı</div>", unsafe_allow_html=True)
            st.write(dynamic_editorial_result)
            st.markdown("</div>", unsafe_allow_html=True)

        render_editor_comments(result, selected_editor_name, ai_report)

        c3, c4 = st.columns(2, gap="large")
        with c3:
            render_bullets("Çekim notları", dynamic_shooting_notes, "📷")
        with c4:
            render_bullets("Düzenleme notları", dynamic_editing_notes, "🎛️")

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
    st.markdown("<div class='section-title' style='margin-top:1.25rem;'>Hızlı uygulama özeti</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="footer-note">
            ÇOFSAT ön değerlendirme sistemi, hızlı karar vermek için görsel okumayı destekler.
            Son yargı yine fotoğrafçının niyeti, bağlamı ve seçki içindeki yeriyle verilir.
        </div>
        """,
        unsafe_allow_html=True,
    )


@dataclass
class SceneProfile:
    scene_type: str
    subject_hint: str
    visual_mood: str
    light_character: str
    complexity_level: str
    balance_character: str
    primary_region: str
    secondary_region: str
    distraction_region: str
    face_count: int
    human_presence_score: float
    main_subject_confidence: float
    concrete_detail_1: str
    concrete_detail_2: str
    concrete_detail_3: str
    detail_signature: str
    subject_position: str
    light_type_detail: str
    environment_type: str
    human_action: str
    dominant_shape: str
    visual_tension_level: str
    historical_texture_hint: str


def detect_faces(gray: np.ndarray) -> int:
    """Muhafazakâr yüz/insan sayımı.

    Eski sürüm bazı yüksek kontrastlı yüzeyleri ya da küçük sahte adayları fazla sayabiliyordu.
    Burada yalnızca makul alan/oran aralığında ve daha güçlü komşulukla bulunan yüzleri sayıyoruz.
    Büyük belirsizlikte sayıyı abartmak yerine düşük tutmak tercih edilir.
    """
    if cv2 is None:
        return 0
    cascade_path = None
    if hasattr(cv2, "data"):
        candidate = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if candidate.exists():
            cascade_path = str(candidate)
    if not cascade_path:
        return 0

    h, w = gray.shape[:2]
    image_area = float(max(1, h * w))
    detector = cv2.CascadeClassifier(cascade_path)
    try:
        raw_faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=7,
            minSize=(34, 34),
        )
    except Exception:
        return 0

    filtered = []
    for (x, y, fw, fh) in raw_faces:
        area_ratio = (fw * fh) / image_area
        aspect = fw / float(max(1, fh))
        if area_ratio < 0.0012 or area_ratio > 0.14:
            continue
        if aspect < 0.55 or aspect > 1.55:
            continue
        if y + fh >= h * 0.98 or x + fw >= w * 0.995:
            continue
        filtered.append((x, y, fw, fh))

    if not filtered:
        return 0

    strong = []
    for face in filtered:
        x, y, fw, fh = face
        area_ratio = (fw * fh) / image_area
        if area_ratio >= 0.004:
            strong.append(face)

    if strong:
        return int(min(3, len(strong)))
    return 1 if len(filtered) == 1 else int(min(2, len(filtered)))


def region_name(x: float, y: float) -> str:
    horiz = "sol" if x < 1/3 else "sağ" if x > 2/3 else "orta"
    vert = "üst" if y < 1/3 else "alt" if y > 2/3 else "orta"
    if horiz == "orta" and vert == "orta":
        return "merkez"
    if horiz == "orta":
        return f"{vert} merkez"
    if vert == "orta":
        return f"{horiz} merkez"
    return f"{vert} {horiz}"


def classify_visual_mood(metrics: ImageMetrics) -> str:
    if metrics.brightness_mean < 85 and metrics.dynamic_tension_score > 0.55:
        return "gerilimli ve karanlık"
    if metrics.brightness_mean < 110:
        return "sessiz ve içe dönük"
    if metrics.edge_density > 0.16 and metrics.visual_noise_score > 0.35:
        return "hareketli ve kalabalık"
    if metrics.negative_space_score > 0.55:
        return "sade ve nefesli"
    return "dengeli ve gözlemci"


def classify_light_character(metrics: ImageMetrics) -> str:
    sep_h = abs(metrics.left_brightness - metrics.right_brightness)
    sep_v = abs(metrics.top_brightness - metrics.bottom_brightness)
    if max(sep_h, sep_v) < 18:
        return "yaygın ve yumuşak"
    if sep_v > sep_h:
        return "üstten-alttan ayrışan"
    return "yan yönlü"


def classify_scene_type(metrics: ImageMetrics, face_count: int, human_presence_score: float) -> str:
    if face_count >= 1:
        return "Portre"
    if human_presence_score > 0.58 and metrics.dynamic_tension_score > 0.45:
        return "Sokak"
    if metrics.negative_space_score > 0.58 and metrics.symmetry_score > 0.55:
        return "Soyut"
    if human_presence_score > 0.42:
        return "Belgesel"
    if metrics.edge_density > 0.16:
        return "Sokak"
    return "Soyut"



def _color_name_from_rgb(rgb: Tuple[float, float, float]) -> str:
    r, g, b = [float(x) / 255.0 for x in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn
    if mx < 0.18:
        return "koyu ton"
    if diff < 0.08:
        if mx > 0.75:
            return "açık ton"
        if mx > 0.45:
            return "nötr ton"
        return "gri ton"
    if mx == r:
        hue = (g - b) / (diff + 1e-9)
    elif mx == g:
        hue = 2 + (b - r) / (diff + 1e-9)
    else:
        hue = 4 + (r - g) / (diff + 1e-9)
    hue = (hue * 60) % 360
    sat = 0 if mx == 0 else diff / mx
    if sat < 0.18:
        return "nötr ton"
    if hue < 15 or hue >= 345:
        return "kırmızı ton"
    if hue < 40:
        return "turuncu ton"
    if hue < 70:
        return "sarı ton"
    if hue < 165:
        return "yeşil ton"
    if hue < 255:
        return "mavi ton"
    if hue < 315:
        return "mor ton"
    return "pembe ton"


def _dominant_orientation(gray: np.ndarray) -> str:
    gy, gx = np.gradient(gray.astype(np.float32))
    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180) % 180
    mask = mag > np.percentile(mag, 78)
    if mask.mean() < 0.01:
        return "yumuşak akış"
    vals = ang[mask]
    vertical = np.mean(((vals < 20) | (vals > 160)))
    horizontal = np.mean((vals > 70) & (vals < 110))
    diagonal = 1 - max(vertical, horizontal)
    if vertical > max(horizontal, diagonal):
        return "dikey akış"
    if horizontal > max(vertical, diagonal):
        return "yatay akış"
    return "çapraz akış"


def _quadrant_means(arr: np.ndarray) -> Dict[str, float]:
    h, w = arr.shape[:2]
    return {
        "üst sol": float(arr[:h//2, :w//2].mean()),
        "üst sağ": float(arr[:h//2, w//2:].mean()),
        "alt sol": float(arr[h//2:, :w//2].mean()),
        "alt sağ": float(arr[h//2:, w//2:].mean()),
    }


def _build_concrete_details(img: Image.Image, gray: np.ndarray, metrics: ImageMetrics, face_count: int, primary_region: str, secondary_region: str, distraction_region: str) -> Tuple[str, str, str, str]:
    arr = np.array(img).astype(np.float32)
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    bright = arr.mean(axis=2) / 255.0

    sat_mask = sat > np.percentile(sat, 82)
    if sat_mask.mean() < 0.01:
        sat_mask = sat > np.percentile(sat, 70)
    color_rgb = arr[sat_mask].mean(axis=0) if sat_mask.mean() > 0 else arr.reshape(-1, 3).mean(axis=0)
    color_name = _color_name_from_rgb(tuple(color_rgb))

    q_brightness = _quadrant_means(gray.astype(np.float32))
    brightest_quad = max(q_brightness, key=q_brightness.get)
    darkest_quad = min(q_brightness, key=q_brightness.get)

    color_strength = sat * (0.55 + 0.45 * bright)
    q_color = _quadrant_means(color_strength)
    color_quad = max(q_color, key=q_color.get)

    orientation = _dominant_orientation(gray)

    h, w = gray.shape
    px_map = {"merkez": 0.5, "sol merkez": 0.28, "sağ merkez": 0.72, "üst merkez": 0.5, "alt merkez": 0.5, "üst sol": 0.25, "üst sağ": 0.75, "alt sol": 0.25, "alt sağ": 0.75}
    py_map = {"merkez": 0.5, "sol merkez": 0.5, "sağ merkez": 0.5, "üst merkez": 0.25, "alt merkez": 0.75, "üst sol": 0.25, "üst sağ": 0.25, "alt sol": 0.75, "alt sağ": 0.75}
    cx = int(px_map.get(primary_region, 0.5) * w)
    cy = int(py_map.get(primary_region, 0.5) * h)
    rw, rh = max(12, w // 8), max(12, h // 8)
    x1, x2 = max(0, cx - rw), min(w, cx + rw)
    y1, y2 = max(0, cy - rh), min(h, cy + rh)
    patch = gray[y1:y2, x1:x2].astype(np.float32)
    patch_mean = float(patch.mean()) if patch.size else float(gray.mean())
    global_mean = float(gray.mean())
    contrast_phrase = "çevresinden daha parlak" if patch_mean > global_mean + 14 else "çevresinden daha koyu" if patch_mean < global_mean - 14 else "çevresiyle dengeli"

    details = []
    if face_count >= 1:
        details.append(f"{primary_region} bölgesinde insan yüzü hemen ağırlık kuruyor")
    else:
        details.append(f"{primary_region} bölgesindeki ana vurgu {contrast_phrase} bir kütle gibi öne çıkıyor")

    details.append(f"{color_quad} tarafında toplanan {color_name} fotoğrafa ayrı bir dikkat noktası veriyor")

    if metrics.negative_space_score > 0.56:
        details.append(f"{secondary_region} tarafında açılan boşluk ana etkiyi daha yalnız ve daha görünür bırakıyor")
    elif metrics.visual_noise_score > 0.40 or metrics.edge_density > 0.16:
        details.append(f"{distraction_region} tarafındaki yoğun doku gözün sahne içinde daha uzun oyalanmasına neden oluyor")
    else:
        details.append(f"{secondary_region} hattındaki ikinci katman ana odağı destekleyen sakin bir eşlik kuruyor")

    details.append(f"en parlak açıklık {brightest_quad} bölümünde, en koyu ağırlık ise {darkest_quad} bölümünde toplanıyor")
    details.append(f"karede {orientation} hissi bakışı tek noktada tutmak yerine yönlendiriyor")

    uniq = []
    for d in details:
        if d not in uniq:
            uniq.append(d)
    while len(uniq) < 3:
        uniq.append("fotoğrafın içinde küçük ama belirgin bir ilişki hissediliyor")
    signature = " | ".join(uniq[:4])
    return uniq[0], uniq[1], uniq[2], signature



def infer_subject_position(primary_region: str) -> str:
    mapping = {
        "merkez": "merkezde",
        "sol merkez": "sola yaslı",
        "sağ merkez": "sağa yaslı",
        "üst merkez": "üst hatta yakın",
        "alt merkez": "alt hatta yakın",
        "üst sol": "üst sol bölgede",
        "üst sağ": "üst sağ bölgede",
        "alt sol": "alt sol bölgede",
        "alt sağ": "alt sağ bölgede",
    }
    return mapping.get(primary_region, primary_region)


def infer_light_type_detail(metrics: ImageMetrics) -> str:
    sep_h = abs(metrics.left_brightness - metrics.right_brightness)
    sep_v = abs(metrics.top_brightness - metrics.bottom_brightness)
    mean_b = metrics.brightness_mean
    if mean_b < 85 and max(sep_h, sep_v) > 20:
        return "kontrastlı ve yönlü"
    if max(sep_h, sep_v) < 16:
        return "dağınık ve yumuşak"
    if sep_h > sep_v and metrics.left_brightness > metrics.right_brightness:
        return "soldan gelen yönlü"
    if sep_h > sep_v:
        return "sağdan gelen yönlü"
    if metrics.top_brightness > metrics.bottom_brightness:
        return "üstten süzülen"
    return "alttan yükselen"


def infer_environment_type(img: Image.Image, metrics: ImageMetrics, face_count: int) -> str:
    arr = np.array(img).astype(np.float32)
    sat = ((arr.max(axis=2) - arr.min(axis=2)) / (arr.max(axis=2) + 1e-6)).mean()
    gray = np.array(img.convert("L"))
    edges = estimate_edge_density(gray)
    if sat < 0.12 and edges > 0.16:
        return "tarihi doku hissi veren sert yüzeyli mekân"
    if face_count >= 1 and edges > 0.14:
        return "kentsel ya da mimari çevre"
    if metrics.negative_space_score > 0.58:
        return "sadeleştirilmiş ve soyutlanan alan"
    if sat > 0.22 and edges < 0.12:
        return "yumuşak geçişli iç mekân ya da kontrollü sahne"
    return "gündelik çevre"


def infer_human_action(profile_scene: str, face_count: int, metrics: ImageMetrics) -> str:
    if face_count >= 1:
        return "bakış ya da ifade üzerinden duran bir insan varlığı"
    if profile_scene in {"Sokak", "Belgesel"} and metrics.dynamic_tension_score > 0.58:
        return "geçiş, bekleme ya da anlık karşılaşma hissi"
    if profile_scene == "Soyut":
        return "insan eyleminden çok biçimsel bir akış"
    return "belirgin ama sakin bir görsel hareket"


def infer_visual_tension_level(metrics: ImageMetrics) -> str:
    if metrics.dynamic_tension_score > 0.68:
        return "yüksek"
    if metrics.dynamic_tension_score > 0.46:
        return "orta"
    return "düşük"


def infer_historical_texture_hint(img: Image.Image, metrics: ImageMetrics) -> str:
    arr = np.array(img.convert("L")).astype(np.float32)
    local_std = float(ImageStat.Stat(Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.FIND_EDGES)).mean[0])
    if metrics.edge_density > 0.16 and metrics.brightness_std > 48 and local_std > 18:
        return "yüzeylerde yaşanmışlık ve tarih hissi okunuyor"
    if metrics.edge_density > 0.13:
        return "mekânın dokusu yalnızca arka plan değil, anlatının aktif parçası"
    return "mekân dokusu geri planda ama tamamen silinmiş değil"


def build_scene_description_payload(metrics: ImageMetrics, profile: SceneProfile) -> Dict:
    return {
        "subject_position": profile.subject_position,
        "light_type": profile.light_type_detail,
        "environment_type": profile.environment_type,
        "human_action": profile.human_action,
        "dominant_shape": profile.dominant_shape,
        "visual_tension": profile.visual_tension_level,
        "historical_texture_hint": profile.historical_texture_hint,
    }

@st.cache_data(show_spinner=False)
def extract_scene_profile_cached(image_bytes: bytes) -> Dict:
    img = get_resized_rgb(image_bytes)
    gray = pil_to_gray_np(img)
    metrics = ImageMetrics(**extract_metrics_cached(image_bytes))
    attention = build_attention_map(img)
    points = top_regions(attention, n=3, window=max(35, min(img.size) // 12))
    distractions = distraction_regions(attention, points, n=2)
    main_x = points[0][0] / max(1, img.size[0]) if points else 0.5
    main_y = points[0][1] / max(1, img.size[1]) if points else 0.5
    second_x = points[1][0] / max(1, img.size[0]) if len(points) > 1 else main_x
    second_y = points[1][1] / max(1, img.size[1]) if len(points) > 1 else main_y
    dist_x = distractions[0][0] / max(1, img.size[0]) if distractions else main_x
    dist_y = distractions[0][1] / max(1, img.size[1]) if distractions else main_y

    face_count = detect_faces(gray)
    human_presence_score = float(clamp01(0.55 * (face_count > 0) + 0.25 * metrics.dynamic_tension_score + 0.20 * (1 - abs(metrics.center_of_mass_y - 0.58) / 0.58)))
    scene_type = classify_scene_type(metrics, face_count, human_presence_score)

    if face_count >= 1:
        subject_hint = "yüz ve ifade"
    elif metrics.dynamic_tension_score > 0.62:
        subject_hint = "ana figür ya da hareket merkezi"
    elif metrics.negative_space_score > 0.55:
        subject_hint = "biçim ve boşluk ilişkisi"
    else:
        subject_hint = "ana görsel ağırlık merkezi"

    if metrics.visual_noise_score > 0.42 or metrics.edge_density > 0.17:
        complexity = "yoğun"
    elif metrics.negative_space_score > 0.55:
        complexity = "sade"
    else:
        complexity = "orta yoğunlukta"

    if metrics.symmetry_score > 0.72:
        balance = "dengeli"
    elif metrics.dynamic_tension_score > 0.58:
        balance = "gerilimli"
    else:
        balance = "yarı dengeli"
    detail_1, detail_2, detail_3, detail_signature = _build_concrete_details(
        img, gray, metrics, face_count, region_name(main_x, main_y), region_name(second_x, second_y), region_name(dist_x, dist_y)
    )

    return asdict(SceneProfile(
        scene_type=scene_type,
        subject_hint=subject_hint,
        visual_mood=classify_visual_mood(metrics),
        light_character=classify_light_character(metrics),
        complexity_level=complexity,
        balance_character=balance,
        primary_region=region_name(main_x, main_y),
        secondary_region=region_name(second_x, second_y),
        distraction_region=region_name(dist_x, dist_y),
        face_count=face_count,
        human_presence_score=human_presence_score,
        main_subject_confidence=float(attention.max()) if attention.size else 0.5,
        concrete_detail_1=detail_1,
        concrete_detail_2=detail_2,
        concrete_detail_3=detail_3,
        detail_signature=detail_signature,
        subject_position=infer_subject_position(region_name(main_x, main_y)),
        light_type_detail=infer_light_type_detail(metrics),
        environment_type=infer_environment_type(img, metrics, face_count),
        human_action=infer_human_action(scene_type, face_count, metrics),
        dominant_shape=_dominant_orientation(gray),
        visual_tension_level=infer_visual_tension_level(metrics),
        historical_texture_hint=infer_historical_texture_hint(img, metrics),
    ))


def choose_variant(options: List[str], seed_value: float) -> str:
    if not options:
        return ""
    idx = int(abs(seed_value) * 1000) % len(options)
    return options[idx]


def context_seed(metrics: ImageMetrics, profile: SceneProfile, extra: float = 0.0) -> float:
    return metrics.brightness_mean * 0.013 + metrics.edge_density * 5.3 + metrics.dynamic_tension_score * 3.1 + profile.human_presence_score * 2.7 + extra


def build_story_block(metrics: ImageMetrics, profile: SceneProfile, scores_100: Dict[str, float], editor_mode: str) -> Tuple[str, str, str]:
    seed = context_seed(metrics, profile)
    opening_options = [
        f"Bu kare ilk bakışta {profile.subject_hint} etrafında toplanıyor ve okuma {profile.primary_region} bölgesinde başlıyor.",
        f"Fotoğrafın merkezi etkisi {profile.primary_region} tarafta yoğunlaşıyor; göz önce {profile.subject_hint} tarafına tutunuyor.",
        f"Kadrajın asıl çağrısı {profile.primary_region} bölgesinden geliyor ve fotoğraf kendini önce {profile.subject_hint} üzerinden açıyor.",
    ]
    if scores_100["ilk_etki"] < 60:
        opening_options.append(f"Fotoğraf hemen açılmıyor; ama {profile.primary_region} çevresindeki {profile.subject_hint}, dikkat toplamak için en güçlü aday olarak öne çıkıyor.")

    structural_options = [
        f"Yapısal olarak kare {profile.complexity_level} bir yüzeye sahip; görsel denge ise daha çok {profile.balance_character} bir karakterde çalışıyor.",
        f"Kadrajın iç örgüsü {profile.complexity_level} görünüyor; bu yüzden göz akışını belirleyen şey kompozisyondan çok {profile.subject_hint} oluyor.",
        f"Karedeki ağırlık dağılımı {profile.balance_character} bir his veriyor; bu da fotoğrafın ritmini doğrudan etkiliyor.",
    ]
    if scores_100["dikkat_dagitici_unsurlar"] < 60:
        structural_options.append(f"Okumayı en çok zorlayan yer {profile.distraction_region} bölgesi; oradaki enerji ana odağın önüne geçmeye başlıyor.")

    editorial_options = [
        f"Editöryel olarak bu kare, {profile.visual_mood} atmosferini taşıdığı ölçüde çalışıyor; asıl mesele bu atmosferi daha berrak hale getirmek.",
        f"Fotoğrafın editöryel değeri, kurduğu {profile.visual_mood} hava ile {profile.subject_hint} arasındaki ilişkiye bağlı.",
        f"Bu kare seçki içinde yer açacaksa, bunu büyük ihtimalle {profile.visual_mood} tonu ve {profile.light_character} ışık karakteri sayesinde yapacak.",
    ]

    return (
        tone_text(choose_variant(opening_options, seed), editor_mode),
        tone_text(choose_variant(structural_options, seed + 0.37), editor_mode),
        tone_text(choose_variant(editorial_options, seed + 0.73), editor_mode),
    )


def pick_strengths(scores_100: Dict[str, float], metrics: ImageMetrics, profile: SceneProfile, editor_mode: str) -> List[str]:
    ordered = sorted(scores_100.items(), key=lambda x: x[1], reverse=True)
    dynamic = {
        "ilk_etki": [
            f"Fotoğraf ilk bakışta {profile.primary_region} çevresinde bir tutunma noktası kurabiliyor.",
            f"İlk temas gücü zayıf değil; göz özellikle {profile.subject_hint} üzerinden içeri giriyor.",
        ],
        "teknik_butunluk": [
            f"Ton ve netlik tamamen rastlantısal durmuyor; {profile.visual_mood} hava teknik yapı tarafından taşınıyor.",
            f"Teknik yapı fotoğrafı sırtlıyor; özellikle {profile.light_character} ışık dağılımı anlatıyı destekliyor.",
        ],
        "kompozisyon": [
            f"Kadrajın iskeleti dağılmıyor; ağırlık merkezi {profile.primary_region} çevresinde okunabilir kalıyor.",
            f"Kompozisyon kendi içinde tutarlı; {profile.primary_region} ile {profile.secondary_region} arasında bir akış kuruluyor.",
        ],
        "odak_ve_hiyerarsi": [
            f"Gözün tutunacağı yer büyük ölçüde belli; {profile.subject_hint} fotoğrafın ana ağırlığını taşıyor.",
            f"Hiyerarşi çalışıyor; bakış önce {profile.primary_region} bölgesine yerleşiyor sonra kadraja yayılıyor.",
        ],
        "anlati_gucu": [
            f"Kare yalnızca göstermiyor; {profile.visual_mood} bir duygu alanı açıyor.",
            f"Anlatı hissi kurulmuş; fotoğraf {profile.subject_hint} üzerinden bir ilişki başlatıyor.",
        ],
        "gorsel_dil": [
            f"Biçimsel kararlar ortak bir dile dönmeye başlamış; özellikle {profile.complexity_level} yapı bunu destekliyor.",
            f"Görsel dil, boşluk ve kütle ilişkisi üzerinden kendini hissettiriyor.",
        ],
        "sadelik": [
            f"Kare gereksiz yükü büyük ölçüde geri çekmiş; bu da {profile.subject_hint} için temiz bir alan açıyor.",
            f"Sadelik duygusu iyi; ana vurgu çevresinde nefes alan bir boşluk var.",
        ],
        "niyet_tutarliligi": [
            f"Fotoğraf tesadüfi görünmüyor; {profile.subject_hint} üzerine kurulmuş belirgin bir niyet hissi var.",
            f"Niyet okunabiliyor; kare neyi öne almak istediğini gizlemiyor.",
        ],
        "isik_yonu": [
            f"Işık sahnede yalnızca aydınlatmıyor; {profile.light_character} yapısıyla okumayı yönlendiriyor.",
            f"Işık yönü kontrollü; özellikle ana odağın çevresinde toparlayıcı bir etkisi var.",
        ],
        "derinlik_hissi": [
            f"Kare düz kalmıyor; {profile.primary_region} ile {profile.secondary_region} arasında derinlik hissi kuruluyor.",
            f"Katman etkisi var; fotoğraf izleyiciyi yüzeye değil içeriye davet ediyor.",
        ],
        "dikkat_dagitici_unsurlar": [
            f"Dikkat büyük ölçüde kontrolde; yan alanlar ana odağın önüne kolay geçmiyor.",
            f"Göz dağılmadan ilerleyebiliyor; özellikle {profile.distraction_region} tarafı şimdilik baskınlaşmıyor.",
        ],
        "zamanlama": [
            f"An seçimi karenin etkisini destekliyor; görüntü donuk değil, canlı hissediliyor.",
            f"Zamanlama burada yalnızca kayıt değil; fotoğrafın enerjisini taşıyan şeylerden biri.",
        ],
        "negatif_alan": [
            f"Boşluk kullanımı işe yarıyor; kadraj nefes almak için yeterli alan bırakıyor.",
            f"Negatif alan yalnızca boşluk değil, ana vurguya hizmet eden aktif bir yapı gibi çalışıyor.",
        ],
        "duygusal_yogunluk": [
            f"Fotoğrafın duygusal tonu yapay durmadan geçiyor; {profile.visual_mood} hava hissedilebiliyor.",
            f"Duygusal yoğunluk doğrudan bağırmıyor ama içeriden çalışıyor.",
        ],
        "editoryal_deger": [
            f"Bu kare seçki içinde yer açabilecek bir ağırlık taşıyor; yalnızca hoş görünmekle kalmıyor.",
            f"Editöryel olarak kullanılabilir bir çekirdeği var; fotoğrafın bakışı boş değil.",
        ],
        "tekrar_bakma_istegi": [
            f"Kare tek bakışta tükenmiyor; {profile.secondary_region} tarafında ikinci okuma için malzeme bırakıyor.",
            f"İlk bakıştan sonra da çalışıyor; fotoğrafa dönmek için küçük ama gerçek bir sebep var.",
        ],
    }
    out = []
    for i, (k, _) in enumerate(ordered[:4]):
        out.append(tone_text(choose_variant(dynamic[k], context_seed(metrics, profile, i)), editor_mode))
    return out


def pick_development_areas(scores_100: Dict[str, float], metrics: ImageMetrics, profile: SceneProfile, editor_mode: str) -> List[str]:
    ordered = sorted(scores_100.items(), key=lambda x: x[1])
    dynamic = {
        "ilk_etki": [
            f"İlk bakış etkisi biraz daha net kurulursa fotoğraf izleyiciyi daha güçlü durdurabilir.",
            f"Giriş etkisi tam açılmıyor; {profile.subject_hint} çevresindeki vurgu biraz daha belirginleşebilir.",
        ],
        "teknik_butunluk": [
            f"Ton ya da netlikteki küçük dağınıklıklar, kurulan {profile.visual_mood} havayı biraz zayıflatıyor.",
            f"Teknik yapı daha rafine olursa fotoğrafın niyeti daha temiz görünür.",
        ],
        "kompozisyon": [
            f"Kadrajın grafik yapısı biraz daha sıkı kurulursa {profile.primary_region} çevresindeki vurgu daha güçlü okunur.",
            f"Kompozisyon iyi bir çekirdek taşıyor; ama ağırlık merkezleri biraz daha disiplin isteyebilir.",
        ],
        "odak_ve_hiyerarsi": [
            f"Ana görsel ağırlık tam net değil; göz bazen {profile.distraction_region} tarafına gereğinden erken kayabiliyor.",
            f"Hiyerarşi biraz daha temizlenirse izleyici önce nereye bakacağını daha kararlı bilir.",
        ],
        "anlati_gucu": [
            f"Fotoğrafın his tarafı var; ama asıl anlatı biraz daha belirginleşirse kare daha kalıcı olur.",
            f"Anlatı çekirdeği hissediliyor; fakat neyin esas mesele olduğu daha görünür olabilir.",
        ],
        "gorsel_dil": [
            f"Biçimsel tercihler ortak bir dil kurmaya başlamış; ama hâlâ birkaç kararın aynı yöne çekilmesi gerekiyor.",
            f"Görsel dil parçalı duruyor; ton, ritim ve boşluk ilişkisi biraz daha birleşebilir.",
        ],
        "sadelik": [
            f"Bazı yan alanlar fotoğrafı gereğinden fazla yoruyor; kadraj biraz daha sadeleşirse ana vurgu rahatlar.",
            f"Gereksiz görsel yük azaltılırsa {profile.subject_hint} daha rahat nefes alır.",
        ],
        "niyet_tutarliligi": [
            f"Fotoğrafın neden var olduğu hissediliyor ama henüz tam berrak değil; niyet biraz daha görünür hale gelebilir.",
            f"Kare iyi bir yöne bakıyor; fakat hangi kararın neden alındığı daha belirgin okunabilir.",
        ],
        "isik_yonu": [
            f"Işığın ana odağı taşıma biçimi biraz daha bilinçli kurulursa okuma çok daha kararlı olur.",
            f"{profile.light_character.capitalize()} ışık karakteri var; ama bu etki ana vurguda daha net toplanabilir.",
        ],
        "derinlik_hissi": [
            f"Ön, orta ve arka plan ilişkisi biraz daha belirginleşirse kare daha hacimli hissedilir.",
            f"Derinlik duygusu güçlenirse fotoğraf iki boyutlu yüzey hissinden kurtulur.",
        ],
        "dikkat_dagitici_unsurlar": [
            f"Özellikle {profile.distraction_region} bölgesi ana odağın önüne geçmeye başlıyor; görsel ağırlığı biraz geri itmek iyi olur.",
            f"Bazı yan alanlar gereğinden fazla çağırıyor; bunlar bastırılırsa fotoğraf daha temiz konuşur.",
        ],
        "zamanlama": [
            f"Bir an önce ya da sonra çekilse etkinin değişebileceği hissi var; zamanlama biraz daha sıkı olabilir.",
            f"An seçimi fena değil; ama fotoğrafın enerjisi daha isabetli bir anda büyüyebilirdi.",
        ],
        "negatif_alan": [
            f"Boşluk kullanımı biraz daha kararında olursa kadraj daha rafine görünür.",
            f"Negatif alan şu an işlev görüyor ama daha bilinçli kurgulanırsa anlatıyı daha çok taşır.",
        ],
        "duygusal_yogunluk": [
            f"Duygusal etki var ama izleyiciye daha doğrudan geçmesi için vurgu biraz daha yoğunlaşabilir.",
            f"Fotoğrafın ruhu hissediliyor; yine de duygu odağı biraz daha açık seçilebilir.",
        ],
        "editoryal_deger": [
            f"Karede iyi bir fikir var; fakat seçki içinde daha güçlü yer açması için kararların biraz daha keskinleşmesi gerekiyor.",
            f"Editöryel potansiyel mevcut; ama fotoğrafın cümlesi biraz daha netleşirse ağırlığı artar.",
        ],
        "tekrar_bakma_istegi": [
            f"İlk bakış çalışıyor; ikinci bakış için biraz daha katman ya da gecikmeli ödül iyi gelebilir.",
            f"Fotoğraf kendini hemen veriyor; tekrar dönme isteği için ikinci bir vurgu alanı güçlenebilir.",
        ],
    }
    out = []
    for i, (k, _) in enumerate(ordered[:4]):
        out.append(tone_text(choose_variant(dynamic[k], context_seed(metrics, profile, 1.2 + i)), editor_mode))
    return out


def build_key_strength(scores_100: Dict[str, float], metrics: ImageMetrics, profile: SceneProfile, editor_mode: str) -> str:
    key = max(scores_100.items(), key=lambda x: x[1])[0]
    strengths = pick_strengths(scores_100, metrics, profile, editor_mode)
    lead = {
        "kompozisyon": "Bu karenin en güçlü tarafı kompozisyon iskeleti.",
        "anlati_gucu": "Bu karenin en güçlü tarafı anlatı hissi.",
        "odak_ve_hiyerarsi": "Bu karenin en güçlü tarafı odak yapısı.",
    }.get(key, "Bu karenin en güçlü tarafı kurduğu temel vurgu.")
    return tone_text(f"{lead} {strengths[0]}", editor_mode)


def build_key_issue(scores_100: Dict[str, float], metrics: ImageMetrics, profile: SceneProfile, editor_mode: str) -> str:
    key = min(scores_100.items(), key=lambda x: x[1])[0]
    issues = pick_development_areas(scores_100, metrics, profile, editor_mode)
    lead = {
        "dikkat_dagitici_unsurlar": "Bu karede en belirgin sorun dikkat kontrolü.",
        "odak_ve_hiyerarsi": "Bu karede en belirgin sorun hiyerarşi.",
        "anlati_gucu": "Bu karede en belirgin sorun anlatının tam açılmaması.",
    }.get(key, "Bu karede en belirgin sorun tek bir kararın henüz tam yerine oturmaması.")
    return tone_text(f"{lead} {issues[0]}", editor_mode)


def build_one_move_improvement(scores_100: Dict[str, float], metrics: ImageMetrics, profile: SceneProfile, editor_mode: str) -> str:
    key = min(scores_100.items(), key=lambda x: x[1])[0]
    mapping = {
        "ilk_etki": f"Tek hamlede en büyük iyileşme, {profile.primary_region} çevresindeki ana vurguyu daha net açmak olur.",
        "teknik_butunluk": "Tek hamlede en büyük iyileşme, ton geçişlerini ve netlik hissini daha rafine temizlemek olur.",
        "kompozisyon": f"Tek hamlede en büyük iyileşme, {profile.primary_region} ile {profile.secondary_region} arasındaki dengeyi sıkılaştırmak olur.",
        "odak_ve_hiyerarsi": f"Tek hamlede en büyük iyileşme, {profile.subject_hint} etrafındaki görsel ağırlığı netleştirmek olur.",
        "anlati_gucu": "Tek hamlede en büyük iyileşme, fotoğrafın asıl hissetmek istediği şeyi daha görünür kılmak olur.",
        "gorsel_dil": "Tek hamlede en büyük iyileşme, biçimsel kararları tek bir dil altında toplamak olur.",
        "sadelik": f"Tek hamlede en büyük iyileşme, özellikle {profile.distraction_region} tarafındaki gereksiz yükü geri itmek olur.",
        "niyet_tutarliligi": "Tek hamlede en büyük iyileşme, fotoğrafın neden var olduğunu daha açık hissettirmek olur.",
        "isik_yonu": "Tek hamlede en büyük iyileşme, ışığın ana özneye hizmet eden yönünü daha net kurmak olur.",
        "derinlik_hissi": "Tek hamlede en büyük iyileşme, ön-orta-arka plan ayrımını daha görünür kılmak olur.",
        "dikkat_dagitici_unsurlar": f"Tek hamlede en büyük iyileşme, {profile.distraction_region} bölgesinin görsel ağırlığını bastırmak olur.",
        "zamanlama": "Tek hamlede en büyük iyileşme, daha isabetli anı beklemek olur.",
        "negatif_alan": "Tek hamlede en büyük iyileşme, boşluğu daha bilinçli bir anlatı aracına çevirmek olur.",
        "duygusal_yogunluk": "Tek hamlede en büyük iyileşme, duygusal odağı daha görünür hale getirmek olur.",
        "editoryal_deger": "Tek hamlede en büyük iyileşme, karenin editöryel cümlesini daha keskin kurmak olur.",
        "tekrar_bakma_istegi": f"Tek hamlede en büyük iyileşme, {profile.secondary_region} tarafında ikinci bakış için daha güçlü bir katman bırakmak olur.",
    }
    return tone_text(mapping[key], editor_mode)


def build_reading_prompts(scores_100: Dict[str, float], profile: SceneProfile) -> List[str]:
    prompts = [
        f"Gözüm gerçekten önce {profile.primary_region} bölgesine mi gidiyor?",
        f"{profile.subject_hint.capitalize()} fotoğrafın asıl nedeni olarak okunuyor mu?",
        f"{profile.distraction_region.capitalize()} tarafı ana anlatının önüne geçiyor mu?",
        f"Bu karenin kurduğu {profile.visual_mood} atmosfer sahnenin asıl duygusuna hizmet ediyor mu?",
        f"Işığın {profile.light_character} karakteri burada anlam kuruyor mu?",
    ]
    return prompts[:4]


def _normalized_comment_corpus(ai_report: Dict) -> str:
    comments = ai_report.get("editor_comments", {}) if isinstance(ai_report, dict) else {}
    parts: List[str] = []
    if isinstance(comments, dict):
        for value in comments.values():
            if isinstance(value, str) and value.strip():
                parts.append(value.strip().lower())
    for key in ("scene_summary", "global_summary", "one_line_caption"):
        value = ai_report.get(key) if isinstance(ai_report, dict) else None
        if isinstance(value, str) and value.strip():
            parts.append(value.strip().lower())
    return "\n".join(parts)


def _append_unique_note(target: List[str], note: str) -> None:
    note = (note or "").strip()
    if not note:
        return
    if note not in target:
        target.append(note)


def derive_dynamic_summary_sections(ai_report: Dict, result: CritiqueResult) -> Tuple[List[str], str, str, str]:
    strengths = list(result.strengths or [])
    first_reading = result.first_reading
    structural_reading = result.structural_reading
    editorial_result = result.editorial_result

    if not isinstance(ai_report, dict) or not ai_report:
        return strengths[:4], first_reading, structural_reading, editorial_result

    source = str(ai_report.get("_source") or "")
    if source not in {"vision", "qwen"}:
        return strengths[:4], first_reading, structural_reading, editorial_result

    comments = ai_report.get("editor_comments") if isinstance(ai_report.get("editor_comments"), dict) else {}
    sel_comment = comments.get("Sevgin Cingöz") if isinstance(comments, dict) else None
    gulcan_comment = comments.get("Gülcan Ceylan Çağın") if isinstance(comments, dict) else None

    scene_summary = str(ai_report.get("scene_summary") or "").strip()
    global_summary = str(ai_report.get("global_summary") or "").strip()
    concrete = ai_report.get("concrete_details") if isinstance(ai_report.get("concrete_details"), list) else []
    meaning_layers = ai_report.get("meaning_layers") if isinstance(ai_report.get("meaning_layers"), list) else []

    dynamic_strengths: List[str] = []
    for item in concrete[:3]:
        if isinstance(item, str) and item.strip():
            cleaned = item.strip()
            if not cleaned.endswith('.'):
                cleaned += '.'
            dynamic_strengths.append(cleaned[0].upper() + cleaned[1:])
    for item in meaning_layers[:2]:
        if isinstance(item, str) and item.strip() and len(dynamic_strengths) < 4:
            cleaned = item.strip()
            if not cleaned.endswith('.'):
                cleaned += '.'
            dynamic_strengths.append(cleaned[0].upper() + cleaned[1:])

    if dynamic_strengths:
        strengths = dynamic_strengths[:4]

    if scene_summary:
        first_reading = scene_summary
    elif global_summary:
        first_reading = global_summary

    if isinstance(sel_comment, str) and sel_comment.strip():
        structural_reading = sel_comment.strip()
    elif global_summary:
        structural_reading = global_summary

    if isinstance(gulcan_comment, str) and gulcan_comment.strip():
        editorial_result = gulcan_comment.strip()
    elif global_summary:
        editorial_result = global_summary

    return strengths[:4], first_reading, structural_reading, editorial_result



def derive_dynamic_action_notes(ai_report: Dict, result: CritiqueResult, editor_mode: str) -> Tuple[List[str], List[str]]:
    """Editör yorumlarından sahneye özgü çekim ve düzenleme notları türetir."""
    base_shooting = list(result.shooting_notes or [])
    base_editing = list(result.editing_notes or [])
    if not isinstance(ai_report, dict) or not ai_report:
        return base_shooting[:4], base_editing[:4]

    source = str(ai_report.get("_source") or "")
    if source not in {"vision", "qwen"}:
        return base_shooting[:4], base_editing[:4]

    profile_data = (result.metrics or {}).get("scene_profile", {})
    profile = SceneProfile(**profile_data) if isinstance(profile_data, dict) and profile_data else None
    if profile is None:
        return base_shooting[:4], base_editing[:4]

    comments = ai_report.get("editor_comments") if isinstance(ai_report.get("editor_comments"), dict) else {}
    corpus = _normalized_comment_corpus(ai_report)
    if not corpus.strip():
        return base_shooting[:4], base_editing[:4]

    def has_any(*terms: str) -> bool:
        return any(term in corpus for term in terms)

    def pick_comment(*names: str) -> str:
        for name in names:
            value = comments.get(name, "") if isinstance(comments, dict) else ""
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return ""

    sevgin = pick_comment("Sevgin Cingöz")
    guler = pick_comment("Güler Ataşer")
    murside = pick_comment("Mürşide Çilengir")
    gulcan = pick_comment("Gülcan Ceylan Çağın")
    selahattin = pick_comment("Selahattin Kalaycı")

    shooting: List[str] = []
    editing: List[str] = []

    # Kompozisyon / ağırlık / göz akışı
    if has_any("denge", "yerleşim", "göz akışı", "akış", "sol merkez", "sağ üst", "merkez", "ağırlık", "yük"):
        _append_unique_note(
            shooting,
            tone_text(
                f"Çekim anında {profile.primary_region} ile {profile.secondary_region} arasındaki görsel yükü bir adım yer değiştirerek ya da açı kırarak daha kararlı kurmak fotoğrafın ana cümlesini netleştirir.",
                editor_mode,
            ),
        )
        _append_unique_note(
            editing,
            tone_text(
                f"{profile.distraction_region.capitalize()} tarafını yarım ton geri itip {profile.primary_region} çevresinde hafif lokal kontrast toplamak bakışın dağılmasını azaltır.",
                editor_mode,
            ),
        )

    # İnsan / figür / beden dili
    if has_any("figür", "insan", "beden", "yüz", "yakınlık", "yalnız", "ilişki", "jest") or profile.face_count >= 1:
        _append_unique_note(
            shooting,
            tone_text(
                f"{profile.subject_hint.capitalize()} ile arka plan arasına biraz daha nefes açmak ya da çekim mesafesini küçük bir adımla ayarlamak, insani teması daha okunur hale getirir.",
                editor_mode,
            ),
        )
        _append_unique_note(
            editing,
            tone_text(
                f"{profile.subject_position.capitalize()} duran figürün çevresindeki ikincil enerjiyi seçici biçimde sakinleştirmek, öznenin sahne içindeki varlığını daha belirgin kılar.",
                editor_mode,
            ),
        )

    # Işık / ton / yüzey
    if has_any("ışık", "ton", "parlak", "gölge", "kontrast", "yüzey", "sis", "hava", "atmosfer"):
        _append_unique_note(
            shooting,
            tone_text(
                "Çekim anında ışığın yüzeye daha temiz oturduğu ânı kollamak ya da yarım adım açı değiştirerek sert parlamayı kırmak fotoğrafın atmosferini güçlendirir.",
                editor_mode,
            ),
        )
        _append_unique_note(
            editing,
            tone_text(
                f"Işığın kurduğu {profile.visual_mood} havayı bozmadan parlak bölgeleri hafifçe bastırıp gölge bilgisini kontrollü açmak ton cümlesini rafine eder.",
                editor_mode,
            ),
        )

    # Zamanlama / an / jest
    if has_any("an", "zaman", "zamanlama", "beklemek", "yarım saniye", "yürüyen", "adım", "hareket"):
        _append_unique_note(
            shooting,
            tone_text(
                "Yarım saniye daha sabırla beklemek ya da jestin tam kurulduğu anda deklanşöre basmak sahnenin anlatı gücünü belirgin biçimde yükseltebilir.",
                editor_mode,
            ),
        )

    # Editoryal / seçki / crop
    if has_any("seçki", "editoryal", "yayın", "crop", "toparlama", "karar") or gulcan:
        _append_unique_note(
            editing,
            tone_text(
                f"Hafif bir crop ile {profile.distraction_region} tarafındaki gereksiz yükü ayıklamak ve ana cümleyi sıkılaştırmak seçki kararını daha ikna edici hale getirir.",
                editor_mode,
            ),
        )

    # Yorumlara göre daha özgül yönlendirme
    if "sol" in sevgin and "merkez" in sevgin:
        _append_unique_note(
            shooting,
            tone_text("Çekim anında sol tarafta biriken enerjiyi azaltacak küçük bir yer değişimi, kompozisyonun omurgasını daha temiz kurar.", editor_mode),
        )
    if "sağ üst" in guler or "üst bölüm" in guler or "parlak" in guler:
        _append_unique_note(
            editing,
            tone_text("Üst bölümdeki parlaklığı kontrollü biçimde bastırmak, gözün gereksiz sıçramasını azaltır ve ana duyguyu daha sakin taşır.", editor_mode),
        )
    if ("yalnız" in murside or "yakınlık" in murside or "mesafe" in murside) and profile.face_count >= 1:
        _append_unique_note(
            shooting,
            tone_text("Figürle arandaki mesafeyi çok az değiştirerek beden dilini daha okunur bırakmak duygusal çekirdeği güçlendirir.", editor_mode),
        )
    if "niyet" in selahattin or "asıl" in selahattin or "hikâye" in selahattin:
        _append_unique_note(
            shooting,
            tone_text("Çekim sırasında ana hikâyeyi taşıyan tek ilişkiyi daha net seçmek ve ikincil unsurları geri plana itmek fotoğrafın niyetini berraklaştırır.", editor_mode),
        )

    if not shooting:
        shooting = base_shooting[:3]
    if not editing:
        editing = base_editing[:3]

    # aynı notu iki bölümde tekrarlama
    filtered_editing: List[str] = []
    shoot_roots = {s.split(",")[0].strip() for s in shooting}
    for note in editing:
        root = note.split(",")[0].strip()
        if note not in shooting and root not in shoot_roots:
            filtered_editing.append(note)
    if filtered_editing:
        editing = filtered_editing
    else:
        editing = [n for n in base_editing[:3] if n not in shooting] or base_editing[:3]

    return shooting[:4], editing[:4]

def build_shooting_notes(metrics: ImageMetrics, scores_100: Dict[str, float], mode: str, profile: SceneProfile, editor_mode: str) -> List[str]:
    notes = []
    if scores_100["odak_ve_hiyerarsi"] < 60:
        notes.append(f"Çekim anında {profile.subject_hint} etrafındaki görsel ağırlığı biraz daha net kurmak kareyi belirgin biçimde güçlendirebilir.")
    if scores_100["sadelik"] < 60:
        notes.append(f"Özellikle {profile.distraction_region} tarafında kadrajı sadeleştirmek fotoğrafın nefesini açar.")
    if scores_100["kompozisyon"] < 60:
        notes.append(f"{profile.primary_region} ile {profile.secondary_region} arasında daha bilinçli bir denge kurulursa göz akışı güçlenir.")
    if scores_100["zamanlama"] < 60:
        notes.append("Anı biraz daha sabırla beklemek ya da yarım adım önce davranmak sahnenin etkisini yükseltebilir.")
    if scores_100["derinlik_hissi"] < 60:
        notes.append("Ön plan, orta plan ve arka plan ayrımı biraz daha bilinçli kurulursa kare hacim kazanır.")
    if mode == "Portre" or profile.face_count >= 1:
        notes.append("Yüz ve beden dili daha temiz okunacak kadar yakınlık ya da ayrışma kurmak portre etkisini artırabilir.")
    if mode == "Belgesel":
        notes.append("Bağlamı taşıyan ikincil ayrıntıları tamamen öldürmeden ama ana özneden de çaldırmadan kurmak önemli olur.")
    unique = []
    for n in notes:
        t = tone_text(n, editor_mode)
        if t not in unique:
            unique.append(t)
    return unique[:5] if unique else ["Bu kare küçük ama doğru çekim kararlarıyla daha da güçlenebilir."]


def build_editing_notes(metrics: ImageMetrics, scores_100: Dict[str, float], profile: SceneProfile, editor_mode: str) -> List[str]:
    notes = []
    if metrics.highlight_clip_ratio > 0.03:
        notes.append("Parlak bölgeleri biraz geri çekmek, dikkat dengesini daha kontrollü hale getirir.")
    if metrics.shadow_clip_ratio > 0.08:
        notes.append("Siyahları derinleştirirken bilgi kaybını azaltmak kareyi daha rafine gösterir.")
    if scores_100["odak_ve_hiyerarsi"] < 60:
        notes.append(f"{profile.primary_region.capitalize()} çevresinde lokal kontrast ya da parlaklık desteği vermek gözün tutunmasını kolaylaştırır.")
    if scores_100["dikkat_dagitici_unsurlar"] < 60:
        notes.append(f"{profile.distraction_region.capitalize()} bölgesini ton olarak biraz geri itmek anlatıyı öne çıkarır.")
    if scores_100["duygusal_yogunluk"] < 60:
        notes.append(f"Ton geçişlerini {profile.visual_mood} atmosferi bozmayacak biçimde biraz daha kontrollü kurmak duygusal etkiyi artırabilir.")
    unique = []
    for n in notes:
        t = tone_text(n, editor_mode)
        if t not in unique:
            unique.append(t)
    return unique[:5] if unique else ["Bu kare için aşırı filtre yerine küçük ve bilinçli ton dokunuşları en iyi sonucu verir."]


def build_editor_summary(total: float, strengths: List[str], dev_areas: List[str], mode: str, profile: SceneProfile, editor_mode: str) -> str:
    level = score_band(total)
    prefix = EDITOR_MODES[editor_mode]["summary_prefix"]
    improve = EDITOR_MODES[editor_mode]["improve_prefix"]
    text = (
        f"{prefix} Bu **{mode.lower()}** kare şu an için **{level}** bir potansiyel gösteriyor. "
        f"Fotoğraf {profile.visual_mood} bir hava kuruyor ve ana okuma {profile.primary_region} çevresinde başlıyor. "
        f"En çok çalışan taraf şu: {strengths[0].lower()} "
        f"{improve} {dev_areas[0].lower()}"
    )
    return tone_text(text, editor_mode)


def critique_image(image_bytes: bytes, mode: str, editor_mode: str) -> CritiqueResult:
    metrics = ImageMetrics(**extract_metrics_cached(image_bytes))
    profile = SceneProfile(**extract_scene_profile_cached(image_bytes))

    effective_mode = mode
    if mode == "Sokak" and profile.scene_type == "Portre" and profile.face_count >= 1:
        effective_mode = "Portre"

    scores = build_rubric_scores(metrics, effective_mode)
    total = weighted_total(scores)
    scores_100 = {k: round(v * 100, 1) for k, v in scores.items()}

    strengths = pick_strengths(scores_100, metrics, profile, editor_mode)
    dev_areas = pick_development_areas(scores_100, metrics, profile, editor_mode)
    suggested_mode, suggested_mode_reason = suggest_mode(metrics, scores_100)
    if profile.scene_type != suggested_mode and profile.human_presence_score > 0.55:
        suggested_mode = profile.scene_type
        suggested_mode_reason = f"Sahne okuması bu kareyi {profile.scene_type.lower()} yönüne çekiyor; çünkü ana vurgu daha çok {profile.subject_hint} üzerinde kuruluyor."

    first_reading, structural_reading, editorial_story = build_story_block(metrics, profile, scores_100, editor_mode)
    editorial_result = tone_text(f"{editorial_story} {build_editorial_result(total, editor_mode)}", editor_mode)

    metrics_payload = {**asdict(metrics), "scene_profile": asdict(profile), "scene_description": build_scene_description_payload(metrics, profile)}

    return CritiqueResult(
        total_score=total,
        overall_level=score_band(total),
        overall_tag=overall_tag_from_scores(scores_100, effective_mode),
        rubric_scores=scores_100,
        strengths=strengths[:4],
        development_areas=dev_areas[:4],
        editor_summary=build_editor_summary(total, strengths[:4], dev_areas[:4], effective_mode, profile, editor_mode),
        first_reading=first_reading,
        structural_reading=structural_reading,
        editorial_result=editorial_result,
        shooting_notes=build_shooting_notes(metrics, scores_100, effective_mode, profile, editor_mode),
        editing_notes=build_editing_notes(metrics, scores_100, profile, editor_mode),
        reading_prompts=build_reading_prompts(scores_100, profile),
        tags=build_tags(scores_100, total, effective_mode),
        key_strength=build_key_strength(scores_100, metrics, profile, editor_mode),
        key_issue=build_key_issue(scores_100, metrics, profile, editor_mode),
        one_move_improvement=build_one_move_improvement(scores_100, metrics, profile, editor_mode),
        suggested_mode=suggested_mode,
        suggested_mode_reason=suggested_mode_reason,
        metrics=metrics_payload,
    )


# === v3 local editor engine override ===

def _clean_sentence(text: str) -> str:
    text = ' '.join((text or '').replace('\n', ' ').split())
    text = text.replace('..', '.').strip(' .')
    return text + ('' if not text or text.endswith('.') else '.')


def _lower_first(text: str) -> str:
    text = (text or '').strip()
    if not text:
        return text
    return text[0].lower() + text[1:]


def _detail_or_fallback(value: str, fallback: str) -> str:
    value = (value or '').strip()
    if not value:
        value = fallback
    return _clean_sentence(value)


def _profile_action_map(profile: Optional["SceneProfile"], scores: Dict[str, float]) -> Dict[str, str]:
    if profile is None:
        return {
            "meaning": "Kadrajın asıl cümlesi biraz daha açık kurulursa fotoğraf niyetini daha net taşır.",
            "light": "Işık vurgusu ana alanı biraz daha toplarsa sahnenin havası daha güçlü duyulur.",
            "composition": "Ana yük ile yan enerji arasındaki denge biraz daha sıkı kurulursa göz akışı toparlanır.",
            "emotion": "İnsan etkisi biraz daha görünür bırakılırsa fotoğraf daha sıcak bir ilişki kurar.",
            "editorial": "Seçki açısından bir sadeleştirme yapılırsa kare daha net karar verir.",
        }
    hierarchy = float(scores.get("odak_ve_hiyerarsi", 0))
    simplicity = float(scores.get("sadelik", 0))
    narrative = float(scores.get("anlati_gucu", 0))
    emotional = float(scores.get("duygusal_yogunluk", 0))
    editorial = float(scores.get("editoryal_deger", 0))

    meaning = (
        f"{profile.subject_hint.capitalize()} daha açık bir niyete bağlanırsa görüntü yalnızca görünmekle kalmaz, neden var olduğunu da söyler."
        if narrative < 62 else
        f"{profile.subject_hint.capitalize()} etrafındaki anlam katmanı biraz daha cesur bırakılırsa kare düşünsel olarak daha uzun kalır."
    )
    light = (
        f"{profile.light_type_detail.capitalize()} ışığın baskısı ana alanı yer yer sertleştiriyor; ton geçişi biraz yumuşarsa sahnenin havası daha derin duyulur."
        if profile.light_character == 'sert' else
        f"Işığın şu anki sakinliği güzel; ana bölgedeki vurgu hafifçe toparlanırsa fotoğrafın nefesi daha temiz hissedilir."
    )
    composition = (
        f"{profile.distraction_region.capitalize()} tarafındaki ikinci enerji biraz sakinleşirse bakış {profile.primary_region} ile {profile.secondary_region} arasındaki hattı daha kararlı izler."
        if simplicity < 63 or hierarchy < 63 else
        f"Ana yük ile yan hat arasındaki denge küçük bir toplamayla daha berrak olur; o zaman göz akışı daha az sekerek ilerler."
    )
    emotion = (
        f"{profile.human_action.capitalize()} duygusu biraz daha görünür kalırsa fotoğrafın insani çekirdeği daha sıcak çalışır."
        if emotional < 64 else
        f"İnsan varlığına açılan alan biraz genişletilirse bu sessiz ilişki izleyicide daha uzun kalır."
    )
    editorial_txt = (
        f"Seçki açısından yükü artıran bölge {profile.distraction_region}; orası toparlandığında kare daha net bir yayın kararı verir."
        if editorial < 66 else
        f"Editoryal omurga kurulmuş; küçük bir sadeleştirme ile kare seçki içinde daha rahat nefes alır."
    )
    return {
        'meaning': _clean_sentence(meaning),
        'light': _clean_sentence(light),
        'composition': _clean_sentence(composition),
        'emotion': _clean_sentence(emotion),
        'editorial': _clean_sentence(editorial_txt),
    }


def _compose_grounded_editor_comment(editor_name: str, result: CritiqueResult) -> str:
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    profile_data = metrics.get('scene_profile', {}) if isinstance(metrics, dict) else {}
    profile = SceneProfile(**profile_data) if profile_data else None
    scores = result.rubric_scores or {}
    if profile is None:
        return _compose_v8_editor_comment(editor_name, result)

    detail1 = _detail_or_fallback(getattr(profile, 'concrete_detail_1', ''), f"Ana görsel ağırlık {profile.subject_position} hattında toplanıyor")
    detail2 = _detail_or_fallback(getattr(profile, 'concrete_detail_2', ''), f"Göz ilk olarak {profile.primary_region} bölgesine gidiyor")
    detail3 = _detail_or_fallback(getattr(profile, 'concrete_detail_3', ''), f"Işık {profile.light_type_detail} bir karakter taşıyor")
    person = _clean_sentence(_human_count_phrase(profile))
    distraction = _clean_sentence(_profile_distraction_text(profile))
    actions = _profile_action_map(profile, scores)

    if editor_name == 'Selahattin Kalaycı':
        parts = [
            _clean_sentence(f"Ben burada önce fotoğrafın neyi görünür kılmak istediğine bakıyorum; ana cümle {profile.subject_position} hattında kuruluyor"),
            detail1,
            _clean_sentence(f"{_lower_first(detail2[:-1])} ama asıl mesele {profile.subject_hint} etrafında kurulan niyetin {profile.environment_type} duygusuyla nasıl genişlediği"),
            _clean_sentence(f"Bu yüzden kare sadece bir kayıt gibi durmuyor; {profile.visual_mood} bir iç düşünce taşıyor"),
            actions['meaning'],
        ]
    elif editor_name == 'Güler Ataşer':
        parts = [
            _clean_sentence("Ben bu karede önce havayı ve teni olmayan yüzeylerin bile nasıl nefes aldığını duyuyorum"),
            detail3,
            _clean_sentence(f"{_lower_first(detail1[:-1])}; {profile.historical_texture_hint}"),
            _clean_sentence(f"{profile.secondary_region.capitalize()} tarafındaki ikinci hat sert bir gürültü oluşturmadan görüntünün dokusal ritmini uzatıyor"),
            actions['light'],
        ]
    elif editor_name == 'Sevgin Cingöz':
        parts = [
            _clean_sentence("Benim için bu kareyi belirleyen şey yerleşim kararı"),
            detail2,
            _clean_sentence(f"Ana yük {profile.primary_region} ile {profile.secondary_region} arasında bir hat kuruyor; fakat {distraction[:-1]}"),
            _clean_sentence(f"Kompozisyonun iskeleti var ama denge şu an {profile.balance_character} bir yerde duruyor"),
            actions['composition'],
        ]
    elif editor_name == 'Mürşide Çilengir':
        parts = [
            _clean_sentence("Ben burada önce insanın sahneye nasıl değdiğine bakıyorum"),
            person,
            _clean_sentence(f"{profile.human_action.capitalize()} duygusu fotoğrafa sessiz bir yakınlık veriyor ama {profile.subject_hint} bazen insanın önüne geçebiliyor"),
            _clean_sentence(f"{_lower_first(detail3[:-1])}; bu yüzden içtenlik teknik tarafta değil, küçük bir kırılganlıkta görünür oluyor"),
            actions['emotion'],
        ]
    else:
        parts = [
            _clean_sentence("Ben bu kareye seçki mantığıyla bakıyorum ve önce çalışan tarafı teslim etmek gerekiyor"),
            detail1,
            _clean_sentence(f"{_lower_first(detail2[:-1])}; bu yüzden ana cümle tamamen dağılmıyor ve fotoğrafın bir yayın omurgası oluşuyor"),
            _clean_sentence(f"Ama {distraction[:-1]}; bu yük hafiflediğinde karar çok daha temiz verilir"),
            actions['editorial'],
        ]

    return ' '.join(p for p in parts if p).strip()


def _deoverlap_editor_comments(comments: Dict[str, str], profile: Optional["SceneProfile"] = None) -> Dict[str, str]:
    if not isinstance(comments, dict):
        return comments
    out: Dict[str, str] = {}
    used_texts = set()
    closers = {
        'Selahattin Kalaycı': 'Bana göre fotoğrafın asıl değeri burada başlıyor.',
        'Güler Ataşer': 'Ben bu etkinin usulca korunmasından yanayım.',
        'Sevgin Cingöz': 'Benim derdim sadece yapının biraz daha sıkı kurulması.',
        'Mürşide Çilengir': 'Bence fotoğrafın duygusal kapısı tam burada açılıyor.',
        'Gülcan Ceylan Çağın': 'Ben bu karede kapıyı kapatmıyorum; sadece seçki kararını sıkılaştırıyorum.',
    }
    for name in EDITOR_NAMES:
        text = _clean_sentence(comments.get(name, ''))
        if not text:
            text = closers.get(name, '')
        if text in used_texts:
            text = _clean_sentence(text + ' ' + closers.get(name, ''))
        used_texts.add(text)
        out[name] = text
    return out


if __name__ == "__main__":
    main()
