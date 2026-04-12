import os
import re
import io
import math
import json
import base64
import hashlib
import hmac
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from html import escape
from typing import Dict, List, Tuple, Optional
from urllib.parse import quote_plus

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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    font-family: 'Inter', sans-serif;
    color: #F3EEE8;
}

body {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

h1, h2, h3, h4, h5, h6,
.main-title, .section-title, .panel-title, .hero-title,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    color: #F7F1EA;
    letter-spacing: -0.01em;
}

h1, [data-testid="stMarkdownContainer"] h1 {
    font-size: clamp(2.1rem, 1.75rem + 1vw, 2.8rem) !important;
    line-height: 1.08 !important;
    font-weight: 600 !important;
}

h2, [data-testid="stMarkdownContainer"] h2 {
    font-size: clamp(1.55rem, 1.35rem + 0.6vw, 2rem) !important;
    line-height: 1.14 !important;
    font-weight: 600 !important;
}

h3, [data-testid="stMarkdownContainer"] h3 {
    font-size: clamp(1.18rem, 1.05rem + 0.35vw, 1.45rem) !important;
    line-height: 1.2 !important;
    font-weight: 600 !important;
}

p, li, div, span, label, input, textarea, select {
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
.st-emotion-cache-1kyxreq,
.st-emotion-cache-ue6h4q,
.st-emotion-cache-16idsys p {
    font-size: 0.97rem !important;
    line-height: 1.7 !important;
    color: #ECE6DE;
}

small, .small-note, .mini-note, .meta-text, .score-hint-inline {
    font-size: 0.82rem !important;
    line-height: 1.55 !important;
    color: #D9CFC5 !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .sidebar-title {
    font-family: 'Cormorant Garamond', serif !important;
}

.stButton > button, .stDownloadButton > button, button[kind], button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}

button[data-baseweb="tab"] {
    font-size: 0.82rem !important;
    padding-top: 0.42rem !important;
    padding-bottom: 0.42rem !important;
}

.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}

.panel-card, .sidebar-card, .login-card {
    box-shadow: 0 14px 34px rgba(17, 24, 39, 0.06) !important;
}

.hero, .panel-card, .stat-card, .summary-card, .compact-info-card, .upload-panel {
    color: #F2ECE4 !important;
}
.hero p, .panel-card p, .panel-card li, .panel-card div, .stat-card p, .stat-card div,
.summary-card p, .summary-card div, .compact-info-card p, .compact-info-card div,
.upload-panel p, .upload-panel div {
    color: #ECE4DA !important;
}
.hero h1, .hero h2, .hero h3, .panel-card h1, .panel-card h2, .panel-card h3,
.stat-card h1, .stat-card h2, .stat-card h3, .summary-card h1, .summary-card h2, .summary-card h3,
.compact-info-card h1, .compact-info-card h2, .compact-info-card h3 {
    color: #FFF8F1 !important;
}
.hero-badge, .ghost-badge {
    color: #F4ECE4 !important;
}
button[data-baseweb="tab"] {
    color: #F5EFE8 !important;
    background: rgba(255,255,255,0.08) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFF7F0 !important;
    background: rgba(198,120,67,0.28) !important;
}


section[data-testid="stFileUploader"] label,
section[data-testid="stFileUploader"] div,
section[data-testid="stFileUploader"] span,
section[data-testid="stFileUploader"] small,
section[data-testid="stFileUploader"] p {
    color: #EEE6DD !important;
}


/* Selectbox readable on sidebar */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #1F140E !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #F5EEE4 !important;
    color: #1F140E !important;
    border: 1px solid rgba(31,20,14,.18) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #1F140E !important;
    fill: #1F140E !important;
    -webkit-text-fill-color:#1F140E !important;
    opacity: 1 !important;
    font-weight:700 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
    color:#1F140E !important;
    opacity:1 !important;
}


.hero-title-main{
    display:block;
    margin:0;
    font-family:'Cormorant Garamond', serif;
    font-size:clamp(1.95rem,2.8vw,3.6rem);
    line-height:0.95;
    letter-spacing:-0.03em;
    font-weight:700;
    color:#FFF7EE !important;
}
.hero-title-sub{
    display:block;
    margin-top:0.22rem;
    font-family:'Cormorant Garamond', serif;
    font-size:clamp(1.12rem,1.65vw,2.08rem);
    line-height:1.02;
    letter-spacing:-0.02em;
    font-weight:600;
    color:#FFF0E4 !important;
}
.stSelectbox [data-baseweb="select"] > div{
    background:#F5EEE4 !important;
    color:#1F140E !important;
    border:1px solid rgba(31,20,14,.18) !important;
}
.stSelectbox [data-baseweb="select"] *{
    color:#1F140E !important;
    fill:#1F140E !important;
    opacity:1 !important;
    -webkit-text-fill-color:#1F140E !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div{
    background:#F5EEE4 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* --- Hero override: restore two-line title and balanced logo --- */
.hero-title-main{
    display:block !important;
    font-size:clamp(2.05rem, 3.0vw, 3.45rem) !important;
    line-height:0.96 !important;
    margin:0 !important;
    white-space:nowrap !important;
}
.hero-title-sub{
    display:block !important;
    font-size:clamp(1.08rem, 1.45vw, 1.65rem) !important;
    line-height:1.06 !important;
    margin-top:0.24rem !important;
    white-space:nowrap !important;
}
.hero .hero-title-main, .hero .hero-title-sub {
    width:100% !important;
}
.hero .hero-title-sub {
    color:#F2E2D5 !important;
}

/* --- Main content editor selectbox in bronze --- */
[data-testid="stAppViewContainer"] .stSelectbox [data-baseweb="select"] > div,
[data-testid="stAppViewContainer"] div[data-baseweb="select"] > div {
    background: linear-gradient(135deg, #C07A4A, #A8653A) !important;
    border: 1px solid rgba(255,255,255,.18) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 18px rgba(0,0,0,.18) !important;
}
[data-testid="stAppViewContainer"] .stSelectbox [data-baseweb="select"] input,
[data-testid="stAppViewContainer"] .stSelectbox [data-baseweb="select"] span,
[data-testid="stAppViewContainer"] .stSelectbox [data-baseweb="select"] div,
[data-testid="stAppViewContainer"] .stSelectbox [data-baseweb="select"] svg {
    color: #FFF7EF !important;
    fill: #FFF7EF !important;
    -webkit-text-fill-color:#FFF7EF !important;
    opacity:1 !important;
    font-weight:700 !important;
}
[data-testid="stAppViewContainer"] .stSelectbox [data-baseweb="select"] input::placeholder {
    color:#FFF7EF !important;
    opacity:1 !important;
}
/* Keep sidebar type/ton controls readable on light cards */
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background:#F5EEE4 !important;
    border:1px solid rgba(31,20,14,.18) !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg {
    color:#1F140E !important;
    fill:#1F140E !important;
    -webkit-text-fill-color:#1F140E !important;
    opacity:1 !important;
    font-weight:600 !important;
}
/* Dropdown menu readability */
div[role="listbox"], ul[role="listbox"] {
    background:#2B2B2B !important;
    color:#FFF7EE !important;
}
ul[role="listbox"] li, div[role="option"] {
    color:#FFF7EE !important;
}
ul[role="listbox"] li:hover, div[role="option"]:hover {
    background:#C07A4A !important;
    color:#1F140E !important;
}

[data-testid="stAppViewContainer"] ul[role="listbox"] {
    background:#2A211C !important;
    border:1px solid rgba(255,255,255,.12) !important;
}
[data-testid="stAppViewContainer"] ul[role="listbox"] li {
    color:#FFF7EF !important;
}
[data-testid="stAppViewContainer"] ul[role="listbox"] li:hover {
    background:rgba(192,122,74,.35) !important;
}
section[data-testid="stSidebar"] ul[role="listbox"] {
    background:#F5EEE4 !important;
    border:1px solid rgba(31,20,14,.12) !important;
}
section[data-testid="stSidebar"] ul[role="listbox"] li {
    color:#1F140E !important;
}
section[data-testid="stSidebar"] ul[role="listbox"] li:hover {
    background:rgba(192,122,74,.18) !important;
}
</style>
""", unsafe_allow_html=True)




st.markdown("""
<style>
/* --- Upload row restore / clean --- */
section[data-testid="stFileUploader"] {
    margin-top: .35rem !important;
}
section[data-testid="stFileUploader"] > div {
    position: relative !important;
}
section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    position: relative !important;
    min-height: 92px !important;
    border: 2px dashed rgba(255,255,255,0.14) !important;
    border-radius: 22px !important;
    background: rgba(255,255,255,0.02) !important;
    padding: 1rem 1rem 1rem 210px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 1rem !important;
}
section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]::before {
    content: "Yükle";
    position: absolute !important;
    left: 18px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 150px !important;
    height: 54px !important;
    border-radius: 999px !important;
    background: linear-gradient(135deg, #c67843, #b86a38) !important;
    color: #111111 !important;
    font-weight: 800 !important;
    font-size: 1.55rem !important;
    letter-spacing: .01em !important;
    box-shadow: 0 10px 24px rgba(198,120,67,.28) !important;
    z-index: 3 !important;
}
section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
    margin: 0 !important;
    padding: 0 !important;
    flex: 1 1 auto !important;
}
section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child,
section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child * {
    display: none !important;
}
section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] small,
section[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] small * {
    color: #efe5d7 !important;
    opacity: 1 !important;
    font-size: 1.0rem !important;
    font-weight: 700 !important;
}
section[data-testid="stFileUploader"] button[kind="secondary"] {
    font-size: 0 !important;
    min-width: 142px !important;
    height: 54px !important;
    border-radius: 999px !important;
    background: linear-gradient(135deg, #c67843, #b86a38) !important;
    color: transparent !important;
    border: 1px solid rgba(35,22,14,0.18) !important;
    box-shadow: 0 10px 24px rgba(198,120,67,.28) !important;
    position: relative !important;
}
section[data-testid="stFileUploader"] button[kind="secondary"]::after {
    content: "DOSYA";
    position: absolute !important;
    inset: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.02rem !important;
    font-weight: 800 !important;
    color: #111111 !important;
    letter-spacing: .02em !important;
}
section[data-testid="stFileUploader"] button[kind="secondary"] * {
    visibility: hidden !important;
}
section[data-testid="stFileUploader"] > label,
section[data-testid="stFileUploader"] label {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

EMBEDDED_LOGO_B64 = """/9j/4AAQSkZJRgABAQEAeAB4AAD/4QLgRXhpZgAATU0AKgAAAAgABAE7AAIAAAAHAAABSodpAAQAAAABAAABUpydAAEAAAAOAAACyuocAAcAAAEMAAAAPgAAAAAc6gAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAc2VyZGFyAAAABZADAAIAAAAUAAACoJAEAAIAAAAUAAACtJKRAAIAAAADMDUAAJKSAAIAAAADMDUAAOocAAcAAAEMAAABlAAAAAAc6gAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMjAyNjowNDoxMCAxMzo0NDowNAAyMDI2OjA0OjEwIDEzOjQ0OjA0AAAAcwBlAHIAZABhAHIAAAD/4QQZaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA8P3hwYWNrZXQgYmVnaW49J++7vycgaWQ9J1c1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCc/Pg0KPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyI+PHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj48cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0idXVpZDpmYWY1YmRkNS1iYTNkLTExZGEtYWQzMS1kMzNkNzUxODJmMWIiIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIvPjxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSJ1dWlkOmZhZjViZGQ1LWJhM2QtMTFkYS1hZDMxLWQzM2Q3NTE4MmYxYiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIj48eG1wOkNyZWF0ZURhdGU+MjAyNi0wNC0xMFQxMzo0NDowNC4wNDg8L3htcDpDcmVhdGVEYXRlPjwvcmRmOkRlc2NyaXB0aW9uPjxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSJ1dWlkOmZhZjViZGQ1LWJhM2QtMTFkYS1hZDMxLWQzM2Q3NTE4MmYxYiIgeG1sbnM6ZGM9Imh0dHA6Ly9wdXJsLm9yZy9kYy9lbGVtZW50cy8xLjEvIj48ZGM6Y3JlYXRvcj48cmRmOlNlcSB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPjxyZGY6bGk+c2VyZGFyPC9yZGY6bGk+PC9yZGY6U2VxPg0KCQkJPC9kYzpjcmVhdG9yPjwvcmRmOkRlc2NyaXB0aW9uPjwvcmRmOlJERj48L3g6eG1wbWV0YT4NCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA8P3hwYWNrZXQg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"""


def get_embedded_logo_image() -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(EMBEDDED_LOGO_B64))).convert("RGB")


LOGO_PATH = os.path.join(os.path.dirname(__file__), "cofsat_logo.png")


def get_brand_logo_image() -> Image.Image:
    try:
        if os.path.exists(LOGO_PATH):
            return Image.open(LOGO_PATH).convert("RGBA")
    except Exception:
        pass
    return get_embedded_logo_image().convert("RGBA")


def get_brand_logo_data_url(max_size: int = 220) -> str:
    logo = get_brand_logo_image().convert("RGBA")
    w, h = logo.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        logo = logo.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buffer = io.BytesIO()
    logo.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def get_circular_brand_logo_data_url(size: int = 360, y_offset_ratio: float = 0.08) -> str:
    logo = get_brand_logo_image().convert("RGBA")
    w, h = logo.size
    crop_size = min(w, h)
    left = max((w - crop_size) // 2, 0)
    top = max(int((h - crop_size) // 2 + crop_size * y_offset_ratio), 0)
    if top + crop_size > h:
        top = max(h - crop_size, 0)
    logo = logo.crop((left, top, left + crop_size, top + crop_size)).resize((size, size))

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size - 1, size - 1), fill=255)

    circular = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    circular.paste(logo, (0, 0), mask)

    buffer = io.BytesIO()
    circular.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def render_brand_header(title: str, subtitle: str | None = None, logo_width: int = 92) -> None:
    logo_col, text_col = st.columns([0.13, 0.87])
    with logo_col:
        st.image(get_brand_logo_image(), width=logo_width)
    with text_col:
        subtitle_html = f'<div style="color:#6b7280;font-size:0.98rem;margin-top:0.2rem;">{escape(subtitle)}</div>' if subtitle else ''
        st.markdown(
            f"""
            <div style="display:flex;flex-direction:column;justify-content:center;min-height:{max(logo_width, 84)}px;">
                <div style="font-size:2.15rem;font-weight:800;letter-spacing:0.02em;color:#1f2937;line-height:1.05;margin:0;">{escape(title)}</div>
                {subtitle_html}
            </div>
            """,
            unsafe_allow_html=True,
        )



def check_password() -> bool:
    def password_entered() -> None:
        entered_password = st.session_state.get("password", "")
        expected_password = str(st.secrets.get("APP_PASSWORD", ""))
        if expected_password and hmac.compare_digest(entered_password, expected_password):
            st.session_state["password_correct"] = True
            st.session_state.pop("password", None)
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.markdown("""
    <style>
    .login-shell{max-width:920px;margin:2.4rem auto 0 auto;padding:1.2rem 0 0.4rem 0;}
    .login-card{background:linear-gradient(180deg, rgba(255,255,255,.92), rgba(247,248,250,.95));border:1px solid rgba(198,120,67,.18);border-radius:26px;padding:1.5rem 1.5rem 1.2rem 1.5rem;box-shadow:0 18px 48px rgba(15,23,42,.08);}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='login-shell'><div class='login-card'>", unsafe_allow_html=True)
    render_brand_header(
        "ÇOFSAT Giriş",
        "Fotoğraf Ön Değerlendirme sistemine erişmek için giriş şifresini girin.",
        logo_width=118,
    )
    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
    st.text_input(
        "Şifre",
        type="password",
        key="password",
        on_change=password_entered,
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Şifre yanlış.")
    st.markdown("</div></div>", unsafe_allow_html=True)

    return False


if not check_password():
    st.stop()
# v20.7: premium photo-of-day panel redesign

# -------------------------------------------------------
# ÇOFSAT - Fotoğraf Ön Değerlendirme Platformu
# Version : v20.7
# Date    : 2026-04-10
# Notes   :
# - "resim" terminolojisi kaldırıldı.
# - Fotoğraf terminolojisi standardize edildi.
# - Günün fotoğrafı kayıt ve seçim sistemi eklendi.
# - Günün fotoğrafı görünürlüğü güçlendirildi.
# - Türkiye saati görünürlüğü eklendi.
# - 16:00 seçim mantığı ekranda netleştirildi.
# -------------------------------------------------------

PHI = 1.61803398875
MAX_ANALYSIS_SIZE = 1600


OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
QWEN_VISION_MODEL = os.getenv("QWEN_VISION_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")

TURKEY_TZ = ZoneInfo("Europe/Istanbul")
PHOTO_HISTORY_FILE = "cofsat_photo_history.json"
PHOTO_OF_DAY_HOUR = 16
PHOTO_POOL_CLEANUP_HOUR = 20


def enforce_photography_terminology(text: str) -> str:
    if not isinstance(text, str):
        return text

    replacements = {
        r"\bresimde\b": "fotoğrafta",
        r"\bResimde\b": "Fotoğrafta",
        r"\bresmin\b": "fotoğrafın",
        r"\bResmin\b": "Fotoğrafın",
        r"\bresme\b": "fotoğrafa",
        r"\bResme\b": "Fotoğrafa",
        r"\bresmi\b": "fotoğrafı",
        r"\bResmi\b": "Fotoğrafı",
        r"\bresim\b": "fotoğraf",
        r"\bResim\b": "Fotoğraf",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text


def apply_terminology_fix(payload):
    if isinstance(payload, str):
        return enforce_photography_terminology(payload)
    if isinstance(payload, list):
        return [apply_terminology_fix(item) for item in payload]
    if isinstance(payload, dict):
        fixed = {}
        for key, value in payload.items():
            if key.startswith("_"):
                fixed[key] = value
            else:
                fixed[key] = apply_terminology_fix(value)
        return fixed
    return payload


def _safe_json_load(path: str) -> list:
    try:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _safe_json_save(path: str, data: list) -> None:
    try:
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _thumbnail_data_url(image_bytes: bytes, max_side: int = 220) -> str:
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        img.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=82)
        return _image_bytes_to_data_url(buffer.getvalue())
    except Exception:
        return ""



def _round_score_bucket(value: float) -> int:
    try:
        value = float(value)
    except Exception:
        return 0
    value = max(1.0, min(100.0, value))
    return int(round(value / 10.0) * 10)


def calculate_editor_average(editor_scores: Dict[str, float] | None) -> float:
    if not isinstance(editor_scores, dict):
        return 0.0
    values = []
    for v in editor_scores.values():
        try:
            fv = float(v)
        except Exception:
            continue
        if fv > 0:
            values.append(max(1.0, min(100.0, fv)))
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def calculate_final_score(system_score: float, editor_avg: float) -> float:
    try:
        system_score = float(system_score)
    except Exception:
        system_score = 0.0
    try:
        editor_avg = float(editor_avg)
    except Exception:
        editor_avg = 0.0
    if editor_avg <= 0:
        return round(system_score, 2)
    return round((system_score * 0.40) + (editor_avg * 0.60), 2)


def get_entry_display_score(entry: dict) -> float:
    try:
        if isinstance(entry, dict):
            if entry.get("final_score") is not None:
                return float(entry.get("final_score", 0.0))
            if entry.get("score") is not None:
                return float(entry.get("score", 0.0))
            if entry.get("system_score") is not None:
                return float(entry.get("system_score", 0.0))
    except Exception:
        pass
    return 0.0


def _date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _get_upload_pool_date(now: datetime | None = None) -> str:
    now = now or datetime.now(TURKEY_TZ)
    if now.time() < time(PHOTO_OF_DAY_HOUR, 0):
        return _date_str(now)
    return _date_str(now + timedelta(days=1))


def _get_display_pool_date(now: datetime | None = None) -> str:
    now = now or datetime.now(TURKEY_TZ)
    if now.time() < time(PHOTO_POOL_CLEANUP_HOUR, 0):
        return _date_str(now)
    return _date_str(now + timedelta(days=1))


def _cleanup_expired_photo_pools(now: datetime | None = None) -> None:
    now = now or datetime.now(TURKEY_TZ)
    cutoff_date = _get_display_pool_date(now)
    data = _safe_json_load(PHOTO_HISTORY_FILE)
    filtered = []
    changed = False
    for item in data:
        item_date = str(item.get("date", ""))
        if item_date and item_date < cutoff_date:
            changed = True
            continue
        filtered.append(item)
    if changed:
        _safe_json_save(PHOTO_HISTORY_FILE, filtered[-500:])


def save_photo_result(image_bytes: bytes, uploaded_name: str, result) -> None:
    now = datetime.now(TURKEY_TZ)
    _cleanup_expired_photo_pools(now)
    pool_date = _get_upload_pool_date(now)
    upload_date = _date_str(now)
    image_hash = hashlib.md5(image_bytes).hexdigest()
    system_score = float(getattr(result, "total_score", 0.0) or 0.0)
    data = _safe_json_load(PHOTO_HISTORY_FILE)

    existing = None
    kept = []
    for item in data:
        if item.get("date") == pool_date and item.get("image_hash") == image_hash:
            existing = item
        else:
            kept.append(item)

    editor_scores = {}
    if isinstance(existing, dict) and isinstance(existing.get("editor_scores"), dict):
        editor_scores = existing.get("editor_scores", {})

    editor_avg = calculate_editor_average(editor_scores)
    final_score = calculate_final_score(system_score, editor_avg)

    kept.append({
        "date": pool_date,
        "upload_date": upload_date,
        "image_hash": image_hash,
        "filename": uploaded_name or "fotoğraf",
        "system_score": round(system_score, 2),
        "editor_scores": editor_scores,
        "editor_avg": round(editor_avg, 2),
        "final_score": round(final_score, 2),
        "score": round(final_score, 2),
        "mode": getattr(result, "suggested_mode", ""),
        "level": getattr(result, "overall_level", ""),
        "timestamp": now.isoformat(),
        "thumbnail": _thumbnail_data_url(image_bytes),
    })
    _safe_json_save(PHOTO_HISTORY_FILE, kept[-500:])


def update_photo_editor_score(image_hash: str, editor_name: str, score: int, selected_date: str | None = None) -> None:
    now = datetime.now(TURKEY_TZ)
    _cleanup_expired_photo_pools(now)
    target_date = selected_date or _get_display_pool_date(now)
    data = _safe_json_load(PHOTO_HISTORY_FILE)
    for item in data:
        if item.get("date") == target_date and item.get("image_hash") == image_hash:
            editor_scores = item.get("editor_scores", {})
            if not isinstance(editor_scores, dict):
                editor_scores = {}
            editor_scores[editor_name] = _round_score_bucket(score)
            item["editor_scores"] = editor_scores
            system_score = float(item.get("system_score", item.get("score", 0.0)) or 0.0)
            editor_avg = calculate_editor_average(editor_scores)
            final_score = calculate_final_score(system_score, editor_avg)
            item["editor_avg"] = round(editor_avg, 2)
            item["final_score"] = round(final_score, 2)
            item["score"] = round(final_score, 2)
            break
    _safe_json_save(PHOTO_HISTORY_FILE, data)


def get_today_photo_entries(selected_date: str | None = None) -> list:
    now = datetime.now(TURKEY_TZ)
    _cleanup_expired_photo_pools(now)
    target_date = selected_date or _get_display_pool_date(now)
    data = _safe_json_load(PHOTO_HISTORY_FILE)
    todays = [item for item in data if item.get("date") == target_date]
    return sorted(todays, key=lambda x: (get_entry_display_score(x), x.get("timestamp", "")), reverse=True)


def delete_photo_entry(image_hash: str, selected_date: str | None = None) -> None:
    now = datetime.now(TURKEY_TZ)
    target_date = selected_date or _get_display_pool_date(now)
    data = _safe_json_load(PHOTO_HISTORY_FILE)
    filtered = [
        item for item in data
        if not (item.get("date") == target_date and item.get("image_hash") == image_hash)
    ]
    _safe_json_save(PHOTO_HISTORY_FILE, filtered)


def clear_today_photo_entries(selected_date: str | None = None) -> None:
    now = datetime.now(TURKEY_TZ)
    target_date = selected_date or _get_display_pool_date(now)
    data = _safe_json_load(PHOTO_HISTORY_FILE)
    filtered = [item for item in data if item.get("date") != target_date]
    _safe_json_save(PHOTO_HISTORY_FILE, filtered)


def select_photo_of_the_day(selected_date: str | None = None) -> dict:
    now = datetime.now(TURKEY_TZ)
    target_date = selected_date or _get_display_pool_date(now)
    todays = get_today_photo_entries(target_date)
    if not todays:
        return {}
    winner = dict(todays[0])
    today_str = _date_str(now)
    tomorrow_str = _date_str(now + timedelta(days=1))
    if target_date == today_str:
        ready = time(PHOTO_OF_DAY_HOUR, 0) <= now.time() < time(PHOTO_POOL_CLEANUP_HOUR, 0)
    elif target_date == tomorrow_str and now.time() >= time(PHOTO_POOL_CLEANUP_HOUR, 0):
        ready = False
    else:
        ready = target_date < today_str
    winner["ready"] = ready
    winner["entries"] = len(todays)
    winner["display_pool_date"] = target_date
    return winner


def render_photo_of_day_candidates(context: str = "main") -> None:

    now = get_turkey_now()
    display_pool_date = _get_display_pool_date(now)
    entries = get_today_photo_entries(display_pool_date)
    if not entries:
        return

    pod = select_photo_of_the_day(display_pool_date)
    if not pod:
        return

    is_future_pool = display_pool_date != _date_str(now)
    if is_future_pool:
        title = "📸 Bir Sonraki Günün Adayları"
    else:
        title = "📸 Günün Fotoğrafı" if pod.get("ready") else "📸 Günün Fotoğrafı Adayları"

    if context == "sidebar":
        st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
        st.markdown(f"### {title}")
        st.caption(f"Türkiye saati: {now.strftime('%H:%M')}")
        if pod.get("thumbnail"):
            st.image(pod["thumbnail"], use_container_width=True)
        st.markdown(f"**{escape(str(pod.get('filename', 'fotoğraf')))}**")
        st.markdown(
            f"<div class='mini-note'>Nihai puan: {get_entry_display_score(pod):.1f}/100 · Havuz: {int(pod.get('entries', 0))} fotoğraf</div>",
            unsafe_allow_html=True,
        )
        if is_future_pool:
            st.info(f"Saat {PHOTO_OF_DAY_HOUR:02d}:00'dan sonra yüklenen fotoğraflar burada toplanır. Saat {PHOTO_POOL_CLEANUP_HOUR:02d}:00'de bugünün havuzu temizlenir ve bu adaylar görünür kalır.")
        elif pod.get("ready"):
            st.success(f"Türkiye saatine göre {PHOTO_OF_DAY_HOUR:02d}:00 sonrası en yüksek nihai puanlı fotoğraf seçildi. Saat {PHOTO_POOL_CLEANUP_HOUR:02d}:00'de bugünün aday havuzu temizlenir.")
        else:
            st.info(f"Saat {PHOTO_OF_DAY_HOUR:02d}:00'a kadar yüklenen fotoğraflar bugünün havuzuna girer. {PHOTO_OF_DAY_HOUR:02d}:00 sonrası yüklenenler ertesi gün için saklanır.")

        for idx, entry in enumerate(entries[:6]):
            cols = st.columns([0.9, 1.3])
            with cols[0]:
                if entry.get("thumbnail"):
                    st.image(entry["thumbnail"], use_container_width=True)
            with cols[1]:
                st.markdown(f"**{escape(str(entry.get('filename', 'fotoğraf')))}**")
                st.caption(f"Nihai puan: {get_entry_display_score(entry):.1f}/100")
                ts = str(entry.get('timestamp', ''))
                if 'T' in ts:
                    st.caption(f"Saat: {ts[11:16]}")
                if st.button("Sil", key=f"sidebar_delete_{display_pool_date}_{idx}_{entry.get('image_hash', '')}", use_container_width=True):
                    delete_photo_entry(str(entry.get("image_hash", "")), display_pool_date)
                    st.rerun()
        if st.button("Gösterilen adayları temizle", key=f"sidebar_clear_{display_pool_date}", use_container_width=True):
            clear_today_photo_entries(display_pool_date)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)

    badge_label = "🏆 Günün Fotoğrafı" if pod.get("ready") and not is_future_pool else ("⭐ Şu An Lider" if not is_future_pool else "🕘 Ertesi Gün Havuzu")
    leader_name = escape(str(pod.get('filename', 'fotoğraf')))
    leader_score = get_entry_display_score(pod)
    leader_system_score = float(pod.get('system_score', pod.get('score', 0)) or 0.0)
    leader_editor_avg = float(pod.get('editor_avg', 0) or 0.0)
    total_entries = int(pod.get('entries', 0))
    if is_future_pool:
        status_text = f"Bu panelde saat {PHOTO_OF_DAY_HOUR:02d}:00'dan sonra yüklenen fotoğraflar tutulur. Saat {PHOTO_POOL_CLEANUP_HOUR:02d}:00 sonrasında yalnızca bu havuz görünür kalır."
    else:
        status_text = (
            f"Türkiye saatine göre {PHOTO_OF_DAY_HOUR:02d}:00 sonrası en yüksek nihai puanlı fotoğraf otomatik olarak günün fotoğrafı oldu. Saat {PHOTO_POOL_CLEANUP_HOUR:02d}:00'de bu havuz temizlenir."
            if pod.get("ready")
            else f"Saat {PHOTO_OF_DAY_HOUR:02d}:00'a kadar yüklenen fotoğraflar bugünün aday havuzuna girer. {PHOTO_OF_DAY_HOUR:02d}:00 sonrası yüklenenler ertesi gün için saklanır."
        )

    st.markdown("<div class='pod-hero-card'>", unsafe_allow_html=True)
    leader_cols = st.columns([1.35, 1.0], gap="medium")
    with leader_cols[0]:
        if pod.get("thumbnail"):
            st.image(pod["thumbnail"], use_container_width=True)
    with leader_cols[1]:
        st.markdown(f"<div class='pod-badge'>{badge_label}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pod-leader-name'>{leader_name}</div>", unsafe_allow_html=True)
        st.markdown("<div class='pod-subtitle'>Günün fotoğrafı paneli Türkiye saatiyle çalışır ve yükleme saatine göre fotoğrafları doğru havuza ayırır.</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='pod-meta-grid'>
                <div class='pod-meta-card'>
                    <div class='pod-meta-label'>Nihai puan</div>
                    <div class='pod-meta-value'>{leader_score:.1f}/100</div>
                </div>
                <div class='pod-meta-card'>
                    <div class='pod-meta-label'>Sistem puanı</div>
                    <div class='pod-meta-value'>{leader_system_score:.1f}/100</div>
                </div>
                <div class='pod-meta-card'>
                    <div class='pod-meta-label'>Editör ort.</div>
                    <div class='pod-meta-value'>{leader_editor_avg:.1f}/100</div>
                </div>
                <div class='pod-meta-card'>
                    <div class='pod-meta-label'>Toplam aday</div>
                    <div class='pod-meta-value'>{total_entries}</div>
                </div>
                <div class='pod-meta-card'>
                    <div class='pod-meta-label'>Türkiye saati</div>
                    <div class='pod-meta-value'>{now.strftime('%H:%M')}</div>
                </div>
                <div class='pod-meta-card'>
                    <div class='pod-meta-label'>Seçim / temizlik</div>
                    <div class='pod-meta-value'>{PHOTO_OF_DAY_HOUR:02d}:00 · {PHOTO_POOL_CLEANUP_HOUR:02d}:00</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='pod-status-note'>{status_text}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:linear-gradient(90deg, rgba(198,120,67,.65), rgba(255,255,255,.05));margin:20px 0 16px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1.0rem;margin-top:0.1rem;'>Adaylar</div>", unsafe_allow_html=True)
    interactive_pool = context == "main"
    cols = st.columns(3, gap="medium")
    for idx, entry in enumerate(entries[:9]):
        col = cols[idx % 3]
        with col:
            st.markdown("<div class='candidate-card'>", unsafe_allow_html=True)
            if entry.get("thumbnail"):
                st.image(entry["thumbnail"], use_container_width=True)
            st.markdown(f"<div class='candidate-filename'>{escape(str(entry.get('filename', 'fotoğraf')))}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='candidate-score-badge'>Nihai: {get_entry_display_score(entry):.1f}/100 · Sistem: {float(entry.get('system_score', entry.get('score', 0)) or 0.0):.1f} · Editör: {float(entry.get('editor_avg', 0) or 0.0):.1f}</div>", unsafe_allow_html=True)
            ts = str(entry.get('timestamp', ''))
            if 'T' in ts:
                st.caption(f"Saat: {ts[11:16]}")
            if interactive_pool:
                if st.button("🗑️ Sil", key=f"{context}_delete_{display_pool_date}_{idx}_{entry.get('image_hash', '')}", use_container_width=True):
                    delete_photo_entry(str(entry.get("image_hash", "")), display_pool_date)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    if interactive_pool:
        if st.button("🧹 Gösterilen tüm adayları temizle", key=f"{context}_clear_{display_pool_date}", use_container_width=True):
            clear_today_photo_entries(display_pool_date)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_photo_of_the_day_badge() -> None:
    render_photo_of_day_candidates(context="sidebar")


def render_logout_button() -> None:
    if st.sidebar.button("Çıkış Yap", use_container_width=True):
        st.session_state.clear()
        st.rerun()


def get_turkey_now() -> datetime:
    return datetime.now(TURKEY_TZ)


def format_turkey_time(dt: datetime | None = None) -> str:
    current = dt or get_turkey_now()
    return current.strftime("%d.%m.%Y %H:%M")


def render_turkey_time_info(context: str = "sidebar") -> None:
    now = get_turkey_now()
    if context == "sidebar":
        status = "Günün fotoğrafı seçimi aktif durumda." if now.time() >= time(PHOTO_OF_DAY_HOUR, 0) else f"Günün fotoğrafı bugün saat {PHOTO_OF_DAY_HOUR:02d}:00'da seçilir."
        st.markdown(
            f"""
            <div class='sidebar-card'>
                <div class='sidebar-chip'>🕒 Türkiye Saati</div>
                <div class='sidebar-heading'>{escape(now.strftime('%d.%m.%Y %H:%M'))}</div>
                <div class='sidebar-subtext'>{escape(status)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"<div class='mini-note'>🕒 Türkiye saati: <strong>{escape(format_turkey_time(now))}</strong> · Günün fotoğrafı seçimi saat {PHOTO_OF_DAY_HOUR:02d}:00'da yapılır.</div>",
        unsafe_allow_html=True,
    )

def _read_secret_or_env(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
        if value not in {None, ""}:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = _read_secret_or_env(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


def get_openai_api_key() -> str:
    return _read_secret_or_env("OPENAI_API_KEY", "")


def get_llm_provider() -> str:
    provider = _read_secret_or_env("LLM_PROVIDER", "openai").lower()
    return provider if provider in {"openai", "qwen", "local"} else "openai"


def openai_is_forced() -> bool:
    return get_llm_provider() == "openai" or _env_flag("COFSAT_FORCE_OPENAI", False)


def allow_local_fallback() -> bool:
    return _env_flag("COFSAT_ALLOW_LOCAL_FALLBACK", True)


def allow_qwen_fallback() -> bool:
    return _env_flag("COFSAT_ALLOW_QWEN_FALLBACK", False)


def get_provider_debug_snapshot() -> Dict[str, str]:
    return {
        "provider": get_llm_provider(),
        "openai_key": "var" if bool(get_openai_api_key()) else "missing",
        "force_openai": "yes" if openai_is_forced() else "no",
        "allow_qwen_fallback": "yes" if allow_qwen_fallback() else "no",
        "allow_local_fallback": "yes" if allow_local_fallback() else "no",
    }


def openai_vision_available() -> bool:
    return bool(get_openai_api_key())


def qwen_vision_requested() -> bool:
    if openai_is_forced() and not allow_qwen_fallback():
        return False
    return _env_flag("COFSAT_ENABLE_QWEN", False)


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
Sen ÇOFSAT için çalışan, güçlü görsel okuma yapan ve birbirinden net biçimde ayrışan 6 editör sesi üreten fotoğraf yorum motorususun.
Her editör; John Berger, Susan Sontag ve Roland Barthes çizgilerinden türetilmiş genel düşünme biçimlerinden beslenebilir ama bu isimleri asla anmaz, doğrudan taklit yapmaz.

Bu fotoğrafı ÇOFSAT mantığıyla oku.
Yorumlarda:
- her editör kendi persona kaydına sadık kalsın,
- her editör yorumunda fotoğraftan en az 2 somut ayrıntı geçsin,
- figürün yeri, ışığın durumu, mekânın dokusu ve sahne içi gerilim mutlaka hesaba katılsın,
- teknik kadar anlam, duygu, ilişki ve kadraj içi kırılma konuşulsun,
- editörler birbirine benzemesin,
- Ilkay Strebel-Ozmen özellikle ışık, ton, detay, çevresel portre, emek ve yaşam hissi üzerinden; sıcak, içten ve takdir eden bir dille konuşsun,
- her editör kendi kelime dağarcığına ve cümle ritmine sadık kalsın,
- editörler gerekirse birbirine zıt görüş bildirebilsin,
- Berger çizgisi: bakış rejimi, bağlam ve neden böyle görüldüğü,
- Sontag çizgisi: temsil, seçme eylemi, etik/mesafe ve tanıklık,
- Barthes çizgisi: küçük ayrıntının kişisel yarası, punctum etkisi,
- bu çizgiler doğrudan alıntı veya isim vererek değil, yorumların düşünme biçiminde hissedilsin,
- her editör 4-5 cümle kursun,
- cümle açılışları tekrar etmesin,
- genel ve şablon cümlelerden kaçın,
- "resim" kelimesini asla kullanma; bunun yerine fotoğraf, kare, görüntü, sahne veya kadraj de,
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
    "Gülcan Ceylan Çağın": "4-5 cümle",
    "Ilkay Strebel-Ozmen": "4-5 cümle"
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
        return apply_terminology_fix(parsed)
    return apply_terminology_fix({"raw_text": raw_text[:5000], "_model": QWEN_VISION_MODEL, "_provider": "qwen_local"})


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
- "resim" kelimesini asla kullanma; bunun yerine fotoğraf, kare, görüntü, sahne veya kadraj de.
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
        "comment": enforce_photography_terminology((raw_text or "").strip()),
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
"resim" kelimesini asla kullanma; bunun yerine fotoğraf, kare, görüntü, sahne veya kadraj de.
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
        return apply_terminology_fix(parsed)
    return apply_terminology_fix({"raw_text": raw_text[:5000], "_model": OPENAI_VISION_MODEL})


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
    return apply_terminology_fix(result)

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
    "Ilkay Strebel-Ozmen",
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
    "Ilkay Strebel-Ozmen": {
        "focus": ["ışık", "gölge", "ton", "detay", "çevresel portre", "emek", "yaşam hissi", "atmosfer"],
        "tone": "sıcak, zarif, içten ve destekleyici",
        "entry_style": "çoğu zaman ışık, ton ya da hayatın içinden bir ayrıntıyla açar",
        "decision_style": "fotoğrafın insani sıcaklığını, emek duygusunu ve atmosferini görünür kılar",
        "must_do": ["en az bir ışık-gölge ya da ton vurgusu yap", "detay, emek ya da yaşam duygusuna değin", "yorumu sıcak ve takdir eden bir çizgide tut"],
        "avoid": ["ağır teorik jargon", "sert hüküm", "soğuk teknik rapor dili"],
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
    "Ilkay Strebel-Ozmen": {
        "lexicon": ["çok güzel tonlar", "ışık", "gölge", "detay", "çevresel portre", "geçmişin izleri", "yaşamın içinden", "emekçi", "nostalji", "katman", "doku"],
        "openings": [
            "Burada beni ilk karşılayan şey ışığın ve tonların kurduğu hava oluyor:",
            "Bu karede önce yaşamın içinden gelen ayrıntı dikkat çekiyor:",
            "Işıkla gölgenin birlikte kurduğu etki burada çok güzel hissediliyor:"
        ],
        "connectors": ["özellikle", "orada", "aynı zamanda", "bu yüzden"],
        "closings": [
            "Yaşamın içinden gelen bu etki fotoğrafı daha da kıymetli kılıyor.",
            "Tonlar ve detaylar böyle kaldığında görüntü sıcaklığını koruyor.",
            "Samimi ve ilgi çekici bir kare; tebrikler."
        ],
        "cadence": "sıcak giriş / somut görsel ayrıntı / yumuşak takdir",
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
    "Ilkay Strebel-Ozmen": ["Roland Barthes", "John Berger"],
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
    "Ilkay Strebel-Ozmen": {
        "subject_word": "fotoğraf",
        "signature_phrases": ["çok güzel tonlar", "yaşamın içinden", "çevresel portre"],
    },
}

EDITOR_SHARED_WORD_REMAP = {
    "Selahattin Kalaycı": {"fotoğraf": "görüntü", "kare": "görüntü", "ana vurgu": "düşünsel ağırlık"},
    "Güler Ataşer": {"fotoğraf": "kare", "görüntü": "kare", "ana vurgu": "ışık izi"},
    "Sevgin Cingöz": {"fotoğraf": "kompozisyon", "kare": "kompozisyon", "ana vurgu": "taşıyıcı merkez"},
    "Mürşide Çilengir": {"fotoğraf": "sahne", "kare": "sahne", "ana vurgu": "insani çekirdek"},
    "Gülcan Ceylan Çağın": {"fotoğraf": "çalışma", "kare": "çalışma", "ana vurgu": "yayın omurgası"},
    "Ilkay Strebel-Ozmen": {"ana vurgu": "ışık ve tonların ana etkisi", "sorun": "küçük görsel dağınıklık"},
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
            --muted: #eadfd1;
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
        section[data-testid="stSidebar"] .sidebar-shell {
            display:flex;
            flex-direction:column;
            gap:22px;
            padding-bottom:.2rem;
        }
        section[data-testid="stSidebar"] .sidebar-card {
            background: linear-gradient(180deg, rgba(250,242,232,0.96), rgba(242,230,216,0.93)) !important;
            border: 1px solid rgba(83, 41, 9, 0.10) !important;
            box-shadow: 0 16px 36px rgba(82, 39, 11, 0.12) !important;
            backdrop-filter: blur(10px);
            border-radius: 20px !important;
            padding: 1rem 1rem .95rem 1rem !important;
            margin-bottom: 10px !important;
        }
        section[data-testid="stSidebar"] .sidebar-card.sidebar-hero {
            background: linear-gradient(160deg, rgba(252,245,237,0.98), rgba(241,226,211,0.92)) !important;
            box-shadow: 0 18px 38px rgba(82,39,11,0.14) !important;
        }
        section[data-testid="stSidebar"] .sidebar-chip {
            display:inline-flex;
            align-items:center;
            gap:.35rem;
            padding:.34rem .72rem;
            border-radius:999px;
            background: rgba(98,55,26,0.08);
            border: 1px solid rgba(98,55,26,0.10);
            color:#412316 !important;
            font-size:.76rem;
            font-weight:800;
            letter-spacing:.02em;
            text-transform:uppercase;
            margin-bottom:.5rem;
        }
        section[data-testid="stSidebar"] .sidebar-heading {
            font-size:1.06rem;
            font-weight:900;
            line-height:1.2;
            color:#2f170a !important;
            margin:.05rem 0 .3rem 0;
        }
        section[data-testid="stSidebar"] .sidebar-subtext {
            font-size:.88rem;
            line-height:1.58;
            color:#4d2b1b !important;
            opacity:.96;
        }
        section[data-testid="stSidebar"] .sidebar-sep {
            height:1px;
            background: linear-gradient(90deg, rgba(73,36,8,.12), rgba(73,36,8,.04));
            margin:.85rem 0 .7rem 0;
            border:none;
        }
        section[data-testid="stSidebar"] .sidebar-quote {
            background: rgba(255,255,255,.36);
            border: 1px solid rgba(83,41,9,.08);
            border-left: 4px solid rgba(111,63,31,.58);
            border-radius: 14px;
            padding: .82rem .88rem;
            font-size:.9rem;
            line-height:1.58;
            color:#3a1b08 !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.22);
        }
        section[data-testid="stSidebar"] .sidebar-meta-list {
            display:grid;
            gap:.44rem;
            margin-top:.45rem;
        }
        section[data-testid="stSidebar"] .sidebar-meta-item {
            background: rgba(255,255,255,.22);
            border: 1px solid rgba(83,41,9,.08);
            border-radius: 12px;
            padding: .58rem .7rem;
        }
        section[data-testid="stSidebar"] .sidebar-meta-label {
            font-size:.7rem;
            text-transform:uppercase;
            letter-spacing:.06em;
            font-weight:800;
            color:#6c3a1d !important;
            opacity:.88;
            margin-bottom:.18rem;
        }
        section[data-testid="stSidebar"] .sidebar-meta-value {
            font-size:.88rem;
            line-height:1.42;
            color:#2f170a !important;
            font-weight:700;
        }
        section[data-testid="stSidebar"] .sidebar-link-btn {
            display:inline-block;
            text-decoration:none;
            padding:10px 13px;
            border-radius:12px;
            background:#2f170a;
            color:#fff !important;
            font-weight:800;
            font-size:.82rem;
            box-shadow: 0 10px 22px rgba(47,23,10,.18);
            margin-top:.82rem;
        }
        section[data-testid="stSidebar"] .sidebar-footmark {
            margin-top:.2rem;
            padding-top:.25rem;
            font-size:.79rem;
            font-weight:800;
            letter-spacing:.03em;
            color:#3a1b08 !important;
            opacity:.88;
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
            color: #efe5d7 !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] p {
            color: #f4ede3 !important;
            opacity: 1 !important;
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
            font-size: 1.72rem;
            margin: 0 0 .35rem 0;
            line-height: 1.05;
            white-space: nowrap;
            letter-spacing: -0.02em;
        }
        .hero p {
            color: #f5efe6;
            font-size: .98rem;
            margin-bottom: .85rem;
            line-height: 1.55;
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
            color: #f3ede4;
            font-size: .94rem;
            line-height: 1.6;
        }

        .pod-hero-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
            border: 1px solid rgba(198,120,67,.32);
            border-radius: 24px;
            padding: 1.05rem;
            box-shadow: 0 22px 56px rgba(0,0,0,.22);
            margin-top: .35rem;
            margin-bottom: 1rem;
        }
        .pod-badge {
            display:inline-flex;
            align-items:center;
            gap:.35rem;
            padding:.34rem .72rem;
            border-radius:999px;
            background: linear-gradient(135deg, rgba(198,120,67,.98), rgba(210,134,77,.98));
            color:#111;
            font-size:.78rem;
            font-weight:800;
            letter-spacing:.01em;
            box-shadow: 0 10px 24px rgba(198,120,67,.24);
            margin-bottom:.55rem;
        }
        .pod-leader-name {
            font-size:1.38rem;
            font-weight:800;
            line-height:1.15;
            margin:.05rem 0 .55rem 0;
        }
        .pod-meta-grid {
            display:grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap:.65rem;
            margin:.4rem 0 .55rem 0;
        }
        .pod-meta-card {
            background: rgba(255,255,255,.045);
            border: 1px solid rgba(255,255,255,.09);
            border-radius: 16px;
            padding: .75rem .85rem;
        }
        .pod-meta-label {
            font-size:.74rem;
            color: var(--muted);
            margin-bottom:.16rem;
            letter-spacing:.01em;
        }
        .pod-meta-value {
            font-size:1rem;
            font-weight:800;
            color:#f5f7ff;
        }
        .pod-status-note {
            margin-top:.55rem;
            padding:.72rem .82rem;
            border-radius:16px;
            background: rgba(255,255,255,.04);
            border:1px solid rgba(255,255,255,.08);
            color: var(--muted);
            font-size:.91rem;
            line-height:1.55;
        }
        .pod-subtitle {
            color: var(--muted);
            font-size:.92rem;
            margin: .15rem 0 .9rem 0;
        }
        .candidate-card {
            background: rgba(255,255,255,.045);
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 18px;
            padding: .68rem;
            box-shadow: 0 12px 30px rgba(0,0,0,.16);
            min-height: 100%;
        }
        .candidate-card .stImage img {
            border-radius: 12px;
        }
        .candidate-filename {
            font-size: .92rem;
            font-weight: 700;
            margin: .5rem 0 .28rem 0;
            word-break: break-word;
        }
        .candidate-score-badge {
            display:inline-flex;
            align-items:center;
            justify-content:center;
            border-radius:999px;
            padding:.24rem .62rem;
            background: rgba(198,120,67,.18);
            border:1px solid rgba(198,120,67,.36);
            color:#ffd4b9;
            font-size:.78rem;
            font-weight:700;
            margin-bottom:.28rem;
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
        .mini-note, .muted-note, .stat-caption, .upload-hint, .footer-note, .ghost-badge, .hero p {color: #f1e7da !important; opacity: 1 !important;}
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.05);
            border:1px solid rgba(255,255,255,.10);
            border-radius:18px;
            padding: .75rem;
        }
        div[data-testid="stMetric"] label {color: var(--muted)!important;}
        .stTabs [data-baseweb="tab-list"] {
            gap: .28rem;
            background: rgba(255,255,255,.07);
            padding: .28rem;
            border-radius: 16px;
            overflow-x: auto;
            scrollbar-width: thin;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: .46rem .68rem;
            background: rgba(255,255,255,.06);
            font-size: .84rem;
            line-height: 1.08;
            min-height: unset;
            color: #fff6ec !important;
            border: 1px solid rgba(255,255,255,.14);
        }
        .stTabs [data-baseweb="tab"] > div,
        .stTabs [data-baseweb="tab"] p,
        .stTabs [data-baseweb="tab"] span {
            white-space: nowrap;
            color: #f7efe5 !important;
            opacity: 1 !important;
            font-weight: 700 !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(209,161,95,.36), rgba(179,108,68,.28)) !important;
            border: 1px solid rgba(235,196,137,.42) !important;
            box-shadow: 0 8px 20px rgba(0,0,0,.18);
        }
.stSelectbox label {
            color: #f7efe5 !important;
            font-weight: 700 !important;
        }
        .stSelectbox div[data-baseweb="select"] {
            background: rgba(255,255,255,.10) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,.18) !important;
            min-height: 62px !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            background: rgba(255,255,255,.10) !important;
            color: #f7efe5 !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
            min-height: 62px !important;
        }
        .stSelectbox div[data-baseweb="select"] span,
        .stSelectbox div[data-baseweb="select"] input,
        .stSelectbox div[data-baseweb="select"] div,
        .stSelectbox div[data-baseweb="select"] svg {
            color: #f7efe5 !important;
            fill: #f7efe5 !important;
            opacity: 1 !important;
        }
        div[data-baseweb="popover"] * {
            color: #1f1f1f !important;
        }
        
/* Selectbox readable on sidebar */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #1F140E !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #F5EEE4 !important;
    color: #1F140E !important;
    border: 1px solid rgba(31,20,14,.18) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #1F140E !important;
    fill: #1F140E !important;
    -webkit-text-fill-color:#1F140E !important;
    opacity: 1 !important;
    font-weight:700 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
    color:#1F140E !important;
    opacity:1 !important;
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


PHOTO_BOOK_SPOTLIGHT = [
    {"title": "The Decisive Moment", "author": "Henri Cartier-Bresson", "year": "1952", "publisher": "Simon and Schuster / Verve", "genre": "Sokak · Belgesel · Fotoğraf dili", "summary": "Karar anı fikrini fotoğraf düşüncesinin merkezine yerleştiren temel eserlerden biridir.", "reading_value": "Kadraj, zamanlama ve görsel sezginin nasıl tek cümlede birleştiğini görmek için güçlü bir başlangıçtır.", "why_now": "Fotoğrafı yalnızca görmek değil, okumak isteyenler için bakış disiplinini keskinleştirir.", "link": "https://www.magnumphotos.com/photographer/henri-cartier-bresson/"},
    {"title": "The Americans", "author": "Robert Frank", "year": "1958", "publisher": "Grove Press", "genre": "Belgesel · Yol · Toplumsal gözlem", "summary": "Amerikan hayatını parçalı, sert ve kişisel bir bakışla yeniden kuran öncü kitaplardandır.", "reading_value": "Fotoğrafta sekans, atmosfer ve kişisel bakış açısının anlatıyı nasıl dönüştürdüğünü gösterir.", "why_now": "Gündelik hayatın küçük kırılmalarını okumayı öğretir; özellikle sokak ve belgesel için besleyicidir.", "link": "https://steidl.de/Books/The-Americans-0815.html"},
    {"title": "Workers", "author": "Sebastião Salgado", "year": "1993", "publisher": "Aperture", "genre": "Belgesel · Emek · İnsan manzaraları", "summary": "Emek, beden ve ölçeği güçlü siyah-beyaz bir görsel dille bir araya getiren etkileyici bir kitaptır.", "reading_value": "Işık, ritim ve insani yoğunluğun büyük sahnelerde nasıl kurulduğunu anlamak için çok değerlidir.", "why_now": "Yoğun sahnelerde görsel düzen kurmak isteyenler için güçlü bir referans üretir.", "link": "https://www.taschen.com/en/books/photography/01171/sebastiao-salgado-workers/"},
    {"title": "Evidence", "author": "Larry Sultan & Mike Mandel", "year": "1977", "publisher": "D.A.P.", "genre": "Kavramsal · Arşiv · Yeniden bağlamlama", "summary": "Buluntu görselleri yeni bir bağlamda bir araya getirerek fotoğrafın anlam üretme biçimini sarsıcı şekilde açar.", "reading_value": "Fotoğraf okurken yalnızca kadraja değil, bağlama ve sıralamaya da bakmak gerektiğini hatırlatır.", "why_now": "Bir görüntünün ne söylediğini değil, neden öyle okunduğunu sorgulama alışkanlığı kazandırır.", "link": "https://www.dapart.com/book/evidence"},
    {"title": "Sleeping by the Mississippi", "author": "Alec Soth", "year": "2004", "publisher": "Steidl", "genre": "Portre · Yol · Sessiz anlatı", "summary": "Amerikan coğrafyasında yavaş, dikkatli ve duyarlı bir görsel anlatı kuran çağdaş klasiklerden biridir.", "reading_value": "Portre ile mekân arasındaki ilişkiyi ve sessiz atmosfer kurulumunu çok iyi gösterir.", "why_now": "Fotoğrafta sakinlik, mesafe ve duygusal ton kurmak isteyenler için iyi bir eşiktir.", "link": "https://steidl.de/Books/Sleeping-by-the-Mississippi-0227474958.html"},
]


def get_daily_photo_book() -> Dict[str, str]:
    day_index = datetime.now(ZoneInfo("Europe/Istanbul")).timetuple().tm_yday
    return PHOTO_BOOK_SPOTLIGHT[day_index % len(PHOTO_BOOK_SPOTLIGHT)]


def render_daily_photo_book() -> None:
    book = get_daily_photo_book()
    st.markdown(
        f"""
        <div class='sidebar-card sidebar-hero'>
            <div class='sidebar-chip'>📚 Günün Fotoğraf Kitabı</div>
            <div class='sidebar-heading'>{escape(book['title'])}</div>
            <div class='sidebar-subtext'><strong>{escape(book['author'])}</strong> · {escape(book['year'])}</div>
            <div class='sidebar-subtext' style='margin-top:.18rem;'>{escape(book['publisher'])} · {escape(book['genre'])}</div>
            <div class='sidebar-sep'></div>
            <div class='sidebar-subtext'>{escape(book['summary'])}</div>
            <div class='sidebar-meta-list'>
                <div class='sidebar-meta-item'>
                    <div class='sidebar-meta-label'>Fotoğraf okumaya katkısı</div>
                    <div class='sidebar-meta-value'>{escape(book['reading_value'])}</div>
                </div>
                <div class='sidebar-meta-item'>
                    <div class='sidebar-meta-label'>Neden bugün</div>
                    <div class='sidebar-meta-value'>{escape(book['why_now'])}</div>
                </div>
            </div>
            <a href='{book['link']}' target='_blank' class='sidebar-link-btn'>Kitap arşivine git</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar(selected_mode: str, selected_editor_mode: str) -> None:
    with st.sidebar:
        st.markdown("<div class='sidebar-shell'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='sidebar-card sidebar-hero'>
                <div class='sidebar-chip'>📸 Okuma Kontrolü</div>
                <div class='sidebar-heading'>ÇOFSAT kontrol alanı</div>
                <div class='sidebar-meta-list'>
                    <div class='sidebar-meta-item'>
                        <div class='sidebar-meta-label'>Fotoğraf türü</div>
                        <div class='sidebar-meta-value'>{escape(selected_mode)}</div>
                        <div class='sidebar-subtext' style='margin-top:.28rem;'><strong>Tür etkisi:</strong> {escape(MODE_PROFILES[selected_mode]['focus_hint'])}</div>
                    </div>
                    <div class='sidebar-meta-item'>
                        <div class='sidebar-meta-label'>Editör tonu</div>
                        <div class='sidebar-meta-value'>{escape(selected_editor_mode)}</div>
                        <div class='sidebar-subtext' style='margin-top:.28rem;'><strong>Ton etkisi:</strong> {escape(EDITOR_MODES[selected_editor_mode]['ending'])}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_turkey_time_info(context="sidebar")

        st.markdown(
            f"""
            <div class='sidebar-card'>
                <div class='sidebar-chip'>✨ Manifesto Sorusu</div>
                <div class='sidebar-quote'>{escape(CULTURE['temel_soru'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_daily_photo_book()

        st.markdown("<div class='sidebar-footmark'>Serdar Bayram™</div></div>", unsafe_allow_html=True)

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
        "Ilkay Strebel-Ozmen": [8, 3, 5, 9, 10],
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
    if editor_name == "Ilkay Strebel-Ozmen":
        return f"Burada beni en çok çeken şey, {profile.subject_hint} çevresinde ışığın, tonların ve yaşam hissinin birlikte kurduğu sıcak etki oluyor."
    return f"Seçki açısından bakınca bu karenin ilk artısı, {profile.primary_region} çevresinde kurduğu net görsel merkez."


def _editor_issue_sentence(editor_name: str, profile: Optional["SceneProfile"], scores: Dict[str, float]) -> str:
    profile = profile or None
    if editor_name == "Ilkay Strebel-Ozmen":
        sentences = [
            pick([
                f"Burada beni ilk karşılayan şey {detail_1} ve sahnenin taşıdığı yaşam hissi oluyor.",
                f"Işığın ve tonların kurduğu hava, bu fotoğrafı ilk anda sıcak ve ilgi çekici kılıyor.",
            ], "ilkay-open"),
            pick([
                f"{light_type_detail.capitalize()} ışık ile {detail_2}, karede çok güzel bir atmosfer kuruyor.",
                f"{detail_3.capitalize()}; bu yüzden görüntü sadece bakılan değil, hissedilen bir yere dönüşüyor.",
            ], "ilkay-obs"),
            pick([
                f"{environment_type.capitalize()} ve {historical_texture_hint}, fotoğrafa yaşamın içinden gelen samimi bir doku veriyor.",
                f"{relation} Özellikle bu çevresel yapı, sahnenin duygusunu daha görünür kılıyor.",
            ], "ilkay-meaning"),
            pick([issue_core, f"Özellikle {place(distraction)} tarafındaki küçük fazlalıklar hafiflediğinde bu güzel tonlar ve detaylar daha da öne çıkar."], "ilkay-issue"),
            pick([
                "Ben burada sıcak, samimi ve hayatın içinden gelen bir etki görüyorum; tebrikler.",
                "Çok güzel tonlar, detaylar ve doğal bir akış var; küçük bir temizlikle etkisi daha da artar.",
            ], "ilkay-close"),
        ]
        return " ".join(sentences[:5])

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
    if editor_name == "Ilkay Strebel-Ozmen":
        return _score_signal(
            scores,
            "isik_yonu",
            "Işık, ton ve detaylar birlikte çok güzel çalışıyor; bunu küçük bir temizlikle daha da görünür kılmak mümkün.",
            "Fotoğrafın sıcaklığı ve yaşam hissi geliyor; yalnız bazı küçük fazlalıklar ana etkiyi biraz dağıtıyor.",
            f"Ben burada özellikle {getattr(profile, 'distraction_region', 'yan alan')} tarafındaki yükün hafiflemesiyle ışığın kurduğu etkiyi daha temiz duymak isterim."
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
    elif editor_name == "Ilkay Strebel-Ozmen":
        parts = [
            opening,
            f"{obs3[:-1]} ve bu ışık-gölge ilişkisi fotoğrafın havasını çok güzel kuruyor.",
            f"{obs1[:-1]}; detayların ve çevresel yapının birlikte çalışması kareye yaşamın içinden bir sıcaklık veriyor.",
            f"{profile.historical_texture_hint.capitalize()} ile {profile.environment_type}, sahnenin samimiyetini destekliyor; özellikle {profile.subject_hint} çevresinde bu etki daha belirgin hissediliyor.",
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
        tone_mode = "Yapıcı" if editor_name in {"Mürşide Çilengir", "Gülcan Ceylan Çağın", "Ilkay Strebel-Ozmen"} else "Dürüst"
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
            merged["_status_label"] = "Qwen Vision Light aktif"
            merged["_user_message"] = "Yorumlar şu an Qwen Vision Light ile üretiliyor."
            merged["_vision_ok"] = True
            if payload.get("_model"):
                merged["_model"] = payload["_model"]
            return merged
        if isinstance(payload, dict) and payload.get("error"):
            fallback["_qwen_error"] = _safe_provider_error_message(str(payload.get("error")))

    # no providers active
    provider_debug = get_provider_debug_snapshot()
    fallback["_provider_debug"] = provider_debug

    if openai_is_forced() and not openai_vision_available():
        fallback["_status"] = "openai_missing_key"
        fallback["_status_label"] = "OpenAI anahtarı eksik"
        fallback["_user_message"] = "OpenAI seçili ama OPENAI_API_KEY bulunamadı. Secrets/Environment Variables bölümüne OPENAI_API_KEY ekleyin."
        if not allow_local_fallback():
            fallback["editor_comments"] = {
                name: "OpenAI seçili ancak OPENAI_API_KEY tanımlı olmadığı için derin yorum üretilemedi."
                for name in EDITOR_NAMES
            }
        return fallback

    if openai_vision_available():
        fallback["_status"] = "vision_failed_fallback"
        fallback["_status_label"] = "OpenAI başarısız, ÇOFSAT Motoru devrede"
        fallback["_user_message"] = "OpenAI Vision çağrısı başarısız oldu. Yorumlar geçici olarak ÇOFSAT motoru ile üretiliyor."
    elif qwen_vision_runtime_available():
        fallback["_status"] = "qwen_failed_fallback"
        fallback["_status_label"] = "Qwen başarısız, ÇOFSAT Motoru devrede"
        fallback["_user_message"] = "Qwen Vision çağrısı başarısız oldu. Yorumlar geçici olarak ÇOFSAT motoru ile üretiliyor."
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




def render_editor_score_summary_cards(editor_scores: Dict[str, float]) -> None:
    st.markdown(
        """
        <style>
        .editor-score-chip {
            border-radius: 16px;
            padding: 12px 14px;
            border: 1px solid rgba(198,120,67,.28);
            background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(250,245,241,.98));
            box-shadow: 0 8px 20px rgba(15,23,42,.05);
            margin-bottom: 8px;
        }
        .editor-score-name {
            font-weight: 700;
            font-size: 0.95rem;
            color: #1f2937;
            margin-bottom: 4px;
        }
        .editor-score-value {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: linear-gradient(135deg, #c67843, #b91c1c);
            color: white;
            font-weight: 800;
            font-size: 0.92rem;
            letter-spacing: .2px;
        }
        .editor-score-empty {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: rgba(148,163,184,.16);
            color: #475569;
            font-weight: 700;
            font-size: 0.90rem;
        }
        .score-row-label {
            font-weight: 700;
            margin: .55rem 0 .25rem 0;
            color: #111827;
        }
        .score-hint-inline {
            font-size: .88rem;
            color: #6b7280;
            font-weight: 500;
        }
        div[data-testid="stButton"] > button {
            border-radius: 12px;
            min-height: 2.45rem;
            font-weight: 800;
            letter-spacing: .2px;
            border: 1px solid rgba(198,120,67,.22);
            box-shadow: 0 4px 10px rgba(15,23,42,.04);
        }
        
/* Selectbox readable on sidebar */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #1F140E !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #F5EEE4 !important;
    color: #1F140E !important;
    border: 1px solid rgba(31,20,14,.18) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #1F140E !important;
    fill: #1F140E !important;
    -webkit-text-fill-color:#1F140E !important;
    opacity: 1 !important;
    font-weight:700 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
    color:#1F140E !important;
    opacity:1 !important;
}

</style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='score-row-label'>Kim ne puan verdi?</div>", unsafe_allow_html=True)
    cols = st.columns(len(EDITOR_NAMES), gap="small")
    for col, editor_name in zip(cols, EDITOR_NAMES):
        raw_value = editor_scores.get(editor_name, 0) if isinstance(editor_scores, dict) else 0
        try:
            current_value = int(float(raw_value or 0))
        except Exception:
            current_value = 0
        with col:
            badge = f"<span class='editor-score-value'>{current_value} / 100</span>" if current_value else "<span class='editor-score-empty'>Henüz yok</span>"
            st.markdown(
                f"""
                <div class="editor-score-chip">
                    <div class="editor-score-name">{editor_name}</div>
                    {badge}
                </div>
                """,
                unsafe_allow_html=True,
            )

def render_editor_score_controls(image_hash: str, result: CritiqueResult) -> None:
    entries = get_today_photo_entries()
    current_entry = next((item for item in entries if item.get("image_hash") == image_hash), None)

    system_score = float(getattr(result, "total_score", 0.0) or 0.0)
    editor_scores = current_entry.get("editor_scores", {}) if isinstance(current_entry, dict) else {}
    if not isinstance(editor_scores, dict):
        editor_scores = {}
    editor_avg = calculate_editor_average(editor_scores)
    final_score = calculate_final_score(system_score, editor_avg)

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Editör Puanlaması</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='mini-note'>Bilgisayar puanı %40, editör ortalaması %60 etkili olur. "
        "Her editör 1–100 arası puan verir; aşağıdaki küçük puan butonları 10 puanlık adımlarla çalışır.</div>",
        unsafe_allow_html=True,
    )

    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Bilgisayar puanı", f"{system_score:.1f}/100")
    with summary_cols[1]:
        st.metric("Editör ortalaması", f"{editor_avg:.1f}/100")
    with summary_cols[2]:
        st.metric("Nihai puan", f"{final_score:.1f}/100")

    render_editor_score_summary_cards(editor_scores)

    for editor_name in EDITOR_NAMES:
        current_value = int(float(editor_scores.get(editor_name, 0) or 0))
        current_label = f"{current_value} / 100" if current_value else "Henüz puan verilmedi"
        badge_html = (
            f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;background:linear-gradient(135deg,#c67843,#b91c1c);color:#fff;font-weight:800;font-size:.88rem;'>{current_label}</span>"
            if current_value else
            "<span style='display:inline-block;padding:4px 10px;border-radius:999px;background:rgba(148,163,184,.16);color:#475569;font-weight:700;font-size:.88rem;'>Henüz puan verilmedi</span>"
        )
        st.markdown(
            f"""
            <div style='display:flex;align-items:center;justify-content:space-between;gap:12px;margin:.95rem 0 .35rem 0;flex-wrap:wrap;'>
                <div style='font-weight:800;color:#111827;'>{editor_name}</div>
                <div>{badge_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        score_cols = st.columns(10, gap="small")
        for idx, score_value in enumerate(range(10, 101, 10)):
            with score_cols[idx]:
                is_active = current_value == score_value
                if st.button(
                    f"{score_value}",
                    key=f"editor_score_{image_hash}_{editor_name}_{score_value}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    update_photo_editor_score(image_hash, editor_name, score_value)
                    st.rerun()
        st.markdown(
            "<div class='score-hint-inline'>10'luk adımlarla hızlı puan verilir. Seçili puan renkli olarak vurgulanır.</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_single_editor_score_box(image_hash: str, editor_name: str, editor_scores: Dict[str, float], key_prefix: str = "inline") -> None:
    raw_value = editor_scores.get(editor_name, 0) if isinstance(editor_scores, dict) else 0
    try:
        current_value = int(float(raw_value or 0))
    except Exception:
        current_value = 0

    current_label = f"{current_value} / 100" if current_value else "Henüz puan verilmedi"
    badge_html = (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;background:linear-gradient(135deg,#c67843,#b91c1c);color:#fff;font-weight:800;font-size:.88rem;'>{current_label}</span>"
        if current_value else
        "<span style='display:inline-block;padding:4px 10px;border-radius:999px;background:rgba(148,163,184,.16);color:#475569;font-weight:700;font-size:.88rem;'>Henüz puan verilmedi</span>"
    )

    st.markdown("<div class='panel-card' style='margin-top:.7rem;'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display:flex;align-items:center;justify-content:space-between;gap:12px;margin:.05rem 0 .55rem 0;flex-wrap:wrap;'>
            <div style='font-weight:800;color:#111827;'>{editor_name} puanı</div>
            <div>{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_cols = st.columns(10, gap="small")
    for idx, score_value in enumerate(range(10, 101, 10)):
        with score_cols[idx]:
            is_active = current_value == score_value
            if st.button(
                f"{score_value}",
                key=f"{key_prefix}_editor_score_inline_{image_hash}_{editor_name}_{score_value}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                update_photo_editor_score(image_hash, editor_name, score_value)
                st.rerun()

    st.markdown(
        "<div class='score-hint-inline'>Bu editör için puan burada verilir. Seçili puan renkli görünür.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)



def render_editor_comments(
    result: CritiqueResult,
    selected_editor_name: str,
    image_hash: str,
    editor_scores: Dict[str, float],
    ai_report: Optional[Dict] = None,
) -> None:
    st.markdown("<div class='section-title' style='margin-top:1rem;'>Ortak editör özeti</div>", unsafe_allow_html=True)


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



def _band_label(value: float, low: float, high: float, low_label: str, mid_label: str, high_label: str) -> str:
    if value < low:
        return low_label
    if value < high:
        return mid_label
    return high_label


def build_technical_readout(metrics: Dict[str, float]) -> Dict[str, object]:
    brightness = float(metrics.get("brightness_mean", 0.0) or 0.0)
    contrast = float(metrics.get("contrast_std", 0.0) or 0.0)
    edge = float(metrics.get("edge_density", 0.0) or 0.0)
    highlight = float(metrics.get("highlight_clip_ratio", 0.0) or 0.0) * 100.0
    shadow = float(metrics.get("shadow_clip_ratio", 0.0) or 0.0) * 100.0
    symmetry = float(metrics.get("symmetry_score", 0.0) or 0.0) * 100.0
    negative = float(metrics.get("negative_space_score", 0.0) or 0.0) * 100.0
    tension = float(metrics.get("dynamic_tension_score", 0.0) or 0.0) * 100.0

    tone_balance = "dengeli"
    if highlight > 1.5 or shadow > 2.5:
        tone_balance = "riskli"
    elif contrast > 58:
        tone_balance = "güçlü"
    elif contrast < 28:
        tone_balance = "yumuşak"

    detail_density = _band_label(edge, 0.03, 0.09, "çok sade", "orta yoğun", "yüksek yoğun")
    composition_stability = _band_label((symmetry + negative) / 2.0, 38, 64, "serbest", "dengeli", "yerleşik")
    technical_risk = "düşük"
    if highlight > 3 or shadow > 4:
        technical_risk = "yüksek"
    elif highlight > 1 or shadow > 2:
        technical_risk = "orta"

    movement = _band_label(tension, 22, 55, "sakin", "ölçülü", "gerilimli")

    summary = (
        f"Bu kare teknik olarak {tone_balance} bir ton yapısı gösteriyor. "
        f"Detay yoğunluğu {detail_density}, kompozisyon hissi ise {composition_stability}. "
        f"Parlak alan taşması %{highlight:.2f}, koyu alan sıkışması ise %{shadow:.2f} seviyesinde; bu da teknik riski {technical_risk} tarafta tutuyor."
    )

    positives = []
    if negative >= 33:
        positives.append("Boşluk kullanımı ana vurguyu rahatlatıyor ve kadraja nefes veriyor.")
    if symmetry >= 60:
        positives.append("Yerleşim duygusu güçlü; kadraj dağılmadan okunuyor.")
    if highlight <= 0.8 and shadow <= 1.2:
        positives.append("Ton uçlarında belirgin patlama ya da ezilme yok; dosya temiz davranıyor.")
    if 30 <= contrast <= 60:
        positives.append("Kontrast yapısı kontrollü; sahne gereğinden sertleşmeden ayakta kalıyor.")
    if not positives:
        positives.append("Teknik yapı genel olarak fotoğrafın niyetini taşımaya yetiyor.")

    suggestions = []
    if edge < 0.03:
        suggestions.append("Ana vurgu çevresinde mikro kontrast hafif artırılırsa yüzey daha canlı hissedilir.")
    if negative < 20:
        suggestions.append("Kadraj biraz sadeleşirse ana görsel ağırlık daha rahat öne çıkar.")
    if symmetry < 35 and tension < 25:
        suggestions.append("Göz akışını güçlendirmek için ağırlık merkezi biraz daha net kurulabilir.")
    if highlight > 1.2:
        suggestions.append("Parlak bölgeler hafif geri çekilirse ton geçişleri daha rafine görünür.")
    if shadow > 1.8:
        suggestions.append("Koyu alanlarda küçük bir açma, detay kaybını azaltıp derinliği koruyabilir.")
    if not suggestions:
        suggestions.append("Mevcut teknik yapı korunup yalnızca küçük yerel dokunuşlarla rafine edilebilir.")

    cards = [
        ("Ton dengesi", tone_balance.capitalize(), f"Parlaklık {brightness:.1f} · Kontrast {contrast:.1f}"),
        ("Detay yoğunluğu", detail_density.capitalize(), f"Edge yoğunluğu {edge:.3f}"),
        ("Kompozisyon hissi", composition_stability.capitalize(), f"Simetri {symmetry:.1f} · Negatif alan {negative:.1f}"),
        ("Teknik risk", technical_risk.capitalize(), f"Highlight %{highlight:.2f} · Shadow %{shadow:.2f}"),
        ("Görsel hareket", movement.capitalize(), f"Dinamik gerilim {tension:.1f}"),
    ]

    raw_cards = [
        ("Boyut", f"{int(metrics.get('width', 0))} × {int(metrics.get('height', 0))}", "Dosya ölçüsü"),
        ("Parlaklık ort.", f"{brightness:.1f}", "Genel ton seviyesi"),
        ("Kontrast std", f"{contrast:.1f}", "Ton farkı"),
        ("Edge yoğunluğu", f"{edge:.3f}", "Detay / çizgi miktarı"),
        ("Highlight clip", f"%{highlight:.2f}", "Parlak alan kaybı"),
        ("Shadow clip", f"%{shadow:.2f}", "Koyu alan kaybı"),
        ("Simetri", f"{symmetry:.1f}", "Yerleşim dengesi"),
        ("Negatif alan", f"{negative:.1f}", "Boşluk kullanımı"),
        ("Dinamik gerilim", f"{tension:.1f}", "Görsel hareket"),
    ]

    return {
        "cards": cards,
        "summary": summary,
        "positives": positives[:3],
        "suggestions": suggestions[:3],
        "raw_cards": raw_cards,
    }


def render_technical_insights(metrics: Dict[str, float]) -> None:
    info = build_technical_readout(metrics)

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Teknik genel bakış</div>", unsafe_allow_html=True)
    top_cols = st.columns(len(info["cards"]), gap="small")
    for col, (title, value, caption) in zip(top_cols, info["cards"]):
        with col:
            render_compact_info_card(title, value, caption)
    st.markdown(f"<div class='mini-note' style='margin-top:.8rem; white-space: normal; line-height:1.8;'>{escape(info['summary'])}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        render_bullets("Teknik olarak çalışan taraflar", info["positives"], "✅")
    with right:
        render_bullets("Tek hamlede iyileştirme", info["suggestions"], "🛠️")




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
        st.caption(openai_error)
    if qwen_error:
        st.error(f"Qwen Vision Light hatası: {qwen_error}")

    st.markdown("</div>", unsafe_allow_html=True)



def build_text_report(result: CritiqueResult, ai_report: Dict, dynamic_shooting_notes: List[str], dynamic_editing_notes: List[str], dynamic_strengths: List[str], dynamic_first_reading: str, dynamic_structural_reading: str, dynamic_editorial_result: str) -> str:
    editor_comments = ai_report.get("editor_comments", {}) if isinstance(ai_report, dict) else {}
    cleaned_shooting = [str(x).strip() for x in (dynamic_shooting_notes or []) if str(x).strip()]
    cleaned_editing = [str(x).strip() for x in (dynamic_editing_notes or []) if str(x).strip()]

    lines = [
        "ÇOFSAT Fotoğraf Ön Değerlendirme",
        "=" * 34,
        f"Skor: {result.total_score:.1f}/100",
        f"Seviye: {result.overall_level}",
        f"Tür önerisi: {result.suggested_mode}",
        "",
        "Genel editör özeti:",
        str(result.editor_summary or "").strip(),
        "",
        "Ana güçlü taraf:",
        str(result.key_strength),
        "",
        "Ana sorun:",
        str(result.key_issue),
        "",
        "Tek hamlede iyileştirme:",
        str(result.one_move_improvement or "").strip(),
        "",
        "İlk okuma:",
        dynamic_first_reading,
        "",
        "Yapısal okuma:",
        dynamic_structural_reading,
        "",
        "Festival / seçki kararı:",
        dynamic_editorial_result,
    ]

    if cleaned_shooting:
        lines += ["", "Çekim notları:", *[f"- {x}" for x in cleaned_shooting]]

    if cleaned_editing:
        lines += ["", "Düzenleme notları:", *[f"- {x}" for x in cleaned_editing]]

    lines += ["", "Editör yorumları:"]
    for name in EDITOR_NAMES:
        comment = str(editor_comments.get(name) or "").strip()
        if comment:
            lines += [f"[{name}]", comment, ""]
    return "\n".join(lines).strip()


def build_general_editor_txt_report(
    result: CritiqueResult,
    dynamic_shooting_notes: List[str],
    dynamic_editing_notes: List[str],
    dynamic_strengths: List[str],
    dynamic_first_reading: str,
    dynamic_structural_reading: str,
    dynamic_editorial_result: str,
) -> str:
    cleaned_strengths = [str(x).strip() for x in (dynamic_strengths or []) if str(x).strip()]
    cleaned_shooting = [str(x).strip() for x in (dynamic_shooting_notes or []) if str(x).strip()]
    cleaned_editing = [str(x).strip() for x in (dynamic_editing_notes or []) if str(x).strip()]

    lines = [
        "ÇOFSAT Genel Editör Raporu",
        "=" * 28,
        "",
        "Genel yorum:",
        str(dynamic_first_reading or "").strip(),
        "",
        "Yapısal okuma:",
        str(dynamic_structural_reading or "").strip(),
        "",
        "Küratöryel sonuç:",
        str(dynamic_editorial_result or "").strip(),
    ]

    if cleaned_strengths:
        lines += ["", "Öne çıkan güçlü yönler:", *[f"- {x}" for x in cleaned_strengths]]

    lines += [
        "",
        "Ana güçlü taraf:",
        str(result.key_strength or "").strip(),
        "",
        "Ana sorun:",
        str(result.key_issue or "").strip(),
        "",
        "Tek hamlede iyileştirme:",
        str(result.one_move_improvement or "").strip(),
    ]

    if cleaned_shooting:
        lines += ["", "Çekim notları:", *[f"- {x}" for x in cleaned_shooting]]

    if cleaned_editing:
        lines += ["", "Düzenleme notları:", *[f"- {x}" for x in cleaned_editing]]

    return "\n".join(lines).strip()

def main() -> None:
    inject_css()


    with st.sidebar:
        selected_mode = st.selectbox("Tür", list(MODE_PROFILES.keys()), index=0)
        selected_editor_mode = st.selectbox("Ton", list(EDITOR_MODES.keys()), index=0)

    selected_editor_name = EDITOR_NAMES[0]
    sidebar_editor_placeholder = None

    render_sidebar(selected_mode, selected_editor_mode)

    hero_logo_url = get_brand_logo_data_url(max_size=200)
    st.markdown(
        f"""
        <div class="hero">
            <div style="display:flex;align-items:flex-start;gap:18px;flex-wrap:nowrap;">
                <img src="{hero_logo_url}" style="width:100px;height:auto;display:block;flex:0 0 auto;margin-top:28px;align-self:flex-start;" />
                <div style="display:flex;flex-direction:column;justify-content:flex-start;min-width:0;padding-top:0;">
                    <div class="hero-title-main">ÇOFSAT</div>
                    <div class="hero-title-sub">Fotoğraf Ön Değerlendirme</div>
                    <p style="margin-top:0.95rem;font-size:1.02rem;line-height:1.6;color:#EEE6DD;">
                        Fotoğrafı yalnızca göstermek için değil, okumak için ele alan;
                        odak, akış, anlatı ve görsel dengeyi tek bakışta okunur hale getiren premium değerlendirme deneyimi.
                    </p>
                    <div class="hero-badges">
                        <span class="hero-badge">Aktif tür: {selected_mode}</span>
                        <span class="hero-badge">Ton: {selected_editor_mode}</span>
                        <span class="ghost-badge">Isı Haritası + Göz Akışı + Altın Oran</span>
                    </div>
                </div>
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
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        st.session_state["_cofsat_last_uploaded_bytes"] = uploaded_file.getvalue()
        st.session_state["_cofsat_last_uploaded_name"] = uploaded_file.name
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:.75rem; margin:.55rem 0 .35rem 0;">
                <div style="width:14px;height:14px;border-radius:999px;background:#ff8c3a;box-shadow:0 0 0 rgba(255,140,58,.6);animation:pulseDot 1.2s infinite;"></div>
                <div style="padding:.75rem 1rem;border:1px solid rgba(255,255,255,.08);border-radius:999px;background:rgba(255,255,255,.03);font-weight:600;">Fotoğraf analiz ediliyor...</div>
            </div>
            <div class='mini-note' style='margin:.05rem 0 1rem 2.05rem;'>Hazır olduğunda analiz burada görünecek. 60 saniye civarı sürebilir.</div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("Yüklenen fotoğraflar analiz için otomatik optimize edilir; görünüm kalitesi korunurken analiz daha hızlı ve daha verimli çalışır.")

    image_bytes = st.session_state.get("_cofsat_last_uploaded_bytes")
    uploaded_name = st.session_state.get("_cofsat_last_uploaded_name") or "yüklenen_fotoğraf"

    if image_bytes is None:
        render_turkey_time_info(context="main")
        render_photo_of_day_candidates(context="main")
        return

    analysis_bytes = optimize_uploaded_bytes(image_bytes)
    analysis_key = _current_analysis_key(analysis_bytes, selected_mode, selected_editor_mode)
    if st.session_state.get("deep_ai_analysis_key") != analysis_key:
        st.session_state.setdefault("deep_ai_analysis_key", None)

    use_deep_ai = bool(openai_vision_available() or qwen_vision_runtime_available())
    st.markdown(
        f"<div class='mini-note' style='margin-top:.35rem;'>Yüklenen dosya: {human_file_size(len(image_bytes))} · Analiz sürümü: {human_file_size(len(analysis_bytes))}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        @keyframes pulseDot {0% {transform:scale(0.9); box-shadow:0 0 0 0 rgba(255,140,58,.55);} 70% {transform:scale(1); box-shadow:0 0 0 12px rgba(255,140,58,0);} 100% {transform:scale(0.9); box-shadow:0 0 0 0 rgba(255,140,58,0);} }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #1F140E !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #F5EEE4 !important;
            color: #1F140E !important;
            border: 1px solid rgba(31,20,14,.18) !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] input,
        section[data-testid="stSidebar"] [data-baseweb="select"] div,
        section[data-testid="stSidebar"] [data-baseweb="select"] span,
        section[data-testid="stSidebar"] [data-baseweb="select"] svg {
            color: #1F140E !important;
            fill: #1F140E !important;
            -webkit-text-fill-color:#1F140E !important;
            opacity: 1 !important;
            font-weight:700 !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
            color:#1F140E !important;
            opacity:1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("_cofsat_cached_analysis_key") != analysis_key:
        with st.spinner("Fotoğraf analiz ediliyor..."):
            image = get_resized_rgb(image_bytes)
            result = critique_image(analysis_bytes, selected_mode, selected_editor_mode)
            save_photo_result(image_bytes, uploaded_name, result)

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

            st.session_state["_cofsat_cached_analysis_key"] = analysis_key
            st.session_state["_cofsat_cached_analysis_bundle"] = {
                "image": image,
                "result": result,
                "attention": attention,
                "main_points": main_points,
                "distraction_points": distraction_points,
                "overlay_img": overlay_img,
                "heatmap_img": heatmap_img,
                "phi_grid_img": phi_grid_img,
                "diagonal_img": diagonal_img,
                "spiral_img": spiral_img,
                "best_scheme": best_scheme,
                "scheme_reason": scheme_reason,
                "ai_report": ai_report,
                "dynamic_shooting_notes": dynamic_shooting_notes,
                "dynamic_editing_notes": dynamic_editing_notes,
                "dynamic_strengths": dynamic_strengths,
                "dynamic_first_reading": dynamic_first_reading,
                "dynamic_structural_reading": dynamic_structural_reading,
                "dynamic_editorial_result": dynamic_editorial_result,
            }

    bundle = st.session_state.get("_cofsat_cached_analysis_bundle") or {}
    image = bundle.get("image")
    result = bundle.get("result")
    attention = bundle.get("attention")
    main_points = bundle.get("main_points")
    distraction_points = bundle.get("distraction_points")
    overlay_img = bundle.get("overlay_img")
    heatmap_img = bundle.get("heatmap_img")
    phi_grid_img = bundle.get("phi_grid_img")
    diagonal_img = bundle.get("diagonal_img")
    spiral_img = bundle.get("spiral_img")
    best_scheme = bundle.get("best_scheme")
    scheme_reason = bundle.get("scheme_reason")
    ai_report = bundle.get("ai_report")
    dynamic_shooting_notes = bundle.get("dynamic_shooting_notes")
    dynamic_editing_notes = bundle.get("dynamic_editing_notes")
    dynamic_strengths = bundle.get("dynamic_strengths")
    dynamic_first_reading = bundle.get("dynamic_first_reading")
    dynamic_structural_reading = bundle.get("dynamic_structural_reading")
    dynamic_editorial_result = bundle.get("dynamic_editorial_result")

    st.markdown("<div id='analysis-results'></div>", unsafe_allow_html=True)
    # Smooth-scroll helper removed for stability.

    top_left, top_right = st.columns([1.08, 0.92], gap="large")

    with top_left:
        st.image(image, caption=f"Yüklenen fotoğraf · {uploaded_name}", use_container_width=True)

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

    current_image_hash = hashlib.md5(image_bytes).hexdigest()
    current_entry = next((item for item in get_today_photo_entries() if item.get("image_hash") == current_image_hash), None)
    editor_scores = current_entry.get("editor_scores", {}) if isinstance(current_entry, dict) else {}
    if not isinstance(editor_scores, dict):
        editor_scores = {}

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Editörler bu kareye ne diyor?</div>", unsafe_allow_html=True)
    st.markdown("<div class='mini-note' style='margin-bottom:.7rem;'>Bir editör seçtiğinizde yorumu ve hemen altında yalnızca o editöre ait puan alanı görünür.</div>", unsafe_allow_html=True)
    _report_editor_comments = ai_report.get("editor_comments", {}) if isinstance(ai_report, dict) else {}
    selected_quick_editor = st.selectbox(
        "Editör seçin",
        EDITOR_NAMES,
        key=f"quick_editor_select_{current_image_hash}",
        label_visibility="collapsed",
    )
    _quick_comment = _report_editor_comments.get(selected_quick_editor) if isinstance(_report_editor_comments, dict) else None
    if not isinstance(_quick_comment, str) or not _quick_comment.strip():
        _quick_comment = build_editor_comment(selected_quick_editor, result)
    st.markdown(f"<div class='mini-note' style='white-space: normal; line-height:1.72; font-size:1rem;'>{escape(str(_quick_comment).strip())}</div>", unsafe_allow_html=True)
    render_single_editor_score_box(current_image_hash, selected_quick_editor, editor_scores, key_prefix=f"quick_{selected_quick_editor}")
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

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Fotoğraf Havuzu", "Görsel Katmanlar", "Editör Okuması", "Puanlama", "Altın Oran", "Teknik"]
    )

    with tab0:
        st.markdown("<div class='panel-title'>Fotoğraf Havuzu</div>", unsafe_allow_html=True)
        st.caption("Havuz burada görünür. Silme ve temizleme işlemleri üstteki ana havuz panelinden yapılır.")
        render_photo_of_day_candidates(context="analysis_results")
        st.markdown("<div style='height:.35rem;'></div>", unsafe_allow_html=True)

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
            st.markdown("<div class='section-title'>Genel yorum</div>", unsafe_allow_html=True)
            st.write(dynamic_first_reading)
            st.markdown("<div class='section-title'>Yapısal okuma</div>", unsafe_allow_html=True)
            st.write(dynamic_structural_reading)
            st.markdown("<div class='section-title'>Küratöryel sonuç</div>", unsafe_allow_html=True)
            st.write(dynamic_editorial_result)
            general_editor_report_text = build_general_editor_txt_report(
                result,
                dynamic_shooting_notes,
                dynamic_editing_notes,
                dynamic_strengths,
                dynamic_first_reading,
                dynamic_structural_reading,
                dynamic_editorial_result,
            )
            st.download_button(
                "Genel editör yorumunu indir (TXT)",
                data=general_editor_report_text.encode("utf-8"),
                file_name="cofsat_genel_editor_yorumu.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_general_editor_report",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        render_editor_comments(result, selected_editor_name, current_image_hash, editor_scores, ai_report)

        c3, c4 = st.columns(2, gap="large")
        with c3:
            render_bullets("Çekim notları", dynamic_shooting_notes, "📷")
        with c4:
            render_bullets("Düzenleme notları", dynamic_editing_notes, "🎛️")

        render_bullets("Kendine sor", result.reading_prompts, "❓")

    with tab3:
        left, right = st.columns([0.58, 0.42], gap="large")
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

        st.info("Editör puanları kendi yorumlamalarının altında yer almaktadır.")

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
        render_technical_insights(metrics)

    render_turkey_time_info(context="main")
    render_photo_of_day_candidates(context="main")


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
    """Editör yorumları ve genel özetten daha bağlamsal çekim / düzenleme notları türetir."""
    base_shooting = list(result.shooting_notes or [])
    base_editing = list(result.editing_notes or [])
    if not isinstance(ai_report, dict) or not ai_report:
        return base_shooting[:3], base_editing[:3]

    source = str(ai_report.get("_source") or "")
    if source not in {"vision", "qwen"}:
        return base_shooting[:3], base_editing[:3]

    profile_data = (result.metrics or {}).get("scene_profile", {})
    profile = SceneProfile(**profile_data) if isinstance(profile_data, dict) and profile_data else None
    if profile is None:
        return base_shooting[:3], base_editing[:3]

    comments = ai_report.get("editor_comments") if isinstance(ai_report.get("editor_comments"), dict) else {}
    corpus = _normalized_comment_corpus(ai_report)
    if not corpus.strip():
        return base_shooting[:3], base_editing[:3]

    summary_text = " ".join([
        str(ai_report.get("global_summary") or ""),
        str(result.key_strength or ""),
        str(result.key_issue or ""),
        str(result.one_move_improvement or ""),
        corpus,
    ]).lower()

    def has_any(*terms: str) -> bool:
        return any(term in summary_text for term in terms)

    def find_region_phrase() -> str:
        regions = [
            profile.primary_region, profile.secondary_region, profile.distraction_region,
            "alt merkez", "üst merkez", "sol", "sağ", "sol merkez", "sağ merkez",
            "sağ üst", "sol üst", "sağ alt", "sol alt", "merkez"
        ]
        for region in regions:
            reg = (region or "").lower().strip()
            if reg and reg in summary_text:
                return reg
        return str(profile.primary_region or "merkez").lower()

    region = find_region_phrase()
    subject_hint = str(profile.subject_hint or "ana özne")
    mood = str(profile.visual_mood or "atmosfer")

    shooting: List[str] = []
    editing: List[str] = []

    if has_any("katman", "derinlik", "ön plan", "arka plan", "ayrım", "hacim", "nefes"):
        _append_unique_note(
            shooting,
            tone_text(
                f"Çekimde {subject_hint} ile arka plan arasındaki mesafeyi biraz açmak ya da açı değiştirerek katman ayrımını belirginleştirmek karenin hacmini güçlendirir.",
                editor_mode,
            ),
        )
        _append_unique_note(
            editing,
            tone_text(
                f"Düzenlemede {subject_hint} çevresinde hafif lokal kontrast toplayıp arka planı biraz sakinleştirmek derinlik hissini daha okunur kılar.",
                editor_mode,
            ),
        )

    if has_any("kadraj", "denge", "ağırlık", "yük", "yerleşim", "göz akışı", "merkez", "sol", "sağ", "üst", "alt"):
        _append_unique_note(
            shooting,
            tone_text(
                f"Çekim anında {region} tarafında biriken görsel yükü küçük bir kadraj kaydırmasıyla dengelemek ana cümleyi daha kararlı kurar.",
                editor_mode,
            ),
        )
        _append_unique_note(
            editing,
            tone_text(
                f"{region.capitalize()} tarafındaki ikincil ağırlığı ton olarak biraz geri çekmek bakışın ana vurguya daha temiz tutunmasını sağlar.",
                editor_mode,
            ),
        )

    if has_any("figür", "insan", "yüz", "ifade", "jest", "beden", "bakış") or profile.face_count >= 1:
        _append_unique_note(
            shooting,
            tone_text(
                f"Figürün jesti ya da ifadesi tam oturduğu anı biraz daha sabırla beklemek, fotoğrafın insani çekirdeğini daha güçlü hissettirir.",
                editor_mode,
            ),
        )
        _append_unique_note(
            editing,
            tone_text(
                f"{subject_hint.capitalize()} çevresindeki dikkat dağıtan küçük enerjileri seçici olarak yumuşatmak, figürün sahne içindeki varlığını daha belirgin kılar.",
                editor_mode,
            ),
        )

    if has_any("ışık", "parlak", "gölge", "ton", "kontrast", "hava", "atmosfer", "sis"):
        _append_unique_note(
            shooting,
            tone_text(
                f"Işığın {subject_hint} üzerine daha temiz oturduğu anı kollamak ya da yarım adım açı değiştirerek parlamayı kırmak {mood} havayı güçlendirir.",
                editor_mode,
            ),
        )
        _append_unique_note(
            editing,
            tone_text(
                f"Parlak bölgeleri hafifçe bastırıp gölge bilgisini kontrollü açmak, fotoğrafın kurduğu {mood} tonu daha rafine hale getirir.",
                editor_mode,
            ),
        )

    if has_any("crop", "kırp", "seçki", "toparla", "sade", "gereksiz", "ayıkl"):
        _append_unique_note(
            editing,
            tone_text(
                f"Hafif bir crop ile {region} tarafında kalan gereksiz alanı ayıklamak, fotoğrafın kararını daha ikna edici hale getirir.",
                editor_mode,
            ),
        )

    if has_any("niyet", "hikâye", "anlatı", "cümle", "asıl"):
        _append_unique_note(
            shooting,
            tone_text(
                "Deklanşöre basmadan önce sahnenin asıl hikâyesini taşıyan ilişkiyi bir an daha seçmek, fotoğrafın niyetini berraklaştırır.",
                editor_mode,
            ),
        )

    if not shooting:
        shooting = base_shooting[:3]
    if not editing:
        editing = base_editing[:3]

    filtered_editing: List[str] = []
    shoot_roots = {s.split(",")[0].strip() for s in shooting}
    for note in editing:
        root = note.split(",")[0].strip()
        if note not in shooting and root not in shoot_roots:
            filtered_editing.append(note)
    editing = filtered_editing or [n for n in base_editing[:3] if n not in shooting] or base_editing[:3]

    return shooting[:3], editing[:3]

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
        'Ilkay Strebel-Ozmen': 'Ben burada görüntünün düşünsel omurgasını biraz daha berraklaştırmaktan yanayım.',
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
