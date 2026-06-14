"""
FIFA Match Predictor — Streamlit app.
Entry point: streamlit run app.py
"""

import os
import json
from html import escape

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="World Cup 2026 Match Lab",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from data.ingest import load_all, get_all_teams, count_loaded
from data.world_cup_2026 import get_non_world_cup_2026_squad, is_qualified_team
from model.features import FEATURE_COLUMNS

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "artifacts")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
UI_BUILD = "2026.06.11-r9"


def format_probability_shift(value: float) -> str:
    """Format a probability shift without presenting rounded zero as positive."""
    percentage_points = float(value) * 100
    if abs(percentage_points) < 0.05:
        return "0.0%"
    return f"{percentage_points:+.1f}%"


def format_probability_point_note(value: float) -> str:
    percentage_points = float(value) * 100
    if abs(percentage_points) < 0.05:
        return "0.0 percentage points"
    return f"{percentage_points:+.1f} percentage points"


def shift_status(value: float, positive_text: str, negative_text: str) -> str:
    percentage_points = float(value) * 100
    if abs(percentage_points) < 0.05:
        return "No material change"
    return positive_text if percentage_points > 0 else negative_text


def inject_tournament_theme():
    """FIFA World Cup 2026 visual system."""
    st.markdown(
        """
        <style>
        :root {
            --bg:        #04090f;
            --card:      #090f1d;
            --raised:    #0d1828;
            --border:    rgba(255,255,255,.08);
            --border-md: rgba(255,255,255,.14);
            --border-hi: rgba(255,255,255,.22);
            --text:      #edf2f7;
            --muted:     #7a96b0;
            --dim:       #334f68;
            --gold:      #f2c237;
            --gold-dim:  rgba(242,194,55,.14);
            --blue:      #1a6aff;
            --blue-dim:  rgba(26,106,255,.14);
            --cyan:      #14d4f4;
            --cyan-dim:  rgba(20,212,244,.12);
            --green:     #3de89e;
            --lime:      #b2f229;
            --red:       #ff3554;
        }

        html { scroll-behavior: smooth; }

        .stApp {
            background:
                radial-gradient(ellipse 160% 55% at 50% -5%, rgba(26,106,255,.2) 0%, transparent 55%),
                linear-gradient(180deg, #04090f 0%, #060c18 55%, #030810 100%);
            color: var(--text);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .15;
            background-image:
                linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px);
            background-size: 54px 54px;
            mask-image: linear-gradient(to bottom, black 0%, transparent 60%);
        }

        [data-testid="stHeader"] {
            background: rgba(4,9,15,.88);
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(24px);
        }

        [data-testid="stToolbar"] { right: 1.25rem; }

        .block-container {
            max-width: 1280px;
            padding-top: 3rem;
            padding-bottom: 6rem;
        }

        h1, h2, h3 {
            font-family: "Helvetica Neue", Arial, sans-serif;
            color: var(--text) !important;
        }

        h2 {
            font-size: clamp(1.5rem, 2.8vw, 2rem) !important;
            font-weight: 800 !important;
            letter-spacing: -.03em;
            margin-top: .4rem !important;
            padding-bottom: .5rem !important;
            border-bottom: 1px solid var(--border) !important;
        }

        h2::before {
            content: "";
            display: inline-block;
            width: .34rem;
            height: 1.3rem;
            margin-right: .6rem;
            vertical-align: -.12rem;
            border-radius: 99px;
            background: linear-gradient(var(--cyan), var(--blue));
        }

        p, label, li, button, input, [data-baseweb="select"] {
            font-family: "Helvetica Neue", "Segoe UI", sans-serif;
        }

        [data-testid="stCaptionContainer"] { color: var(--muted); }
        hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

        /* ── HERO ──────────────────────────────────────────── */

        .wc-hero {
            position: relative;
            overflow: hidden;
            min-height: 290px;
            padding: clamp(2.5rem,5vw,4rem) clamp(2rem,4vw,3.5rem);
            border: 1px solid var(--border-md);
            border-radius: 24px;
            background: linear-gradient(135deg, #040a11 0%, #071626 52%, #050f1e 100%);
            box-shadow: 0 40px 100px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.05);
        }

        .wc-hero::after {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: linear-gradient(90deg,
                var(--blue) 0%, var(--cyan) 25%,
                var(--lime) 50%, var(--gold) 75%, var(--red) 100%);
        }

        .wc-hero::before {
            content: "26";
            position: absolute;
            right: -0.04em;
            bottom: -.28em;
            font-family: "Helvetica Neue", "Arial Black", sans-serif;
            font-size: clamp(14rem,28vw,27rem);
            font-weight: 900;
            letter-spacing: -.1em;
            line-height: 1;
            color: transparent;
            -webkit-text-stroke: 1.5px rgba(255,255,255,.05);
            transform: skewX(-5deg);
            pointer-events: none;
            user-select: none;
        }

        .wc-build-badge {
            position: absolute;
            z-index: 2;
            top: 1.4rem;
            right: 1.4rem;
            padding: .38rem .72rem;
            border: 1px solid var(--border-md);
            border-radius: 999px;
            background: rgba(4,9,15,.9);
            color: var(--muted);
            font-size: .6rem;
            font-weight: 700;
            letter-spacing: .1em;
            text-transform: uppercase;
            backdrop-filter: blur(12px);
        }

        .wc-kicker {
            position: relative;
            z-index: 1;
            display: inline-flex;
            align-items: center;
            gap: .6rem;
            margin-bottom: .7rem;
            color: var(--gold);
            font-size: .68rem;
            font-weight: 900;
            letter-spacing: .22em;
            text-transform: uppercase;
        }

        .wc-kicker::before {
            content: "";
            width: 1.5rem;
            height: 2px;
            background: var(--gold);
            border-radius: 2px;
        }

        .wc-hero h1 {
            position: relative;
            z-index: 1;
            max-width: 820px;
            margin: 0 0 .85rem;
            color: #fff;
            font-family: "Helvetica Neue", "Arial Black", sans-serif;
            font-size: clamp(3rem, 7vw, 6.8rem);
            font-weight: 900;
            line-height: .87;
            letter-spacing: -.055em;
            text-transform: uppercase;
        }

        .wc-hero h1 span { color: var(--cyan); }

        .wc-hero-copy {
            position: relative;
            z-index: 1;
            max-width: 600px;
            color: var(--muted);
            font-size: clamp(.9rem,1.6vw,1.05rem);
            line-height: 1.7;
        }

        /* ── STATUS GRID ───────────────────────────────────── */

        .wc-status-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            margin: 1.2rem 0 2.5rem;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: var(--border);
        }

        .wc-status-item {
            padding: 1.1rem 1.25rem;
            background: var(--card);
        }

        .wc-status-label {
            display: block;
            margin-bottom: .3rem;
            color: var(--muted);
            font-size: .61rem;
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
        }

        .wc-status-value {
            color: #fff;
            font-size: .97rem;
            font-weight: 700;
        }

        .wc-live-dot {
            display: inline-block;
            width: .46rem;
            height: .46rem;
            margin-right: .36rem;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 0 4px rgba(61,232,158,.12);
            vertical-align: .05rem;
        }

        /* ── SECTION LABELS ────────────────────────────────── */

        .wc-section-kicker {
            margin: 0 0 .2rem;
            color: var(--cyan);
            font-size: .63rem;
            font-weight: 900;
            letter-spacing: .2em;
            text-transform: uppercase;
        }

        .wc-section-title {
            margin: 0 0 1rem;
            color: #fff;
            font-family: "Helvetica Neue", "Arial Black", sans-serif;
            font-size: clamp(1.8rem, 4vw, 3rem);
            font-weight: 900;
            letter-spacing: -.04em;
            line-height: .95;
            text-transform: uppercase;
        }

        /* ── SELECTOR PANEL ────────────────────────────────── */

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--border-md) !important;
            border-radius: 20px !important;
            background: var(--card) !important;
            box-shadow: 0 8px 32px rgba(0,0,0,.25);
        }

        .selector-note {
            margin: -.2rem 0 1rem;
            color: var(--muted);
            font-size: .7rem;
            line-height: 1.5;
        }

        [data-testid="stSelectbox"] label,
        [data-testid="stRadio"] > label {
            color: var(--muted) !important;
            font-size: .64rem !important;
            font-weight: 800 !important;
            letter-spacing: .12em !important;
            text-transform: uppercase;
        }

        [data-baseweb="select"] > div {
            min-height: 3.6rem;
            padding: 0 .35rem;
            color: #fff !important;
            border: 1px solid var(--border-md) !important;
            border-radius: 12px;
            background: var(--raised) !important;
            box-shadow: none;
            transition: border-color .15s;
        }

        [data-baseweb="select"] > div:hover {
            border-color: var(--cyan) !important;
        }

        [data-baseweb="select"] input,
        [data-baseweb="select"] [data-testid="stMarkdownContainer"],
        [data-baseweb="select"] > div > div {
            color: #fff !important;
            font-size: .97rem !important;
            font-weight: 700 !important;
        }

        [data-baseweb="select"] svg { fill: var(--cyan) !important; }

        [data-baseweb="popover"] { z-index: 10000 !important; }

        [data-baseweb="popover"] > div {
            overflow: hidden !important;
            border: 1px solid rgba(20,212,244,.28) !important;
            border-radius: 14px !important;
            background: #090f1d !important;
            box-shadow: 0 24px 64px rgba(0,0,0,.6) !important;
        }

        [data-baseweb="popover"] [data-baseweb="menu"],
        ul[role="listbox"], [role="listbox"] {
            max-height: min(22rem, 58vh) !important;
            padding: .35rem !important;
            background: #090f1d !important;
        }

        li[role="option"], [role="option"] {
            min-height: 2.65rem !important;
            margin: .1rem 0 !important;
            padding: .6rem .8rem !important;
            border-radius: 8px !important;
            background: transparent !important;
            color: #ccd9e5 !important;
            font-size: .9rem !important;
            font-weight: 600 !important;
            transition: background .1s;
        }

        li[role="option"]:hover,
        li[role="option"][aria-selected="true"],
        [role="option"]:hover,
        [role="option"][aria-selected="true"] {
            background: var(--cyan-dim) !important;
            color: #fff !important;
        }

        [role="option"][aria-selected="true"] {
            box-shadow: inset 3px 0 0 var(--cyan);
        }

        [data-testid="stRadio"] [role="radiogroup"] { gap: .5rem; }

        [data-testid="stRadio"] [role="radiogroup"] label {
            min-height: 2.7rem;
            padding: .5rem .9rem;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: transparent;
            color: var(--muted) !important;
            font-size: .75rem !important;
            transition: border-color .15s, color .15s;
        }

        .stButton > button {
            min-height: 3.4rem;
            border: 0;
            border-radius: 11px;
            background: linear-gradient(90deg, var(--blue), var(--cyan));
            box-shadow: 0 8px 24px rgba(26,106,255,.3);
            color: #fff;
            font-size: .8rem;
            font-weight: 900;
            letter-spacing: .14em;
            text-transform: uppercase;
            transition: filter .2s, transform .15s, box-shadow .2s;
        }

        .stButton > button:hover {
            border: 0;
            color: #fff;
            filter: brightness(1.1);
            transform: translateY(-2px);
            box-shadow: 0 14px 36px rgba(26,106,255,.4);
        }

        .action-spacer { height: 1.5rem; }

        /* ── MATCH FIXTURE CARD ────────────────────────────── */

        .match-board {
            position: relative;
            overflow: hidden;
            padding: clamp(2rem,4vw,3.2rem) clamp(1.5rem,4vw,3rem);
            border: 1px solid var(--border-md);
            border-radius: 24px;
            background: linear-gradient(140deg, #050c19 0%, #0a1930 55%, #060e1e 100%);
            box-shadow: 0 28px 72px rgba(0,0,0,.4);
        }

        .match-board::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: linear-gradient(90deg, var(--blue) 0%, var(--cyan) 50%, var(--gold) 100%);
        }

        .match-board::after {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            opacity: .03;
            background-image: repeating-linear-gradient(
                -45deg, transparent 0, transparent 22px,
                rgba(255,255,255,1) 22px, rgba(255,255,255,1) 23px
            );
        }

        .match-meta {
            position: relative;
            z-index: 1;
            margin-bottom: 2rem;
            color: var(--muted);
            font-size: .65rem;
            font-weight: 800;
            letter-spacing: .2em;
            text-align: center;
            text-transform: uppercase;
        }

        .match-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
            align-items: center;
            gap: 1.5rem;
        }

        .team-card { min-width: 0; text-align: center; }

        .team-name {
            overflow-wrap: anywhere;
            color: #fff;
            font-family: "Helvetica Neue", "Arial Black", sans-serif;
            font-size: clamp(2rem, 5vw, 5rem);
            font-weight: 900;
            letter-spacing: -.045em;
            line-height: .88;
            text-transform: uppercase;
        }

        .team-rank {
            display: inline-block;
            margin-top: .85rem;
            padding: .3rem .7rem;
            border: 1px solid var(--border-md);
            border-radius: 999px;
            color: var(--muted);
            font-size: .66rem;
            font-weight: 700;
            letter-spacing: .06em;
            text-transform: uppercase;
        }

        .ranking-source-note {
            position: relative;
            z-index: 1;
            margin-top: 1.4rem;
            color: var(--dim);
            font-size: .6rem;
            font-weight: 600;
            letter-spacing: .07em;
            text-align: center;
            text-transform: uppercase;
        }

        .versus {
            display: grid;
            width: 4rem;
            height: 4rem;
            place-items: center;
            border: 1.5px solid var(--border-md);
            border-radius: 50%;
            background: rgba(4,9,15,.75);
            color: var(--gold);
            font-family: "Helvetica Neue", sans-serif;
            font-size: .95rem;
            font-weight: 900;
            letter-spacing: .06em;
            backdrop-filter: blur(8px);
        }

        /* ── PROBABILITY BAR ──────────────────────────────── */

        .probability-wrap { margin: 1.4rem 0 0; }

        .probability-labels {
            display: flex;
            justify-content: space-between;
            margin-bottom: .55rem;
            color: var(--muted);
            font-size: .65rem;
            font-weight: 800;
            letter-spacing: .1em;
            text-transform: uppercase;
        }

        .probability-bar {
            display: flex;
            width: 100%;
            height: 50px;
            overflow: hidden;
            border-radius: 999px;
            background: var(--raised);
            box-shadow: inset 0 2px 8px rgba(0,0,0,.35);
            gap: 2px;
        }

        .probability-bar > div {
            display: grid;
            min-width: 0;
            place-items: center;
            overflow: hidden;
            color: #fff;
            font-size: .74rem;
            font-weight: 900;
            letter-spacing: .02em;
            white-space: nowrap;
        }

        .prob-a    { background: linear-gradient(90deg, #0f55e8, #1a6aff); border-radius: 999px 0 0 999px; }
        .prob-draw { background: #253d52; }
        .prob-b    { background: linear-gradient(90deg, #e82848, var(--red)); border-radius: 0 999px 999px 0; }

        /* ── REPORT SECTIONS ──────────────────────────────── */

        .report-section {
            margin: 1.4rem 0;
            padding: clamp(1.4rem, 2.6vw, 2rem);
            border: 1px solid var(--border);
            border-radius: 20px;
            background: var(--card);
            box-shadow: 0 10px 30px rgba(0,0,0,.12);
        }

        .report-heading {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.3rem;
            padding-bottom: .9rem;
            border-bottom: 1px solid var(--border);
        }

        .report-eyebrow {
            margin-bottom: .25rem;
            color: var(--cyan);
            font-size: .59rem;
            font-weight: 900;
            letter-spacing: .22em;
            text-transform: uppercase;
        }

        .report-title {
            color: #fff;
            font-family: "Helvetica Neue", "Arial Black", sans-serif;
            font-size: clamp(1.5rem, 2.8vw, 2.15rem);
            font-weight: 900;
            letter-spacing: -.03em;
            line-height: .95;
            text-transform: uppercase;
        }

        .report-note {
            max-width: 42rem;
            color: var(--muted);
            font-size: .74rem;
            line-height: 1.55;
            text-align: right;
        }

        /* ── STAT GRID ─────────────────────────────────────── */

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .9rem;
        }

        .stat-grid.two  { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .stat-grid.five { grid-template-columns: repeat(5, minmax(0, 1fr)); }

        .stat-tile {
            min-width: 0;
            padding: 1.2rem 1.15rem;
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(255,255,255,.03);
        }

        .stat-tile.accent {
            border-color: rgba(26,106,255,.3);
            background: rgba(26,106,255,.1);
        }

        .stat-label {
            display: block;
            min-height: 1.9em;
            margin-bottom: .5rem;
            color: var(--muted);
            font-size: .63rem;
            font-weight: 800;
            letter-spacing: .1em;
            line-height: 1.35;
            text-transform: uppercase;
        }

        .stat-value {
            display: block;
            color: #fff;
            font-family: "Helvetica Neue", "Arial Black", sans-serif;
            font-size: clamp(1.9rem, 3.5vw, 2.9rem);
            font-weight: 900;
            letter-spacing: -.03em;
            line-height: 1;
        }

        .stat-detail {
            display: block;
            margin-top: .5rem;
            color: var(--muted);
            font-size: .7rem;
            line-height: 1.4;
        }

        .shift-status {
            display: inline-flex;
            width: fit-content;
            margin-top: .7rem;
            padding: .24rem .5rem;
            border: 1px solid var(--border);
            border-radius: 999px;
            color: var(--muted);
            background: rgba(255,255,255,.04);
            font-size: .59rem;
            font-weight: 800;
            letter-spacing: .1em;
            text-transform: uppercase;
        }

        /* ── OUTCOME BANNER ───────────────────────────────── */

        .outcome-banner {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.25rem;
            border: 1px solid rgba(61,232,158,.2);
            border-radius: 14px;
            background: linear-gradient(90deg, rgba(61,232,158,.09), rgba(10,25,50,.8));
        }

        .outcome-icon { font-size: 1.3rem; }
        .outcome-label {
            color: var(--green);
            font-size: .6rem;
            font-weight: 900;
            letter-spacing: .15em;
            text-transform: uppercase;
        }
        .outcome-value {
            color: #fff;
            font-size: 1.05rem;
            font-weight: 800;
            margin-top: .12rem;
        }
        .outcome-policy { color: var(--muted); font-size: .7rem; text-align: right; }

        /* ── CONFIDENCE ───────────────────────────────────── */

        .analysis-grid {
            display: grid;
            grid-template-columns: minmax(0, .75fr) minmax(0, 1.25fr);
            gap: 1rem;
        }

        .confidence-score {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100%;
            padding: 1.8rem 1.4rem;
            border: 1px solid rgba(178,242,41,.2);
            border-radius: 16px;
            background:
                radial-gradient(circle at 50% 40%, rgba(178,242,41,.1) 0%, transparent 65%),
                rgba(255,255,255,.025);
            text-align: center;
            gap: .35rem;
        }

        .confidence-score strong {
            display: block;
            color: #fff;
            font-family: "Helvetica Neue", "Arial Black", sans-serif;
            font-size: clamp(3.4rem, 7vw, 5.5rem);
            font-weight: 900;
            line-height: 1;
            letter-spacing: -.04em;
        }

        .confidence-score span {
            display: block;
            color: var(--lime);
            font-size: .65rem;
            font-weight: 900;
            letter-spacing: .17em;
            text-transform: uppercase;
        }

        .confidence-score em {
            display: block;
            color: var(--muted);
            font-size: .68rem;
            font-style: normal;
            line-height: 1.45;
        }

        /* ── KEY FACTORS ──────────────────────────────────── */

        .factor-list {
            display: grid;
            gap: .58rem;
            margin: 0;
            padding: 0;
            list-style: none;
        }

        .factor-list li {
            display: grid;
            grid-template-columns: 1.6rem 1fr;
            align-items: start;
            gap: .65rem;
            padding: .75rem .9rem;
            border: 1px solid var(--border);
            border-radius: 11px;
            background: rgba(255,255,255,.025);
            color: #cddce8;
            font-size: .82rem;
            line-height: 1.5;
        }

        .factor-index {
            display: grid;
            width: 1.6rem;
            height: 1.6rem;
            place-items: center;
            border-radius: 50%;
            background: var(--cyan-dim);
            color: var(--cyan);
            font-size: .59rem;
            font-weight: 900;
        }

        /* ── COMPARISON BOARD ─────────────────────────────── */

        .comparison-board {
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 14px;
        }

        .comparison-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(9rem, .5fr) minmax(0, 1fr);
            align-items: center;
            min-height: 4rem;
            border-bottom: 1px solid var(--border);
        }

        .comparison-row:last-child { border-bottom: 0; }
        .comparison-row.header { min-height: 3rem; background: rgba(26,106,255,.12); }

        .comparison-cell {
            min-width: 0;
            padding: .8rem 1rem;
            color: var(--text);
            font-size: .85rem;
            font-weight: 700;
        }

        .comparison-cell:nth-child(2) {
            border-right: 1px solid var(--border);
            border-left: 1px solid var(--border);
            color: var(--muted);
            font-size: .62rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-align: center;
            text-transform: uppercase;
        }

        .comparison-cell:last-child { text-align: right; }
        .comparison-row.header .comparison-cell {
            color: #fff;
            font-size: .7rem;
            letter-spacing: .06em;
            text-transform: uppercase;
        }
        .comparison-strong { color: var(--lime); }

        /* ── VERDICT CARD ─────────────────────────────────── */

        .verdict-card {
            padding: 1.25rem 1.4rem;
            border-left: 3px solid var(--cyan);
            border-radius: 4px 14px 14px 4px;
            background: rgba(20,212,244,.06);
            color: #ddeaf5;
            font-size: .95rem;
            line-height: 1.7;
        }

        /* ── PLAYER CARDS ─────────────────────────────────── */

        .player-columns {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1.1rem;
        }

        .player-team {
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: rgba(255,255,255,.02);
        }

        .player-team-title {
            padding: .9rem 1.15rem;
            background: rgba(26,106,255,.15);
            border-bottom: 1px solid rgba(26,106,255,.2);
            color: #fff;
            font-size: .74rem;
            font-weight: 900;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .player-card {
            padding: 1rem 1.15rem;
            border-top: 1px solid var(--border);
        }

        .player-name {
            color: #fff;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.25;
            letter-spacing: -.01em;
        }

        .player-pos-badge {
            display: inline-block;
            margin-left: .45rem;
            padding: .1rem .44rem;
            border-radius: 999px;
            background: var(--cyan-dim);
            color: var(--cyan);
            font-size: .59rem;
            font-weight: 900;
            letter-spacing: .08em;
            vertical-align: middle;
        }

        .player-meta {
            margin-top: .35rem;
            color: var(--muted);
            font-size: .74rem;
            line-height: 1.55;
        }

        .player-impact {
            display: inline-block;
            margin-top: .38rem;
            padding: .14rem .48rem;
            border: 1px solid var(--border);
            border-radius: 999px;
            color: #c6d6e3;
            background: rgba(255,255,255,.04);
            font-size: .59rem;
            font-weight: 800;
            letter-spacing: .07em;
        }

        .player-achievement {
            margin-top: .4rem;
            color: #7eaac6;
            font-size: .72rem;
            line-height: 1.5;
        }

        /* ── STREAMLIT OVERRIDES ──────────────────────────── */

        [data-testid="stMetric"] {
            min-height: 110px;
            padding: 1.1rem 1.2rem;
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--card);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: .64rem;
            font-weight: 800;
            letter-spacing: .09em;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: #fff;
            font-family: "Helvetica Neue", "Arial Black", sans-serif;
            font-size: clamp(1.8rem, 3.5vw, 2.8rem);
            font-weight: 900;
            letter-spacing: -.03em;
        }

        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] svg {
            color: var(--muted) !important;
            fill: var(--muted) !important;
        }

        [data-testid="stAlert"] {
            border: 1px solid var(--border-md);
            border-radius: 12px;
            background: var(--card);
        }

        [data-testid="stAlert"] p,
        [data-testid="stAlert"] strong { color: var(--text) !important; }

        [data-testid="stMarkdownContainer"] > p,
        [data-testid="stMarkdownContainer"] > ul,
        [data-testid="stMarkdownContainer"] > ol { color: #cddce8; }

        [data-testid="stExpander"] {
            overflow: hidden;
            border-color: var(--border);
            border-radius: 14px;
            background: rgba(255,255,255,.025);
        }

        [data-testid="stExpander"] summary {
            min-height: 3.6rem;
            font-weight: 800;
        }

        table {
            overflow: hidden;
            border: 1px solid var(--border) !important;
            border-radius: 12px;
            background: rgba(4,20,36,.5);
        }

        table th {
            color: #fff !important;
            background: rgba(26,106,255,.14) !important;
        }

        table td, table th { border-color: var(--border) !important; }

        code {
            color: var(--lime) !important;
            background: rgba(178,242,41,.07) !important;
        }

        /* ── FOOTER ───────────────────────────────────────── */

        .wc-footer {
            margin-top: 4rem;
            padding-top: 1.2rem;
            border-top: 1px solid var(--border);
            color: var(--dim);
            font-size: .66rem;
            letter-spacing: .1em;
            text-align: center;
            text-transform: uppercase;
        }

        /* ── RESPONSIVE ───────────────────────────────────── */

        @media (max-width: 760px) {
            [data-testid="stHeader"] {
                height: 3.25rem;
                background: rgba(4,9,15,.96);
            }

            [data-testid="stToolbar"] { top: .35rem; right: .4rem; }

            .block-container { padding: 4rem .75rem 3rem; }

            .wc-hero {
                min-height: 0;
                padding: 1.5rem 1.25rem 2rem;
                border-radius: 18px;
            }

            .wc-build-badge {
                position: static;
                display: inline-flex;
                margin: .2rem 0 1rem;
                padding: .36rem .6rem;
                font-size: .55rem;
            }

            .wc-kicker { font-size: .58rem; letter-spacing: .14em; }

            .wc-hero h1 {
                font-size: clamp(2.6rem, 13vw, 4rem);
                line-height: .9;
            }

            .wc-hero h1 span { display: block; }
            .wc-hero-copy { font-size: .9rem; }

            .wc-status-grid {
                grid-template-columns: 1fr 1fr;
                margin-bottom: 1.8rem;
                border-radius: 14px;
            }

            .wc-section-title { font-size: 1.8rem; }

            [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px !important; }

            [data-testid="stRadio"] [role="radiogroup"] { flex-wrap: wrap; }

            [data-baseweb="select"] > div { min-height: 3.4rem; }

            [data-baseweb="popover"] > div { max-width: calc(100vw - 1.5rem) !important; }

            [role="listbox"] { max-height: 19rem !important; }
            [role="option"] { min-height: 2.9rem !important; font-size: .92rem !important; }

            .action-spacer { display: none; }

            .match-board { padding: 1.4rem 1rem; border-radius: 18px; }
            .match-meta { font-size: .58rem; letter-spacing: .12em; margin-bottom: 1.3rem; }
            .match-grid { gap: .6rem; }
            .team-name { font-size: clamp(1.3rem, 8vw, 2.2rem); }
            .versus { width: 2.9rem; height: 2.9rem; font-size: .78rem; }
            .team-rank { font-size: .58rem; }

            .probability-labels { font-size: .56rem; }
            .probability-bar { height: 42px; }
            .probability-bar > div { font-size: .6rem; }

            [data-testid="stMetric"] { min-height: 95px; padding: .9rem; }
            [data-testid="stMetricValue"] { font-size: 1.8rem; }

            h2 { font-size: 1.45rem !important; }
            table { font-size: .72rem !important; }

            .report-section { padding: 1rem; border-radius: 16px; }
            .report-heading { display: block; margin-bottom: .9rem; }
            .report-note { margin-top: .45rem; text-align: left; }

            .stat-grid { grid-template-columns: 1fr; gap: .65rem; }
            .stat-grid.two, .stat-grid.five { grid-template-columns: 1fr 1fr; }
            .stat-grid.five .stat-tile:first-child { grid-column: 1 / -1; }
            .stat-tile { padding: .9rem; }
            .stat-label { min-height: 0; font-size: .58rem; }
            .stat-value { font-size: 1.9rem; }

            .outcome-banner { grid-template-columns: auto 1fr; padding: .85rem; }
            .outcome-policy { grid-column: 2; text-align: left; }

            .analysis-grid, .player-columns { grid-template-columns: 1fr; }

            .comparison-row {
                grid-template-columns: minmax(0, 1fr) minmax(6rem, .65fr) minmax(0, 1fr);
                min-height: 3.5rem;
            }

            .comparison-cell {
                padding: .7rem .55rem;
                font-size: .72rem;
                overflow-wrap: anywhere;
            }

            .comparison-cell:nth-child(2) { font-size: .55rem; }

            [data-testid="stHorizontalBlock"] { gap: .7rem; }
        }

        @media (prefers-reduced-motion: no-preference) {
            .wc-hero, .wc-status-grid {
                animation: fade-up .55s ease both;
            }
            .wc-status-grid { animation-delay: .07s; }

            @keyframes fade-up {
                from { opacity: 0; transform: translateY(12px); }
                to   { opacity: 1; transform: translateY(0); }
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Model load-or-train (cached resource — runs once per session) ────────────

@st.cache_resource(show_spinner=False)
def get_model_and_meta(_artifact_stamp: tuple[float, float, float]):
    """Load model artifacts, training inline if not present."""
    from model.train import train_and_save

    if not os.path.exists(MODEL_PATH):
        st.info("No trained model found. Training now — this takes about 30 seconds...")
        meta = train_and_save(verbose=False)
    else:
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                meta = json.load(f)
        else:
            meta = {"model_name": "Unknown", "accuracy": 0.0, "training_rows": 0, "features": len(FEATURE_COLUMNS)}

    import joblib
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, meta


def _artifact_stamp() -> tuple[float, float, float]:
    def _mtime(path: str) -> float:
        return os.path.getmtime(path) if os.path.exists(path) else 0.0

    return (_mtime(MODEL_PATH), _mtime(SCALER_PATH), _mtime(META_PATH))


# ── Data loading ──────────────────────────────────────────────────────────────

inject_tournament_theme()

with st.spinner("Loading match data..."):
    data = load_all()

matches_df = data["matches"]
goalscorers_df = data["goalscorers"]
shootouts_df = data["shootouts"]
rankings_df           = data["rankings"]
substitutions_df      = data.get("substitutions")
player_appearances_df = data.get("player_appearances")
player_goals_df       = data.get("player_goals")
award_winners_df      = data.get("award_winners")
copa_data             = data.get("copa_america")
euro_data             = data.get("euro_2024")
friendlies_data       = data.get("international_friendlies")
world_cup_2026_data   = data.get("world_cup_2026")

loaded_count = count_loaded(data)

if matches_df is None:
    st.error(
        "**matches.csv is required to run this app.**\n\n"
        "Drop it (and optionally goalscorers.csv, shootouts.csv, rankings.csv) "
        "into the project root directory, then refresh the page."
    )
    st.stop()

# ── Model ─────────────────────────────────────────────────────────────────────

with st.spinner("Loading model..."):
    model_obj, scaler_obj, meta = get_model_and_meta(_artifact_stamp())

accuracy = meta.get("accuracy", 0.0)
training_rows = meta.get("training_rows", 0)
model_name = meta.get("model_name", "Unknown")

# ── Tournament masthead + status ──────────────────────────────────────────────

n_matches = len(matches_df)
st.markdown(
    f"""
    <section class="wc-hero">
        <div class="wc-build-badge">Live UI · Build {UI_BUILD}</div>
        <div class="wc-kicker">North America 2026 · Prediction Centre</div>
        <h1>Match<br><span>Intelligence</span></h1>
        <p class="wc-hero-copy">
            Explore international matchups through historical form, rankings,
            tournament pedigree, goal models, and 2026 squad intelligence.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="wc-status-grid">
        <div class="wc-status-item">
            <span class="wc-status-label">Historical matches</span>
            <span class="wc-status-value">{n_matches:,}</span>
        </div>
        <div class="wc-status-item">
            <span class="wc-status-label">Prediction engine</span>
            <span class="wc-status-value"><span class="wc-live-dot"></span>{escape(str(model_name))}</span>
        </div>
        <div class="wc-status-item">
            <span class="wc-status-label">Training rows</span>
            <span class="wc-status-value">{training_rows:,}</span>
        </div>
        <div class="wc-status-item">
            <span class="wc-status-label">Core data coverage</span>
            <span class="wc-status-value">{loaded_count}/4 sources online</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if accuracy < 0.55 and accuracy > 0:
    st.sidebar.warning(
        "Model accuracy is below 55%. Consider retraining with a narrower date range."
    )

# ── Matchup controls ──────────────────────────────────────────────────────────

all_teams = get_all_teams(matches_df)
ranked_teams = (
    set(rankings_df["country_full"].dropna().astype(str))
    if rankings_df is not None and "country_full" in rankings_df.columns
    else set()
)
qualified_teams = set()
if world_cup_2026_data:
    qualified_frame = world_cup_2026_data.get("teams")
    if qualified_frame is not None and not qualified_frame.empty:
        qualified_teams = set(qualified_frame["Team"].dropna().astype(str))

available_teams = set(all_teams)
selectable_teams = sorted(available_teams & qualified_teams) + sorted(
    (available_teams & ranked_teams) - qualified_teams
)
if not selectable_teams:
    selectable_teams = all_teams

st.markdown('<p class="wc-section-kicker">Build the fixture</p>', unsafe_allow_html=True)
st.markdown('<p class="wc-section-title">Choose your matchup</p>', unsafe_allow_html=True)
st.markdown(
    f"""
    <p class="selector-note">
        {len(selectable_teams)} active national teams · World Cup 2026 teams listed first · Type to search
    </p>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox(
            "Team A",
            selectable_teams,
            index=selectable_teams.index("Argentina") if "Argentina" in selectable_teams else 0,
        )
    with col2:
        team_b = st.selectbox(
            "Team B",
            selectable_teams,
            index=selectable_teams.index("France") if "France" in selectable_teams else 1,
        )

    venue_col, action_col = st.columns([2, 1])
    with venue_col:
        venue_choice = st.radio(
            "Match setting",
            ["Team A home", "Neutral", "Team B home"],
            index=1,
            horizontal=True,
        )
    with action_col:
        st.markdown(
            '<div class="action-spacer" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        predict_btn = st.button(
            "Run match analysis",
            type="primary",
            use_container_width=True,
        )

venue_mode = {
    "Team A home": "team_a_home",
    "Neutral": "neutral",
    "Team B home": "team_b_home",
}[venue_choice]

# ── Prediction ────────────────────────────────────────────────────────────────

if predict_btn:
    if team_a == team_b:
        st.error("Please select two different teams.")
        st.stop()

    with st.spinner(f"Analysing {n_matches:,} historical matches..."):
        from model.predict import predict_match
        result = predict_match(
            team_a=team_a,
            team_b=team_b,
            matches_df=matches_df,
            goalscorers_df=goalscorers_df,
            shootouts_df=shootouts_df,
            rankings_df=rankings_df,
            substitutions_df=substitutions_df,
            player_appearances_df=player_appearances_df,
            player_goals_df=player_goals_df,
            award_winners_df=award_winners_df,
            copa_data=copa_data,
            euro_data=euro_data,
            friendlies_data=friendlies_data,
            world_cup_2026_data=world_cup_2026_data,
            venue_mode=venue_mode,
        )

    win_prob = result["win_prob"]
    draw_prob = result["draw_prob"]
    loss_prob = result["loss_prob"]
    conf_score = result["confidence_score"]
    conf_label = result["confidence_label"]

    st.divider()

    # ── Match board ───────────────────────────────────────────────────────────
    rank_a = result["team_a_rank"]
    rank_b = result["team_b_rank"]
    rank_date_a = result.get("team_a_rank_date")
    rank_date_b = result.get("team_b_rank_date")
    rank_label_a = (
        pd.Timestamp(rank_date_a).strftime("%b %Y")
        if rank_date_a else "snapshot date unavailable"
    )
    rank_label_b = (
        pd.Timestamp(rank_date_b).strftime("%b %Y")
        if rank_date_b else "snapshot date unavailable"
    )
    rank_sources = {
        str(result.get("team_a_rank_source") or "Ranking snapshot"),
        str(result.get("team_b_rank_source") or "Ranking snapshot"),
    }
    display_rank_source = (
        "Official FIFA snapshot"
        if rank_sources == {"Official FIFA snapshot"}
        else "Latest available snapshot per team"
    )
    model_rank_date = result.get("model_rank_date")
    model_rank_label = (
        pd.Timestamp(model_rank_date).strftime("%b %Y")
        if model_rank_date else "date unavailable"
    )
    safe_team_a = escape(str(team_a))
    safe_team_b = escape(str(team_b))
    safe_venue = escape(str(result.get("venue_label", venue_choice)))
    st.markdown(
        f"""
        <section class="match-board">
            <div class="match-meta">World Cup 2026 Match Lab · {safe_venue}</div>
            <div class="match-grid">
                <div class="team-card">
                    <div class="team-name">{safe_team_a}</div>
                    <div class="team-rank">FIFA rank #{rank_a if rank_a else "N/A"} · {rank_label_a}</div>
                </div>
                <div class="versus">VS</div>
                <div class="team-card">
                    <div class="team-name">{safe_team_b}</div>
                    <div class="team-rank">FIFA rank #{rank_b if rank_b else "N/A"} · {rank_label_b}</div>
                </div>
            </div>
            <div class="ranking-source-note">
                Display ranking: {escape(display_rank_source)} · Model ranking features: {escape(model_rank_label)}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── Probabilities ─────────────────────────────────────────────────────────
    label_a = f"{team_a} ({win_prob:.0%})" if win_prob > 0.15 else f"{win_prob:.0%}" if win_prob > 0.05 else ""
    label_draw = f"Draw ({draw_prob:.0%})" if draw_prob > 0.15 else f"{draw_prob:.0%}" if draw_prob > 0.05 else ""
    label_b = f"{team_b} ({loss_prob:.0%})" if loss_prob > 0.15 else f"{loss_prob:.0%}" if loss_prob > 0.05 else ""

    html_prob_bar = f"""
    <div class="probability-wrap">
        <div class="probability-labels">
            <span>Outcome probability</span>
            <span>Model distribution · 100%</span>
        </div>
        <div class="probability-bar">
            <div class="prob-a" style="width:{win_prob*100:.2f}%">{escape(label_a)}</div>
            <div class="prob-draw" style="width:{draw_prob*100:.2f}%">{escape(label_draw)}</div>
            <div class="prob-b" style="width:{loss_prob*100:.2f}%">{escape(label_b)}</div>
        </div>
    </div>
    """
    st.markdown(html_prob_bar, unsafe_allow_html=True)

    predicted_label = result.get("predicted_label")
    predicted_scoreline = result.get("predicted_scoreline")
    tile_a_class = "stat-tile accent" if predicted_label == 2 else "stat-tile"
    tile_draw_class = "stat-tile accent" if predicted_label == 1 else "stat-tile"
    tile_b_class = "stat-tile accent" if predicted_label == 0 else "stat-tile"
    if predicted_label == 2:
        outcome_text = f"{team_a} win"
        outcome_icon = "🏆"
    elif predicted_label == 0:
        outcome_text = f"{team_b} win"
        outcome_icon = "🏆"
    else:
        outcome_text = "Draw"
        outcome_icon = "🤝"
    if predicted_scoreline:
        outcome_text = f"{outcome_text} ({predicted_scoreline})"

    st.markdown(
        f"""
        <section class="report-section">
            <div class="report-heading">
                <div>
                    <div class="report-eyebrow">Match prediction</div>
                    <div class="report-title">Probability breakdown</div>
                </div>
                <div class="report-note">
                    {safe_team_a} vs {safe_team_b} · Probabilities sum to 100%
                </div>
            </div>
            <div class="stat-grid">
                <div class="{tile_a_class}">
                    <span class="stat-label">{safe_team_a} win</span>
                    <span class="stat-value">{win_prob:.1%}</span>
                </div>
                <div class="{tile_draw_class}">
                    <span class="stat-label">Draw</span>
                    <span class="stat-value">{draw_prob:.1%}</span>
                </div>
                <div class="{tile_b_class}">
                    <span class="stat-label">{safe_team_b} win</span>
                    <span class="stat-value">{loss_prob:.1%}</span>
                </div>
            </div>
            <div class="outcome-banner" style="margin-top:1rem">
                <div class="outcome-icon">{outcome_icon}</div>
                <div>
                    <div class="outcome-label">Predicted outcome</div>
                    <div class="outcome-value">{escape(outcome_text)}</div>
                </div>
                <div class="outcome-policy">{escape(result.get('decision_policy', 'highest probability'))}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── 2026 Squad Adjustments ────────────────────────────────────────────────
    wc2026_context = result.get("world_cup_2026_context", {})
    if wc2026_context.get("applied"):
        base = result.get("base_probabilities", {})
        classifier_probs = result.get("classifier_probabilities", {})
        shift_a = wc2026_context.get("team_a_probability_shift", 0)
        shift_draw = wc2026_context.get("draw_probability_shift", 0)
        shift_b = wc2026_context.get("team_b_probability_shift", 0)
        squad_edge_a = wc2026_context.get("team_a_squad_edge_shift", shift_a)
        squad_edge_b = wc2026_context.get("team_b_squad_edge_shift", shift_b)
        draw_context_shift = wc2026_context.get("draw_context_shift", shift_draw)
        base_a = float(base.get("win_prob", 0))
        base_draw = float(base.get("draw_prob", 0))
        base_b = float(base.get("loss_prob", 0))
        context_a = float(classifier_probs.get("win_prob", win_prob))
        context_draw = float(classifier_probs.get("draw_prob", draw_prob))
        context_b = float(classifier_probs.get("loss_prob", loss_prob))
        status_a = shift_status(squad_edge_a, "Squad edge boosts chance", "Squad edge reduces chance")
        status_draw = shift_status(draw_context_shift, "Close matchup boosts draw", "Close matchup lowers draw")
        status_b = shift_status(squad_edge_b, "Squad edge boosts chance", "Squad edge reduces chance")
        st.markdown(
            f"""
            <section class="report-section">
                <div class="report-heading">
                    <div>
                        <div class="report-eyebrow">Roster intelligence</div>
                        <div class="report-title">2026 squad impact</div>
                    </div>
                    <div class="report-note">
                        Probability movement after evaluating squad depth, age balance,
                        top-league experience, and the current 2026 roster snapshot.
                    </div>
                </div>
                <div class="stat-grid">
                    <div class="stat-tile">
                        <span class="stat-label">{safe_team_a} squad edge</span>
                        <span class="stat-value">{format_probability_shift(squad_edge_a)}</span>
                        <span class="shift-status">{escape(status_a)}</span>
                        <span class="stat-detail">Classifier probability {base_a:.1%} → {context_a:.1%} · net {format_probability_point_note(shift_a)}</span>
                    </div>
                    <div class="stat-tile">
                        <span class="stat-label">Draw context shift</span>
                        <span class="stat-value">{format_probability_shift(draw_context_shift)}</span>
                        <span class="shift-status">{escape(status_draw)}</span>
                        <span class="stat-detail">Classifier probability {base_draw:.1%} → {context_draw:.1%} · net {format_probability_point_note(shift_draw)}</span>
                    </div>
                    <div class="stat-tile">
                        <span class="stat-label">{safe_team_b} squad edge</span>
                        <span class="stat-value">{format_probability_shift(squad_edge_b)}</span>
                        <span class="shift-status">{escape(status_b)}</span>
                        <span class="stat-detail">Classifier probability {base_b:.1%} → {context_b:.1%} · net {format_probability_point_note(shift_b)}</span>
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    # ── Goal prediction ──────────────────────────────────────────────────────
    if result.get("goals_error"):
        st.warning(f"Goal prediction is unavailable: {result['goals_error']}")
    elif "expected_team_a_goals" in result and "expected_team_b_goals" in result:
        goal_meta = result.get("goal_model_meta", {})
        goal_metrics = goal_meta.get("metrics", {})
        validation_note = (
            f"Validation: MAE {goal_metrics.get('combined_mae', 0):.2f} · "
            f"exact score {goal_metrics.get('exact_score_accuracy', 0):.1%} · "
            f"over 2.5 accuracy {goal_metrics.get('over_2_5_accuracy', 0):.1%}"
            if goal_metrics else "Independent expected-goals model"
        )
        st.markdown(
            f"""
            <section class="report-section">
                <div class="report-heading">
                    <div>
                        <div class="report-eyebrow">Score forecast</div>
                        <div class="report-title">Goal prediction</div>
                    </div>
                    <div class="report-note">
                        {safe_venue} · {escape(validation_note)}
                    </div>
                </div>
                <div class="stat-grid five">
                    <div class="stat-tile">
                        <span class="stat-label">{safe_team_a} expected goals</span>
                        <span class="stat-value">{result['expected_team_a_goals']:.2f}</span>
                    </div>
                    <div class="stat-tile accent">
                        <span class="stat-label">Likely score</span>
                        <span class="stat-value">{result['likely_team_a_goals']}-{result['likely_team_b_goals']}</span>
                    </div>
                    <div class="stat-tile">
                        <span class="stat-label">{safe_team_b} expected goals</span>
                        <span class="stat-value">{result['expected_team_b_goals']:.2f}</span>
                    </div>
                    <div class="stat-tile">
                        <span class="stat-label">Total expected goals</span>
                        <span class="stat-value">{result['expected_total_goals']:.2f}</span>
                    </div>
                    <div class="stat-tile">
                        <span class="stat-label">Over 2.5 goals</span>
                        <span class="stat-value">{result['over_2_5_probability']:.1%}</span>
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    # ── Confidence ────────────────────────────────────────────────────────────
    n_used = result["n_features_used"]
    n_total = result.get("n_features_total", 10)
    n_displayed = len(result.get("key_factors", []))
    factors_html = "".join(
        f'<li><span class="factor-index">{idx:02d}</span><span>{escape(str(factor))}</span></li>'
        for idx, factor in enumerate(result["key_factors"], start=1)
    )
    st.markdown(
        f"""
        <section class="report-section">
            <div class="report-heading">
                <div>
                    <div class="report-eyebrow">Model context</div>
                    <div class="report-title">Confidence & deciding factors</div>
                </div>
                <div class="report-note">
                    {n_displayed} of {n_total} data categories displayed · {n_used} active · Rating: {escape(conf_label)}
                </div>
            </div>
            <div class="analysis-grid">
                <div class="confidence-score">
                    <strong>{conf_score:.0%}</strong>
                    <span>{escape(conf_label)} confidence</span>
                    <em>{n_displayed} categories shown · {n_used} data sources active</em>
                </div>
                <ul class="factor-list">{factors_html}</ul>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── Historical stats ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        <section class="report-section">
            <div class="report-heading">
                <div>
                    <div class="report-eyebrow">Tournament pedigree</div>
                    <div class="report-title">World Cup history</div>
                </div>
                <div class="report-note">Side-by-side historical comparison</div>
            </div>
            <div class="comparison-board">
                <div class="comparison-row header">
                    <div class="comparison-cell">{safe_team_a}</div>
                    <div class="comparison-cell">Metric</div>
                    <div class="comparison-cell">{safe_team_b}</div>
                </div>
                <div class="comparison-row">
                    <div class="comparison-cell">{result['team_a_avg_goals_wc']:.2f}</div>
                    <div class="comparison-cell">Goals per game</div>
                    <div class="comparison-cell">{result['team_b_avg_goals_wc']:.2f}</div>
                </div>
                <div class="comparison-row">
                    <div class="comparison-cell">{result['team_a_wc_titles']}</div>
                    <div class="comparison-cell">World Cup titles</div>
                    <div class="comparison-cell">{result['team_b_wc_titles']}</div>
                </div>
                <div class="comparison-row">
                    <div class="comparison-cell">{result['team_a_penalty_shootout_wins']}W / {result['team_a_penalty_shootout_losses']}L</div>
                    <div class="comparison-cell">Penalty shootouts</div>
                    <div class="comparison-cell">{result['team_b_penalty_shootout_wins']}W / {result['team_b_penalty_shootout_losses']}L</div>
                </div>
                <div class="comparison-row">
                    <div class="comparison-cell">{result['team_a_clean_sheet_rate']:.1%}</div>
                    <div class="comparison-cell">Clean sheet rate</div>
                    <div class="comparison-cell">{result['team_b_clean_sheet_rate']:.1%}</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── Official 2026 squad stats ────────────────────────────────────────────
    squad_a = result.get("team_a_2026_squad", {})
    squad_b = result.get("team_b_2026_squad", {})
    non_wc_squad_a = get_non_world_cup_2026_squad(team_a, world_cup_2026_data)
    non_wc_squad_b = get_non_world_cup_2026_squad(team_b, world_cup_2026_data)
    if squad_a or squad_b or non_wc_squad_a or non_wc_squad_b:
        def _fallback_squad(team_name: str, fallback_squad: list[dict]):
            if not fallback_squad:
                return
            source = fallback_squad[0]
            st.caption(
                f"Latest 2026 non-World-Cup matchday squad for {team_name}: "
                f"{source.get('source_match', 'recent match')} ({source.get('source_date', 'date unavailable')})."
            )
            starters = [p for p in fallback_squad if p.get("squad_role") == "Starter"]
            substitutes = [p for p in fallback_squad if p.get("squad_role") != "Starter"]
            if starters:
                st.markdown(
                    "**Starters:** "
                    + ", ".join(f"#{p.get('shirt_number')} {p.get('player_name')}" for p in starters)
                )
            if substitutes:
                st.markdown(
                    "**Substitutes:** "
                    + ", ".join(f"#{p.get('shirt_number')} {p.get('player_name')}" for p in substitutes)
                )

        if squad_a and squad_b:
            html_table = f"""
            <section class="report-section">
                <div class="report-heading">
                    <div>
                        <div class="report-eyebrow">Team composition</div>
                        <div class="report-title">2026 squad comparison</div>
                    </div>
                    <div class="report-note">Current roster profile and coaching context</div>
                </div>
                <div class="comparison-board">
                    <div class="comparison-row header">
                        <div class="comparison-cell">{safe_team_a}</div>
                        <div class="comparison-cell">Attribute</div>
                        <div class="comparison-cell">{safe_team_b}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{squad_a.get('depth_score', 0):.2f}</div>
                        <div class="comparison-cell">Depth score</div>
                        <div class="comparison-cell">{squad_b.get('depth_score', 0):.2f}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{squad_a.get('avg_age', 0):.1f} yrs</div>
                        <div class="comparison-cell">Average age</div>
                        <div class="comparison-cell">{squad_b.get('avg_age', 0):.1f} yrs</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{squad_a.get('top_league_share', 0):.0%}</div>
                        <div class="comparison-cell">Top-league share</div>
                        <div class="comparison-cell">{squad_b.get('top_league_share', 0):.0%}</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{squad_a.get('avg_height_cm', 0):.1f} cm</div>
                        <div class="comparison-cell">Average height</div>
                        <div class="comparison-cell">{squad_b.get('avg_height_cm', 0):.1f} cm</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{squad_a.get('goalkeepers', 0)} GK · {squad_a.get('defenders', 0)} DF · {squad_a.get('midfielders', 0)} MF · {squad_a.get('forwards', 0)} FW</div>
                        <div class="comparison-cell">Positions</div>
                        <div class="comparison-cell">{squad_b.get('goalkeepers', 0)} GK · {squad_b.get('defenders', 0)} DF · {squad_b.get('midfielders', 0)} MF · {squad_b.get('forwards', 0)} FW</div>
                    </div>
                    <div class="comparison-row">
                        <div class="comparison-cell">{escape(str(squad_a.get('head_coach', 'N/A')))}<br><span class="stat-detail">{escape(str(squad_a.get('coach_nationality', 'N/A')))}</span></div>
                        <div class="comparison-cell">Head coach</div>
                        <div class="comparison-cell">{escape(str(squad_b.get('head_coach', 'N/A')))}<br><span class="stat-detail">{escape(str(squad_b.get('coach_nationality', 'N/A')))}</span></div>
                    </div>
                </div>
            </section>
            """
            st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="report-eyebrow" style="margin-bottom:.3rem">Team composition</div>
                <div class="report-title" style="margin-bottom:1.1rem;color:#fff;font-family:'Avenir Next Condensed','DIN Condensed',sans-serif;font-size:clamp(1.6rem,3vw,2.3rem);font-weight:900;letter-spacing:-.025em;line-height:1;text-transform:uppercase">
                    2026 squad data
                </div>
                """,
                unsafe_allow_html=True,
            )
            def _squad_metrics(team_name: str, squad: dict, fallback_squad: list[dict], col):
                with col:
                    st.markdown(f"**{team_name}**")
                    if not squad:
                        if is_qualified_team(team_name, world_cup_2026_data):
                            st.caption("Official 2026 squad data unavailable.")
                        else:
                            st.caption("Not a qualified 2026 World Cup team; no official World Cup squad exists.")
                            _fallback_squad(team_name, fallback_squad)
                        return
                    st.metric("Squad depth parameter", f"{squad.get('depth_score', 0):.2f}")
                    st.metric("Average age", f"{squad.get('avg_age', 0):.1f}")
                    st.metric("Top-league share", f"{squad.get('top_league_share', 0):.0%}")
                    st.caption(
                        f"{squad.get('players', 0)} players · "
                        f"{squad.get('goalkeepers', 0)} GK / {squad.get('defenders', 0)} DF / "
                        f"{squad.get('midfielders', 0)} MF / {squad.get('forwards', 0)} FW · "
                        f"avg height {squad.get('avg_height_cm', 0):.1f} cm"
                    )
                    coach = squad.get("head_coach")
                    if coach:
                        st.caption(f"Head coach: {coach} ({squad.get('coach_nationality', 'N/A')})")

            sq_col1, sq_col2 = st.columns(2)
            _squad_metrics(team_a, squad_a, non_wc_squad_a, sq_col1)
            _squad_metrics(team_b, squad_b, non_wc_squad_b, sq_col2)


    # ── Super-sub alerts ──────────────────────────────────────────────────────
    ss_a = result.get("supersub_a")
    ss_b = result.get("supersub_b")
    if ss_a or ss_b:
        supersub_cards = []
        for ss in [ss_a, ss_b]:
            if ss:
                supersub_cards.append(
                    f"""
                    <div class="stat-tile">
                        <span class="stat-label">⚡ {escape(str(ss['team']))} super-sub</span>
                        <span class="outcome-value">{escape(str(ss['name']))}</span>
                        <span class="stat-detail">
                            {ss['goals_as_sub']} goals in {ss['sub_appearances']} substitute appearances ·
                            {ss['goal_rate']:.2f} goals per appearance
                        </span>
                    </div>
                    """
                )
        st.markdown(
            f"""
            <section class="report-section">
                <div class="report-heading">
                    <div>
                        <div class="report-eyebrow">Impact from the bench</div>
                        <div class="report-title">Super-sub alert</div>
                    </div>
                    <div class="report-note">Players with elite goal-scoring records as substitutes</div>
                </div>
                <div class="stat-grid two">{"".join(supersub_cards)}</div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    # ── Model verdict ─────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <section class="report-section">
            <div class="report-heading">
                <div>
                    <div class="report-eyebrow">Final assessment</div>
                    <div class="report-title">Model verdict</div>
                </div>
            </div>
            <div class="verdict-card">{escape(str(result['verdict']))}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── Key players ───────────────────────────────────────────────────────────
    kp_a = result.get("key_players_a", [])
    kp_b = result.get("key_players_b", [])
    if kp_a or kp_b:
        AWARD_BADGE = {
            "Golden Ball":  "🏆 Golden Ball",
            "Golden Boot":  "👟 Golden Boot",
            "Golden Glove": "🧤 Golden Glove",
        }

        def _player_cards(players: list) -> str:
            if not players:
                return '<div class="player-card"><div class="player-meta">No player data available.</div></div>'
            cards = []
            for p in players:
                award_badges = " · ".join(AWARD_BADGE.get(a, a) for a in p.get("awards", []))
                pos = p.get("position", "")
                pos_html = f'<span class="player-pos-badge">{escape(pos)}</span>' if pos else ""
                if p.get("source") == "Official 2026 squad":
                    primary_meta = (
                        f"#{p.get('shirt_number', '–')} · {escape(p.get('club', 'Club N/A'))} · "
                        f"Age {p.get('age', '–')} · {p.get('height_cm', '–')} cm"
                    )
                else:
                    assists = int(p.get("assists", 0) or 0)
                    assists_text = f" · {assists} ast" if assists else ""
                    primary_meta = (
                        f"{p.get('goals', 0)} goals{assists_text} · "
                        f"{p.get('appearances', 0)} apps · "
                        f"{p.get('goal_rate', 0):.2f} per app"
                    )
                impact_html = (
                    f'<span class="player-impact">Impact {p["impact_score"]:.2f}</span>'
                    if p.get("impact_score") is not None else ""
                )
                achievement_parts = [
                    str(v) for v in [
                        p.get("euro_summary", ""),
                        p.get("achievement", ""),
                        " · ".join(p.get("achievements", [])[:4]),
                        award_badges,
                    ] if v
                ]
                achievement_html = (
                    f'<div class="player-achievement">{escape(" · ".join(achievement_parts))}</div>'
                    if achievement_parts else ""
                )
                cards.append(
                    '<div class="player-card">'
                    f'<div class="player-name">{escape(str(p.get("name", "Unknown")))}{pos_html}</div>'
                    f'<div class="player-meta">{primary_meta}</div>'
                    f'{impact_html}'
                    f'{achievement_html}'
                    '</div>'
                )
            return "".join(cards)

        st.markdown(
            f"""
            <section class="report-section">
                <div class="report-heading">
                    <div>
                        <div class="report-eyebrow">Players to watch</div>
                        <div class="report-title">Key players</div>
                    </div>
                    <div class="report-note">Roster role, club context, and model impact</div>
                </div>
                <div class="player-columns">
                    <div class="player-team">
                        <div class="player-team-title">{safe_team_a}</div>
                        {_player_cards(kp_a)}
                    </div>
                    <div class="player-team">
                        <div class="player-team-title">{safe_team_b}</div>
                        {_player_cards(kp_b)}
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="wc-footer">Independent prediction experience · Historical data and model estimates are informational</div>',
    unsafe_allow_html=True,
)
