"""
ui_theme.py
===========
Presentation layer: styling, human-readable labels, and translation of internal
backend objects into language a *user* understands.

Design rule for this file: **nothing technical reaches the screen by default.**
Route names, backend identifiers, similarity scores, OCR confidence and plan
strings are engineering concepts. They are useful during development, so they
are kept — but only behind an explicit "Developer mode" toggle.

Anything that maps internal state → user-facing words belongs here, not in
app.py, so the vocabulary stays consistent across the whole UI.
"""
from __future__ import annotations

import re
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

# A calm, professional palette. Deliberately low-chroma: this is an analysis
# tool, and saturated colour reads as decoration rather than signal.
COLORS = {
    "bg":            "#0F1117",
    "surface":       "#161923",
    "surface_alt":   "#1C2029",
    "border":        "#262B36",
    "text":          "#E6E8EE",
    "text_muted":    "#9BA1AE",
    "accent":        "#4F7DF3",
    "accent_soft":   "#1E2A4A",
    "success":       "#3FB984",
    "success_soft":  "#12301F",
    "warning":       "#E0A33E",
    "warning_soft":  "#33280F",
    "danger":        "#E0625E",
    "danger_soft":   "#331A19",
}

CSS = f"""
<style>
/* ── Base ────────────────────────────────────────────────────────────────── */
.stApp {{
    background: {COLORS['bg']};
}}

/* Tighten Streamlit's default chrome: less vertical padding, calmer rhythm. */
.block-container {{
    padding-top: 3rem;
    padding-bottom: 6rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1140px;
}}

/* Streamlit's main block can clip content at the right edge on wide viewports;
   give the scroll container a little breathing room. */
[data-testid="stMainBlockContainer"] {{
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}}

/* Streamlit ships a light shell behind the app; force the dark surface. */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"] {{
    background: {COLORS['bg']} !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stBottom"], [data-testid="stBottom"] > div {{
    background: {COLORS['bg']} !important;
}}

/* ── Typography ──────────────────────────────────────────────────────────── */
/* Setting `color` on html/body is not enough: Streamlit renders message text
   inside its own markdown containers, which carry a light-theme default
   (rgb(49,51,63)) — near-black on our near-black surface. The colour has to be
   pushed down to the actual text nodes. */
html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stMarkdown"],
[data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"],
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em {{
    color: {COLORS['text']} !important;
}}

/* Retire Streamlit's light base palette so nothing inherits near-black text. */
:root {{
    --text-color: {COLORS['text']};
    --body-text-color: {COLORS['text']};
    --secondary-text-color: {COLORS['text_muted']};
}}

/* Readable defaults for loose prose outside chat bubbles too. */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    color: {COLORS['text']};
    line-height: 1.62;
}}

h1, h2, h3, h4 {{
    font-weight: 600 !important;
    letter-spacing: -0.015em;
    color: {COLORS['text']} !important;
}}

/* ── Header ──────────────────────────────────────────────────────────────── */
.app-header {{
    display: flex;
    align-items: center;
    gap: 0.85rem;
    /* Give the title room to breathe; Streamlit's header floats above it. */
    margin: 0 0 0.5rem 0;
    padding: 0 0 0.9rem 0;
    border-bottom: 1px solid {COLORS['border']};
    min-height: 46px;
}}
.app-header .mark {{
    width: 40px; height: 40px;
    border-radius: 10px;
    background: linear-gradient(135deg, {COLORS['accent']}, #7A5AF8);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    color: #fff;
    flex-shrink: 0;
    line-height: 1;
}}
.app-header > div:nth-child(2) {{ min-width: 0; flex: 1 1 auto; }}
.app-header .title {{
    font-size: 1.32rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.2;
    color: {COLORS['text']};
}}
.app-header .subtitle {{
    font-size: 0.82rem;
    color: {COLORS['text_muted']};
    margin-top: 0.15rem;
}}
.app-header .badge {{
    margin-left: auto;
    flex-shrink: 0;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {COLORS['success']};
    background: {COLORS['success_soft']};
    border: 1px solid {COLORS['success']}44;
    padding: 0.28rem 0.6rem;
    border-radius: 999px;
    white-space: nowrap;
    line-height: 1.1;
}}

/* ── Attachment area ─────────────────────────────────────────────────────── */
/* The pinned-image bar and its Remove control sit between the transcript and
   the docked composer. Constrain both to the transcript column and size the
   button to its label, so they read as one attached unit. */
.attach-bar + div,
[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"]:has(
    button[data-testid^="stBaseButton"]
) {{
    max-width: 1140px;
    margin-left: auto;
    margin-right: auto;
}}

/* "Remove image": a small, clearly-interactive secondary control, not a
   full-width slab. */
[data-testid="stMainBlockContainer"] button[data-testid^="stBaseButton"] {{
    font-size: 0.78rem !important;
    padding: 0.3rem 0.85rem !important;
    background-color: transparent !important;
    border: 1px solid #3A4150 !important;
    color: {COLORS['text_muted']} !important;
}}
[data-testid="stMainBlockContainer"] button[data-testid^="stBaseButton"] *,
[data-testid="stMainBlockContainer"] button[data-testid^="stBaseButton"] p {{
    color: {COLORS['text_muted']} !important;
    -webkit-text-fill-color: {COLORS['text_muted']} !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}}
[data-testid="stMainBlockContainer"] button[data-testid^="stBaseButton"]:hover {{
    border-color: {COLORS['danger']}aa !important;
    background-color: {COLORS['danger_soft']} !important;
}}
[data-testid="stMainBlockContainer"] button[data-testid^="stBaseButton"]:hover *,
[data-testid="stMainBlockContainer"] button[data-testid^="stBaseButton"]:hover p {{
    color: {COLORS['danger']} !important;
    -webkit-text-fill-color: {COLORS['danger']} !important;
}}

/* ── Chat bubbles ────────────────────────────────────────────────────────── */
/* Both roles previously shared one surface colour and full width, so the user's
   message and the assistant's reply ran together with no visible boundary —
   the user's text looked like it was floating unattached.

   Give the user's turn a distinct, slightly raised treatment and a visible
   label, so each turn is unambiguous. */
[data-testid="stChatMessage"] {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 1rem 1.15rem;
    /* `overflow: visible` stops the first bubble's top edge being clipped by the
       header rule directly above it. */
    margin: 0.5rem 0 0.6rem 0;
    overflow: visible;
}}

/* Role labels — a small caption above each turn. */
[data-testid="stChatMessage"]::before {{
    display: block;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {COLORS['text_muted']};
    margin-bottom: 0.45rem;
}}

/* The user's turn is marked by an avatar element inside the wrapper. Verified
   from the live DOM: the testid is "stChatMessageAvatarUser" (capital U). An
   earlier guess at "chatAvatarUser" matched nothing, which is why the bubbles
   stayed visually identical. */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
    background: {COLORS['accent_soft']} !important;
    border-color: {COLORS['accent']}55 !important;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])::before {{
    content: "You";
}}
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"]))::before {{
    content: "Assistant";
}}

/* The very first message needs clearance from the header's bottom border. */
[data-testid="stChatMessage"]:first-of-type {{
    margin-top: 0.9rem;
}}

/* ── Confidence pill ─────────────────────────────────────────────────────── */
.pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.22rem 0.62rem;
    border-radius: 999px;
    border: 1px solid transparent;
    white-space: nowrap;
}}
.pill-high    {{ color: {COLORS['success']}; background: {COLORS['success_soft']};
                 border-color: {COLORS['success']}55; }}
.pill-medium  {{ color: {COLORS['warning']}; background: {COLORS['warning_soft']};
                 border-color: {COLORS['warning']}55; }}
.pill-low     {{ color: {COLORS['text_muted']}; background: {COLORS['surface_alt']};
                 border-color: {COLORS['border']}; }}
.pill-question{{ color: {COLORS['accent']}; background: {COLORS['accent_soft']};
                 border-color: {COLORS['accent']}55; }}

/* ── Language chips ──────────────────────────────────────────────────────── */
/* A quiet row under each answer showing the language it was written in, and
   flagging a language switch or code-mixed input. Muted by default so it reads
   as metadata, not decoration. */
.lang-row {{
    display: flex; flex-wrap: wrap; gap: 0.35rem;
    margin: 0.5rem 0 0.15rem 0;
}}
.lang-chip {{
    font-size: 0.7rem;
    font-weight: 600;
    color: {COLORS['text_muted']};
    background: {COLORS['surface_alt']};
    border: 1px solid {COLORS['border']};
    border-radius: 999px;
    padding: 0.2rem 0.55rem;
    white-space: nowrap;
}}
.lang-chip-switch {{
    color: {COLORS['accent']};
    background: {COLORS['accent_soft']};
    border-color: {COLORS['accent']}55;
}}
.lang-chip-mixed {{
    color: {COLORS['warning']};
    background: {COLORS['warning_soft']};
    border-color: {COLORS['warning']}55;
}}

/* ── Evidence chips ──────────────────────────────────────────────────────── */
.source-row {{
    display: flex; flex-wrap: wrap; gap: 0.4rem;
    margin: 0.55rem 0 0.1rem 0;
}}
.source-chip {{
    font-size: 0.72rem;
    color: {COLORS['text_muted']};
    background: {COLORS['surface_alt']};
    border: 1px solid {COLORS['border']};
    border-radius: 7px;
    padding: 0.24rem 0.55rem;
}}
.source-chip .dot {{
    display: inline-block;
    width: 5px; height: 5px;
    border-radius: 50%;
    background: {COLORS['accent']};
    margin-right: 0.4rem;
    vertical-align: middle;
}}

/* ── Pinned image attachment (shown once, above the composer) ────────────── */
/* Streamlit's docked composer is inset from the main content column, so the
   attachment strip is inset to match — otherwise the two edges disagree. */
.attach-bar {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    /* Full-width inside the content column so its edges match the message
       bubbles above it (measured: bubbles are x=340 w=900; the bar was
       x=360 w=860, giving a visible 20px step). */
    margin: 0 0 0.6rem 0;
    max-width: none;
}}
.attach-bar img {{
    width: 44px; height: 44px;
    object-fit: cover;
    border-radius: 7px;
    border: 1px solid {COLORS['border']};
    flex-shrink: 0;
}}
.attach-bar .meta {{ min-width: 0; flex: 1 1 auto; }}
.attach-bar .name {{
    font-size: 0.8rem;
    font-weight: 600;
    color: {COLORS['text']};
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.attach-bar .hint {{
    font-size: 0.72rem;
    color: {COLORS['text_muted']};
    margin-top: 0.1rem;
}}
.attach-bar .tag {{
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: {COLORS['accent']};
    background: {COLORS['accent_soft']};
    border: 1px solid {COLORS['accent']}55;
    border-radius: 999px;
    padding: 0.16rem 0.5rem;
    flex-shrink: 0;
}}

/* An image introduced in chat: smaller than the old full-width repeat. */
.chat-image {{
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    margin: 0.2rem 0 0.55rem 0;
    max-width: 260px;
}}

/* ── Composer area ───────────────────────────────────────────────────────── */
/* The composer is DOCKED as a fixed overlay (~153px tall), not in normal flow,
   so the transcript must reserve at least that much trailing space or the last
   message renders underneath it and gets clipped.

   Measured: with 1.5rem the final message ended 330px below the dock top.
   6rem left too much dead space on short chats, so this is sized to clear the
   dock exactly, with a little slack for a two-line composer. */
[data-testid="stMainBlockContainer"] {{
    padding-bottom: 11rem;
}}
[data-testid="stBottom"] > div {{
    padding-top: 0.6rem;
    padding-bottom: 0.75rem;
    border-top: 1px solid {COLORS['border']};
    background: {COLORS['bg']};
}}
/* Measured misalignment: header and message bubbles sit at x=340 (width 900),
   while the docked composer sat at x=420 (width 740).

   Traced by walking the DOM: the offender is `stBottomBlockContainer`, which
   Streamlit ships with `padding: 0 80px`. An earlier attempt added padding on
   the dock wrapper instead, which made it worse. Zero the real culprit. */
/* Streamlit's shipped 80px side padding on the dock block is left alone.

   Several attempts to "align" the composer with the transcript by zeroing or
   overriding padding/max-width here backfired: the composer collapsed into a
   narrow strip that overlapped the last message bubble. The dock's own layout
   is load-bearing, and a small horizontal offset is far less costly than a
   broken composer. */
[data-testid="stBottomBlockContainer"] {{
    padding-bottom: 0.25rem !important;
}}

/* Keep the composer and the attach control on one visual row. */
[data-testid="stBottom"] [data-testid="stHorizontalBlock"] {{
    align-items: center;
    gap: 0.6rem;
}}

/* ── Chat history list (sidebar) ─────────────────────────────────────────── */
/* Every sidebar button is reset to a quiet row first, then specific roles are
   layered on. Streamlit renders buttons with a light default surface and a
   centred, wrapping label — both look wrong in a narrow list. */
[data-testid="stSidebar"] [data-testid="stButton"] button {{
    text-align: left;
    justify-content: flex-start;
    padding: 0.45rem 0.55rem;
    font-size: 0.79rem;
    font-weight: 400;
    line-height: 1.3;
    background: transparent;
    border: 1px solid transparent;
    color: {COLORS['text_muted']};
    height: auto;
    min-height: 0;
    box-shadow: none;
}}
/* The label inside a Streamlit button is a <div>/<p>; force it to a single
   clipped line so a long title can never blow the row apart. */
[data-testid="stSidebar"] [data-testid="stButton"] button p,
[data-testid="stSidebar"] [data-testid="stButton"] button div {{
    text-align: left;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
    font-size: inherit;
    color: inherit !important;
}}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {{
    background: {COLORS['surface_alt']};
    border-color: {COLORS['border']};
    color: {COLORS['text']};
}}
/* The active conversation reads as a selected row, not a loud CTA.
   Both `color` and `opacity` must be pinned: Streamlit fades primary buttons
   and the fade is what made the selected title look disabled. */
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"],
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] *,
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] p,
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] div {{
    color: {COLORS['text']} !important;
    opacity: 1 !important;
    -webkit-text-fill-color: {COLORS['text']} !important;
}}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {{
    background: {COLORS['accent_soft']} !important;
    border: 1px solid {COLORS['accent']}66 !important;
    font-weight: 600;
}}
/* Streamlit dims secondary button labels hard; unselected history titles were
   almost unreadable. Use the muted tone (readable, still clearly secondary) and
   go full-brightness on hover — the selected row is styled primary above. */
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] p,
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] div {{
    color: #C3C8D4 !important;
    opacity: 1 !important;
}}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover p,
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover div {{
    color: {COLORS['text']} !important;
}}
/* The delete glyph should stay quiet until the row is hovered. */
[data-testid="stSidebar"] [data-testid="stButton"] button:has(p:text-is("×")) p {{
    color: {COLORS['text_muted']} !important;
}}

/* The delete button: square, centred glyph, muted until hovered.
   The glyph is U+00D7 (×), not U+2715 — the selector must match the character
   actually rendered, or the button collapses to zero width and disappears. */
[data-testid="stSidebar"] [data-testid="stButton"] button:has(p:text-is("×")),
[data-testid="stSidebar"] [data-testid="stButton"] button:has(p:text-is("✕")) {{
    justify-content: center;
    text-align: center;
    padding: 0.45rem 0;
    min-width: 26px;
    color: {COLORS['text_muted']};
    opacity: 0.7;
}}
[data-testid="stSidebar"] [data-testid="stButton"] button:has(p:text-is("×")):hover,
[data-testid="stSidebar"] [data-testid="stButton"] button:has(p:text-is("✕")):hover {{
    opacity: 1;
    color: {COLORS['danger']} !important;
    border-color: {COLORS['danger']}55 !important;
    background: {COLORS['danger_soft']} !important;
}}
[data-testid="stSidebar"] [data-testid="stButton"] button:has(p:text-is("×")) p,
[data-testid="stSidebar"] [data-testid="stButton"] button:has(p:text-is("✕")) p {{
    text-align: center;
    font-size: 0.95rem;
}}

/* "＋ New chat" — the sidebar's primary action. Streamlit's primary variant
   renders with a tinted fill that made the dark label low-contrast here. */
[data-testid="stSidebar"] hr {{ display: none; }}

[data-testid="stSidebar"] button[data-testid^="stBaseButton"] {{
    background-color: {COLORS['accent']}22 !important;
    border: 1px solid {COLORS['accent']}88 !important;
}}
[data-testid="stSidebar"] button[data-testid^="stBaseButton"] *,
[data-testid="stSidebar"] button[data-testid^="stBaseButton"] p,
[data-testid="stSidebar"] button[data-testid^="stBaseButton"] span {{
    color: {COLORS['text']} !important;
    -webkit-text-fill-color: {COLORS['text']} !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}}
[data-testid="stSidebar"] button[data-testid^="stBaseButton"]:hover {{
    background-color: {COLORS['accent']} !important;
}}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {COLORS['surface']};
    border-right: 1px solid {COLORS['border']};
}}
/* Streamlit's sidebar block has almost no horizontal padding, so rows ran to
   the panel edge and the history title was clipped by the rounded border. */
[data-testid="stSidebar"] .block-container,
[data-testid="stSidebar"] [data-testid="stSidebarContent"] > div {{
    padding-top: 1.4rem;
    padding-left: 0.9rem;
    padding-right: 0.9rem;
}}
/* Column gaps inside the sidebar are generous by default; tighten them so the
   title and its delete control read as one row. */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
    gap: 0.25rem !important;
}}

.status-row {{
    display: flex; align-items: center; gap: 0.55rem;
    font-size: 0.84rem;
    padding: 0.34rem 0;
    color: {COLORS['text']};
}}
.status-dot {{
    width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
}}
.dot-ok   {{ background: {COLORS['success']}; box-shadow: 0 0 0 3px {COLORS['success']}22; }}
.dot-warn {{ background: {COLORS['warning']}; box-shadow: 0 0 0 3px {COLORS['warning']}22; }}
.dot-off  {{ background: {COLORS['danger']};  box-shadow: 0 0 0 3px {COLORS['danger']}22; }}

/* Section headings in the sidebar. Streamlit wraps `st.markdown` in its own
   containers that apply a bright default colour, so every descendant needs the
   muted tone pinned — otherwise these render as loud white caps. */
.side-label,
.side-label *,
[data-testid="stSidebar"] .side-label p {{
    font-size: 0.66rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {COLORS['text_muted']} !important;
    margin: 1.1rem 0 0.5rem 0 !important;
    line-height: 1.2 !important;
}}

/* ── Buttons (main area) ─────────────────────────────────────────────────── */
/* The styling above is scoped to the sidebar, which left buttons in the main
   pane — such as "Detach image" — with Streamlit's default look.

   Measured symptoms before this rule was made authoritative: the button's fill
   computed to rgb(255,255,255) while its label computed to rgb(230,232,238),
   i.e. light text on a white pill, which rendered the control invisible.
   `background` needs `!important` here because Streamlit's own theme rule wins
   on specificity, and the label needs `-webkit-text-fill-color` because
   Streamlit sets that property directly on the <p>. */
/* Streamlit styles its buttons with an emotion CSS-in-JS class injected AFTER
   this stylesheet, so equal-specificity rules lose. The reliable hook is the
   element's own data-testid (e.g. stBaseButton-secondary), targeted directly
   with !important.

   NOTE: the selector must NOT be scoped under [data-testid="stMain"]. The
   composer and its buttons live in the bottom dock (stBottom), a sibling of
   stMain — scoping there silently missed "Detach image" and left it as light
   text on Streamlit's white pill, i.e. invisible. */
/* Measured: a secondary button's fill (surface_alt, #1C2029) sits only ~7
   luminance steps above the page (#0F1117), so buttons like "Remove image"
   read as disabled even though they are enabled and their label is bright.
   Raise the fill and strengthen the border so the control looks interactive. */
button[data-testid^="stBaseButton"],
[data-testid="stButton"] button,
.stButton > button {{
    border-radius: 8px !important;
    border: 1px solid #39404F !important;
    background-color: #232834 !important;
    background-image: none !important;
    box-shadow: none !important;
    font-size: 0.84rem !important;
    opacity: 1 !important;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}}
button[data-testid^="stBaseButton"]:hover,
[data-testid="stButton"] button:hover,
.stButton > button:hover {{
    border-color: {COLORS['accent']}aa !important;
    background-color: {COLORS['accent_soft']} !important;
}}
/* The label: Streamlit sets -webkit-text-fill-color directly, which beats
   `color` alone — both must be overridden. */
button[data-testid^="stBaseButton"] *,
button[data-testid^="stBaseButton"] p,
[data-testid="stButton"] button p,
.stButton > button p,
.stButton > button span {{
    color: {COLORS['text']} !important;
    -webkit-text-fill-color: {COLORS['text']} !important;
    opacity: 1 !important;
    font-size: 0.84rem !important;
}}
button[data-testid^="stBaseButton"]:hover,
[data-testid="stButton"] button:hover,
.stButton > button:hover {{
    border-color: {COLORS['accent']}aa !important;
    background-color: {COLORS['accent_soft']} !important;
}}
button[data-testid^="stBaseButton"]:hover *,
[data-testid="stButton"] button:hover p,
.stButton > button:hover p {{
    color: {COLORS['text']} !important;
    -webkit-text-fill-color: {COLORS['text']} !important;
}}
/* A disabled control must still be readable, not invisible. */
button[data-testid^="stBaseButton"]:disabled,
.stButton > button:disabled {{
    opacity: 0.6 !important;
    cursor: not-allowed;
}}
button[data-testid^="stBaseButton"]:disabled *,
.stButton > button:disabled p {{
    color: {COLORS['text_muted']} !important;
    -webkit-text-fill-color: {COLORS['text_muted']} !important;
}}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
/* Streamlit's chat input has a hard-coded light background; restyle the whole
   control so it matches the dark surface instead of flashing white. */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInputContainer"],
[data-testid="stChatInput"] textarea {{
    background: {COLORS['surface']} !important;
    color: {COLORS['text']} !important;
    font-size: 0.92rem;
}}
[data-testid="stChatInput"] {{ border-radius: 12px; }}
[data-testid="stChatInput"] textarea::placeholder {{
    color: {COLORS['text_muted']} !important;
}}
[data-testid="stChatInput"] button {{ color: {COLORS['text_muted']} !important; }}

/* Compact, self-contained uploader that sits beside the composer rather than
   sprawling across the row and overlapping it. */
/* A compact "attach" control that reads as a button, not a drop zone.
   Streamlit renders the uploader as a full-width bordered <section>; strip that
   chrome entirely and style only the inner button.

   Centred over a narrow column so it matches the attachment bar's footprint
   and sits directly above the composer instead of floating mid-transcript. */
[data-testid="stFileUploader"] {{
    margin: 0 auto;
    width: 100%;
    max-width: 220px;
}}
/* The widget's own label ("Add an image to this chat") takes a whole line and
   left a large empty band above the composer. Hide it — the button reads
   clearly on its own, and the empty state already explains attaching. */
[data-testid="stFileUploader"] > label {{
    display: none !important;
}}
/* Give the wrapper a matching height so it aligns with the composer row. */
[data-testid="stFileUploader"] > section {{
    height: 42px;
}}
[data-testid="stFileUploader"] > section {{
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    background: {COLORS['surface']};
    min-height: 46px;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
}}
[data-testid="stFileUploader"] > section > div {{
    padding: 0;
    width: 100%;
}}
[data-testid="stFileUploader"] > section:hover {{
    border-color: {COLORS['accent']}88;
    background: {COLORS['accent_soft']};
}}
/* Hide the verbose drop-zone copy ("200MB per file…") and the upload icon — the
   button label says enough, and that copy is what made this read as a slab. */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] > section svg,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {{
    display: none !important;
}}
[data-testid="stFileUploader"] button {{
    font-size: 0.8rem;
    padding: 0.35rem 0.4rem;
    background: transparent;
    border: none;
    color: {COLORS['text_muted']};
    box-shadow: none;
    width: 100%;
    white-space: nowrap;
}}
[data-testid="stFileUploader"] button:hover {{ color: {COLORS['text']}; }}
[data-testid="stFileUploader"] button p {{
    color: inherit !important;
    font-size: 0.8rem;
    white-space: nowrap;
}}

/* ── Expanders (developer panels) ────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid {COLORS['border']};
    border-radius: 9px;
    background: {COLORS['surface_alt']};
}}
[data-testid="stExpander"] summary {{
    font-size: 0.82rem;
    color: {COLORS['text_muted']};
}}
[data-testid="stExpander"] summary:hover {{ color: {COLORS['text']}; }}

/* ── History row: keep title and delete control on one measured line ────── */
/* The title's column and the × column must not collapse: when they do, the ×
   drifts outside the row and the title clips mid-word. */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div,
[data-testid="stSidebar"] [data-testid="column"] {{
    min-width: 0;
    overflow: hidden;
}}
[data-testid="stSidebar"] [data-testid="stButton"] {{
    max-width: 100%;
}}

/* ── Toggle (Developer mode) ─────────────────────────────────────────────── */
/* Streamlit renders the label as a sibling of the control rather than inside
   the <label>, so target the widget row and everything within it. The label
   must have NO background of its own — the default gave it a grey box that
   looked like a broken text input. */
/* Measured via the DOM: the toggle's <label> is itself a filled, bordered
   container (rgb(28,32,41) on a 1px rgb(38,43,54) border). That is the grey
   box behind "Developer mode".

   The label carries no data-testid — Streamlit styles it with single-class
   utility names ("st-af", "st-ag", …) injected after this sheet. A selector
   of equal specificity therefore loses, so these rules deliberately add
   specificity: `#root` (present on every Streamlit app) + attribute + element
   + element. */
#root [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
#root [data-testid="stSidebar"] [data-testid="stToggle"] label,
#root [data-testid="stSidebar"] [data-testid="stCheckbox"] label > div,
#root [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
#root [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    border: none !important;
    border-color: transparent !important;
    box-shadow: none !important;
}}

/* NOTE: a pair of "re-assert the switch fill" rules lived here. They were left
   over from when Developer mode was an st.toggle, and `:has(> input)` matched
   the checkbox's *label* once the widget changed — painting it surface_alt with
   a border, which is exactly the grey box that appeared behind the text. The
   real checkbox styles are defined above; nothing is needed here. */
[data-testid="stCheckbox"],
[data-testid="stCheckbox"] *,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stToggle"],
[data-testid="stToggle"] * {{
    color: {COLORS['text']} !important;
    font-size: 0.84rem !important;
    opacity: 1 !important;
}}
[data-testid="stCheckbox"] p,
[data-testid="stWidgetLabel"] p {{
    margin: 0 !important;
    line-height: 1.35 !important;
}}
/* Streamlit's toggle is NOT a [role="checkbox"] element. The real structure is
   an <input aria-checked> inside a <label>, with the track and knob as plain
   sibling <div>s. Selecting by role matched nothing, which is why the control
   kept its default near-white look. */
[data-testid="stCheckbox"] label > div:has(> input),
[data-testid="stToggle"] label > div:has(> input) {{
    border: 1px solid {COLORS['border']} !important;
}}
/* The (?) help ring is absolutely positioned by Streamlit, which flings it to
   the far edge of the row. Pull it back so it sits beside its label. */
[data-testid="stTooltipIcon"],
[data-testid="stTooltipHoverTarget"] {{
    opacity: 0.5 !important;
    color: {COLORS['text_muted']} !important;
    position: static !important;
    margin-left: 0.3rem !important;
    vertical-align: middle;
}}
[data-testid="stTooltipHoverTarget"] svg {{
    width: 13px !important;
    height: 13px !important;
}}

/* Keep the switch and its label on one tight line. */
[data-testid="stCheckbox"],
[data-testid="stToggle"] {{
    display: flex !important;
    align-items: center !important;
    gap: 0.3rem !important;
    justify-content: flex-start !important;
}}
[data-testid="stCheckbox"] > label,
[data-testid="stToggle"] > label {{
    display: flex !important;
    align-items: center !important;
    gap: 0.3rem !important;
}}

/* ── Checkbox appearance (Developer mode) ───────────────────────────────── */
/* Developer mode is a st.checkbox, not a st.toggle. A checkbox renders a real
   <input type="checkbox"> with a styled sibling box, which themes reliably —
   unlike the toggle, whose <label> carried a fill that Streamlit's own
   single-class utility rules kept re-applying over ours.

   Off state: an outlined, unfilled box. Any coloured or filled box would read
   as "on" at a glance. */
[data-testid="stCheckbox"] span[aria-checked="false"],
[data-testid="stCheckbox"] div[aria-checked="false"],
[data-testid="stCheckbox"] > label > div:first-of-type {{
    background-color: transparent !important;
    background-image: none !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 5px !important;
    box-shadow: none !important;
}}

/* On state: accent fill with a tick, clearly different from off. */
[data-testid="stCheckbox"] span[aria-checked="true"],
[data-testid="stCheckbox"] div[aria-checked="true"],
[data-testid="stCheckbox"] input:checked + div {{
    background-color: {COLORS['accent']} !important;
    border: 1px solid {COLORS['accent']} !important;
    border-radius: 5px !important;
}}

/* Legacy toggle rules — harmless, and retained so a future re-introduction of
   st.toggle still gets the dark track instead of a near-white one. */
[data-testid="stCheckbox"] :has(> input[aria-checked="true"]) {{
    background: {COLORS['accent']} !important;
    border: 1px solid {COLORS['accent']} !important;
    border-radius: 999px !important;
}}
[data-testid="stCheckbox"] :has(> input[aria-checked="true"]) > div {{
    background: #ffffff !important;
    opacity: 1 !important;
}}

/* ── Sidebar captions / helper text ──────────────────────────────────────── */
/* Scope the muted colour to loose markdown only. A blanket `p, span` rule bled
   onto buttons and toggles, dimming text that should stay bright. */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    color: {COLORS['text_muted']};
    font-size: 0.78rem;
}}

/* ── Toggle help affordance ─────────────────────────────────────────────── */
/* The little (?) ring inherits a default that renders as a stray grey circle. */
[data-testid="stTooltipIcon"], [data-testid="stTooltipHoverTarget"] svg {{
    opacity: 0.45;
}}
[data-testid="stTooltipHoverTarget"] svg:hover {{ opacity: 0.9; }}

/* ── Metrics (developer only) ────────────────────────────────────────────── */
[data-testid="stMetricValue"] {{ font-size: 1.05rem; }}
[data-testid="stMetricLabel"] {{ font-size: 0.7rem; color: {COLORS['text_muted']}; }}

/* ── Chat message avatars ────────────────────────────────────────────────── */
/* Streamlit renders a default avatar per role. On this theme the user avatar
   came out as a saturated red circle and the assistant as an orange square —
   both read as error states. Hide them: the role is already obvious from the
   alignment, bubble tone and content. */
[data-testid="stChatMessage"] > img,
[data-testid="stChatMessageAvatar"],
[data-testid="stChatMessage"] [data-testid*="Avatar"] {{
    display: none !important;
}}

/* ── Hide Streamlit furniture we don't want in a product UI ──────────────── */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stStatusWidget"] {{ display: none; }}

/* Monospace only where it is genuinely code (dev panels). */
code {{
    background: {COLORS['surface_alt']} !important;
    color: {COLORS['text_muted']} !important;
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    padding: 0.05rem 0.32rem;
    font-size: 0.78rem;
}}
</style>
"""


def inject() -> None:
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Internal state → user-facing language
# ─────────────────────────────────────────────────────────────────────────────

# What the assistant *did*, phrased as an activity rather than a route name.
_ACTIVITY_LABELS = {
    "clarify": "Needs a little more detail",
    "describe_image": "Looked at your image",
    "read_text": "Read the text in your image",
    "analyse_chart": "Analysed the chart",
    "answer_with_image": "Answered from your image",
    "follow_up": "Used earlier context",
    "general_question": "Answered from reference material",
    "small_talk": "Replied",
    "refuse_unsupported": "No reliable answer available",
}

_ACTIVITY_ICONS = {
    "clarify": "💬",
    "describe_image": "🖼️",
    "read_text": "🔤",
    "analyse_chart": "📈",
    "answer_with_image": "🔍",
    "follow_up": "🔗",
    "general_question": "📄",
    "small_talk": "💬",
    "refuse_unsupported": "⚠️",
}


def activity_label(route: str) -> str:
    """A short, human description of what the assistant did this turn."""
    return _ACTIVITY_LABELS.get(route, "Answered")


def activity_icon(route: str) -> str:
    return _ACTIVITY_ICONS.get(route, "💬")


# Language codes → a short label for the badge. Native script is used for the
# Indic languages so the badge is legible to the speaker, not just to us.
_LANGUAGE_BADGES = {
    "en": "EN · English",
    "hi": "हि · Hindi",
    "or": "ଓ · Odia",
    "bn": "বা · Bengali",
}


def language_badge(turn) -> str:
    """
    A small chip showing which language the assistant answered in — and, when
    the user switched language, that the switch was understood.

    The badge exists because multilingual behaviour is otherwise invisible: a
    correct answer in the wrong language looks like a bug, and a correct answer
    in the right language after a switch looks like luck. This makes the
    handling explicit.
    """
    report = getattr(turn, "language", None)
    if report is None:
        return ""

    response_lang = getattr(turn, "response_language", None) or getattr(report, "code", "en")
    label = _LANGUAGE_BADGES.get(response_lang, response_lang.upper())

    bits = [f'<span class="lang-chip">🌐 {label}</span>']

    if getattr(turn, "language_switched", False):
        bits.append(
            '<span class="lang-chip lang-chip-switch">⇄ language switch</span>'
        )
    if getattr(report, "mixed", False):
        others = ", ".join(
            _LANGUAGE_BADGES.get(c, c).split(" · ")[0]
            for c in (getattr(report, "mixed_languages", []) or [])
        )
        if others:
            bits.append(
                f'<span class="lang-chip lang-chip-mixed">mixed +{others}</span>'
            )

    return '<div class="lang-row">' + "".join(bits) + "</div>"


def confidence_pill(validation) -> str:
    """
    Translate a ValidationReport into a single user-facing trust pill.

    The raw statuses (pass / qualified / needs_clarification / failed) are
    engineering vocabulary. What a user needs to know is: can I rely on this,
    should I double-check it, or is the assistant asking me something?
    """
    if validation is None:
        return ""

    status = getattr(validation, "status", None)
    status_value = getattr(status, "value", str(status))

    if status_value == "needs_clarification":
        return (
            '<span class="pill pill-question">❓ Question for you</span>'
        )
    if status_value == "failed":
        return '<span class="pill pill-low">⚠️ Couldn\'t answer</span>'

    confidence = float(getattr(validation, "confidence", 0.0) or 0.0)
    if status_value == "qualified" or confidence < 0.6:
        return '<span class="pill pill-medium">⚠️ Double-check this</span>'
    if confidence >= 0.8:
        return '<span class="pill pill-high">✓ Verified</span>'
    return '<span class="pill pill-medium">✓ Answered</span>'


def quality_notice(diagnostics) -> Optional[str]:
    """
    A plain-language warning about image quality, or None.

    Diagnostics expose many numbers (sharpness, brightness, Laplacian variance).
    A user needs the *consequence*, not the measurement.
    """
    if diagnostics is None or not getattr(diagnostics, "valid", False):
        return None

    problems = []
    if getattr(diagnostics, "is_blurry", False):
        problems.append("it's a little blurry")
    if getattr(diagnostics, "is_low_resolution", False):
        problems.append("the resolution is low")
    if getattr(diagnostics, "is_too_dark", False):
        problems.append("it's quite dark")
    if getattr(diagnostics, "is_too_bright", False):
        problems.append("it's washed out")

    if not problems:
        return None
    return (
        "Your image isn't perfectly clear — "
        + " and ".join(problems)
        + " — so fine details may be harder to read."
    )


def source_names(evidence) -> list[str]:
    """
    Turn retrieved chunks into friendly document names.

    `machine_learning_basics.md-3` → "Machine learning basics".
    Multiple chunks from one document collapse into a single name.
    """
    names: list[str] = []
    for item in evidence or []:
        raw = getattr(item, "source", "") or ""
        # Drop the chunk suffix and extension.
        stem = re.sub(r"-\d+$", "", raw)
        stem = re.sub(r"\.(md|txt)$", "", stem, flags=re.IGNORECASE)
        pretty = stem.replace("_", " ").replace("-", " ").strip()
        pretty = pretty[:1].upper() + pretty[1:] if pretty else "Reference"
        if pretty not in names:
            names.append(pretty)
    return names


def source_chips(evidence) -> str:
    """Rendered HTML for the 'based on' row, or empty string if no evidence."""
    names = source_names(evidence)
    if not names:
        return ""
    chips = "".join(
        f'<span class="source-chip"><span class="dot"></span>{n}</span>'
        for n in names
    )
    return f'<div class="source-row">{chips}</div>'


def clean_answer(text: str) -> str:
    """
    Strip presentation artefacts the pipeline adds for the developer panel.

    The validator prepends an italic caveat and the pipeline may include
    markdown that reads like internal log output. The user-facing surface keeps
    the caveat (it is genuinely useful) but not the machinery around it.
    """
    if not text:
        return ""
    # Remove any leaked key=value style debug fragments on their own line.
    text = re.sub(r"^\s*route=.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*confidence=\d.*$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
