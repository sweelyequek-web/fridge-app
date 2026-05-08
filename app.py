"""
Smart Fridge Recipe Finder
==========================
Run:
    export GEMINI_API_KEY="your-key-here"
    pip install streamlit google-genai pillow
    streamlit run app.py
"""

import json
import os
from io import BytesIO

import streamlit as st
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("Missing dependency. Install with: `pip install google-genai`")
    st.stop()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-2.5-flash"

PANTRY_BASE = [
    "salt", "pepper", "soy sauce", "rice",
    "neutral cooking oil", "garlic", "onions", "water",
]

DIETARY_OPTIONS = ["None", "Vegetarian", "Gluten-free", "Dairy-free"]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

VISION_SYSTEM_PROMPT = """You are an ingredient-identification assistant for a recipe app.

Look at the photo of a fridge interior and identify visible food ingredients.

Rules:
1. Only list ingredients you can identify with high confidence. If unsure, omit it.
2. Use common, generic names (e.g., "cheddar cheese" not "Tillamook Sharp Cheddar").
3. Do NOT include items already assumed as pantry staples: salt, pepper, soy sauce, rice, oil, garlic, onions, water.
4. Do NOT include packaging, drinks, or non-food items.
5. Return a flat JSON array of lowercase strings, e.g. ["eggs", "spinach", "cheddar cheese"].
6. If the image is unclear, empty, or not a fridge, return an empty array: [].

Return ONLY the JSON array, no commentary."""


def build_recipe_prompt(ingredients: list[str], servings: int, dietary: str) -> str:
    dietary_clause = (
        "" if dietary == "None"
        else f"\n\nAll recipes MUST be {dietary.lower()}."
    )
    return f"""You are a recipe-generation assistant.

The user has these ingredients available in their fridge:
{json.dumps(ingredients)}

They also have these pantry staples (always available, NEVER list as missing):
{json.dumps(PANTRY_BASE)}

Generate EXACTLY 3 recipes for {servings} serving(s), one in each category:

1. "Quick & Easy" — total_time_min must be <= 20, steps array length <= 6.
2. "Chef's Choice" — prioritize flavor depth and technique; total time unconstrained.
3. "Waste Not" — maximize the count of fridge ingredients used. If two recipe ideas
   tie on count, prefer the one using the most perishable items (leafy greens,
   herbs, dairy, fresh proteins) over shelf-stable ones.

Rules:
- "fridge_ingredients_used" must only contain items from the user's fridge list above.
- "missing_ingredients" must EXCLUDE anything in the pantry staples list.
- Steps must be clear, numbered actions a home cook can follow.
- prep_time_min + cook_time_min should equal total_time_min.{dietary_clause}

Return a JSON object matching the provided schema."""


RECIPE_SCHEMA = {
    "type": "object",
    "properties": {
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["Quick & Easy", "Chef's Choice", "Waste Not"],
                    },
                    "title": {"type": "string"},
                    "prep_time_min": {"type": "integer"},
                    "cook_time_min": {"type": "integer"},
                    "total_time_min": {"type": "integer"},
                    "servings": {"type": "integer"},
                    "fridge_ingredients_used": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "missing_ingredients": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "category", "title", "prep_time_min", "cook_time_min",
                    "total_time_min", "servings", "fridge_ingredients_used",
                    "missing_ingredients", "steps",
                ],
            },
        }
    },
    "required": ["recipes"],
}

# ---------------------------------------------------------------------------
# Gemini API calls
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "GEMINI_API_KEY environment variable not set. "
            "Get a key at https://aistudio.google.com/apikey"
        )
        st.stop()
    return genai.Client(api_key=api_key)


def identify_ingredients(image: Image.Image) -> list[str]:
    client = get_client()
    buf = BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            "Identify the ingredients in this fridge.",
        ],
        config=types.GenerateContentConfig(
            system_instruction=VISION_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema={"type": "array", "items": {"type": "string"}},
            temperature=0.2,
        ),
    )

    try:
        result = json.loads(response.text)
        seen = set()
        cleaned = []
        for item in result:
            norm = item.strip().lower()
            if norm and norm not in seen and norm not in PANTRY_BASE:
                seen.add(norm)
                cleaned.append(norm)
        return cleaned
    except (json.JSONDecodeError, TypeError):
        return []


def generate_recipes(ingredients: list[str], servings: int, dietary: str) -> dict:
    client = get_client()
    prompt = build_recipe_prompt(ingredients, servings, dietary)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RECIPE_SCHEMA,
            temperature=0.7,
        ),
    )
    return json.loads(response.text)


# ---------------------------------------------------------------------------
# App icon (generated once at startup)
# ---------------------------------------------------------------------------

def _ensure_app_icon() -> str:
    from PIL import Image, ImageDraw

    icon_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    icon_path = os.path.join(icon_dir, "apple-touch-icon.png")

    if os.path.exists(icon_path):
        return icon_path

    os.makedirs(icon_dir, exist_ok=True)

    SIZE = 512
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=100, fill="#1F3D2D")

    fridge_w, fridge_h = int(SIZE * 0.55), int(SIZE * 0.70)
    fx, fy = (SIZE - fridge_w) // 2, int(SIZE * 0.14)
    fx2, fy2 = fx + fridge_w, fy + fridge_h

    draw.rounded_rectangle([fx - 6, fy - 6, fx2 + 6, fy2 + 6], radius=18, fill="#FFFFFF")
    draw.rounded_rectangle([fx, fy, fx2, fy2], radius=12, fill="#F0EBE3")

    fz_h = int(fridge_h * 0.30)
    draw.rounded_rectangle([fx, fy, fx2, fy + fz_h], radius=12, fill="#D6EAF8")
    draw.rectangle([fx, fy + fz_h - 3, fx2, fy + fz_h + 3], fill="#FFFFFF")

    shelf_section = fridge_h - fz_h
    for i in (1, 2):
        sy = fy + fz_h + int(shelf_section * i / 3)
        draw.rectangle([fx + 12, sy - 3, fx2 - 12, sy + 3], fill="#FFFFFF")

    hx = fx2 - 20
    ht = fy + fz_h + int(shelf_section * 0.2)
    hb = ht + int(shelf_section * 0.3)
    draw.rounded_rectangle([hx, ht, hx + 10, hb], radius=5, fill="#FFFFFF")

    dot_cx, dot_cy, dot_r = int(SIZE * 0.734), int(SIZE * 0.22), 44
    draw.ellipse([dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r], fill="#E76F51")

    final = Image.new("RGB", (SIZE, SIZE), "#1F3D2D")
    final.paste(img, mask=img.split()[3])
    final.save(icon_path, "PNG", optimize=True)

    return icon_path


_APP_ICON_PATH = _ensure_app_icon()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Fridge Recipe Finder",
    page_icon=_APP_ICON_PATH,
    layout="centered",
)

st.markdown(
    """
<link rel="apple-touch-icon" sizes="512x512" href="/app/static/apple-touch-icon.png">

<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

div[data-testid="stAppViewContainer"] {
    background-color: #FAF7F2;
}

div[data-testid="stDecoration"] {
    display: none;
}

div[data-testid="stHeader"] {
    background: transparent;
}

.main .block-container {
    max-width: 760px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: #1F3D2D !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.25rem !important;
}

h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #1F3D2D !important;
    border-left: 4px solid #2D6A4F;
    padding-left: 0.65rem;
    margin-top: 1.8rem !important;
    margin-bottom: 0.5rem !important;
}

p, li, .stMarkdown, label, .stCaption {
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #374151;
}

.stCaption, small {
    color: #6B7280 !important;
    font-size: 0.875rem !important;
}

.stButton > button {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-weight: 600;
    font-size: 0.95rem;
    border-radius: 50px;
    padding: 0.6rem 2rem;
    border: none !important;
    background: linear-gradient(135deg, #2D6A4F 0%, #52B788 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(45, 106, 79, 0.30);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(45, 106, 79, 0.40) !important;
    background: linear-gradient(135deg, #245c43 0%, #40a875 100%) !important;
}

.stButton > button:active {
    transform: translateY(0px);
    box-shadow: 0 2px 6px rgba(45, 106, 79, 0.25) !important;
}

.stButton > button:disabled {
    background: #C8D8C8 !important;
    box-shadow: none;
    transform: none;
    cursor: not-allowed;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed #2D6A4F;
    border-radius: 16px;
    background-color: #F4F9F6;
    padding: 1rem;
    transition: border-color 0.2s ease, background-color 0.2s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #52B788;
    background-color: #EAF4EE;
}

div[data-testid="stFileUploader"] label {
    color: #2D6A4F !important;
    font-weight: 500;
}

.stTextArea textarea {
    border-radius: 12px !important;
    border: 1.5px solid #E8DDD0 !important;
    background-color: #FDFAF6 !important;
    color: #1F3D2D !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.stTextArea textarea:focus {
    border-color: #2D6A4F !important;
    box-shadow: 0 0 0 3px rgba(45, 106, 79, 0.15) !important;
    outline: none !important;
}

button[data-baseweb="tab"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 50px !important;
    padding: 0.4rem 1.2rem !important;
    color: #6B7280 !important;
    background-color: transparent !important;
    border: 1.5px solid #E8DDD0 !important;
    transition: all 0.2s ease;
}

button[data-baseweb="tab"]:hover {
    border-color: #2D6A4F !important;
    color: #2D6A4F !important;
    background-color: #F4F9F6 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #2D6A4F 0%, #52B788 100%) !important;
    color: #FFFFFF !important;
    border-color: transparent !important;
    box-shadow: 0 3px 10px rgba(45, 106, 79, 0.30);
}

div[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border-radius: 14px;
    border: 1px solid #E8DDD0;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.07);
    padding: 1rem 1.25rem !important;
    text-align: center;
}

div[data-testid="metric-container"] label {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6B7280 !important;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #2D6A4F !important;
}

div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

div[data-testid="stImage"] img {
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.10);
    border: 1px solid #E8DDD0;
}

code {
    background-color: #EAF4EE !important;
    color: #1F3D2D !important;
    border-radius: 6px !important;
    padding: 2px 7px !important;
    font-size: 0.88em !important;
    border: 1px solid #C8DDD0 !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F0EBE3; }
::-webkit-scrollbar-thumb { background: #2D6A4F; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #52B788; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🍳 Smart Fridge Recipe Finder")
st.caption("Upload a fridge photo, confirm what's inside, and get three recipes.")

# Session state
if "ingredients" not in st.session_state:
    st.session_state.ingredients = []
if "recipes" not in st.session_state:
    st.session_state.recipes = None
if "scan_done" not in st.session_state:
    st.session_state.scan_done = False

# --- Step 1: Upload ---
st.header("1. Upload a fridge photo")
uploaded_file = st.file_uploader(
    "Take or upload a clear photo of your fridge interior",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your fridge", use_container_width=True)

    if st.button("🔍 Identify ingredients", type="primary"):
        with st.spinner("Looking inside your fridge..."):
            try:
                detected = identify_ingredients(image)
                st.session_state.ingredients = detected
                st.session_state.scan_done = True
                st.session_state.recipes = None
                if not detected:
                    st.warning(
                        "Couldn't identify any ingredients. "
                        "Try a clearer photo, or add ingredients manually below."
                    )
            except Exception as e:
                st.error(f"Vision API error: {e}")
                st.session_state.scan_done = True

# --- Step 2: Inventory ---
if st.session_state.scan_done or st.session_state.ingredients:
    st.header("2. Confirm your ingredients")
    st.caption(
        f"Pantry staples ({', '.join(PANTRY_BASE)}) are already assumed."
    )

    ingredients_text = st.text_area(
        "Ingredients (one per line)",
        value="\n".join(st.session_state.ingredients),
        height=200,
        help="Add, remove, or correct ingredients. One per line.",
    )

    current_ingredients = [
        line.strip().lower()
        for line in ingredients_text.split("\n")
        if line.strip()
    ]
    st.session_state.ingredients = current_ingredients

    # --- Step 3: Preferences ---
    st.header("3. Recipe preferences")
    col1, col2 = st.columns(2)
    with col1:
        servings = st.slider("Servings", 1, 6, 2)
    with col2:
        dietary = st.selectbox("Dietary filter", DIETARY_OPTIONS)

    if st.button("👨‍🍳 Generate recipes", type="primary"):
        if not current_ingredients:
            st.error("Please add at least one ingredient before generating recipes.")
        else:
            with st.spinner("Crafting three recipes..."):
                try:
                    st.session_state.recipes = generate_recipes(
                        current_ingredients, servings, dietary
                    )
                except json.JSONDecodeError:
                    st.error("The model returned invalid JSON. Please try again.")
                except Exception as e:
                    st.error(f"Recipe generation failed: {e}")

# --- Step 4: Recipe cards ---
if st.session_state.recipes:
    st.header("4. Your recipes")
    recipes = st.session_state.recipes.get("recipes", [])
    category_order = ["Quick & Easy", "Chef's Choice", "Waste Not"]
    icons = {"Quick & Easy": "⏱️", "Chef's Choice": "⭐", "Waste Not": "♻️"}

    recipes_sorted = sorted(
        recipes,
        key=lambda r: category_order.index(r["category"])
        if r["category"] in category_order else 99,
    )

    tabs = st.tabs([
        f"{icons.get(r['category'], '🍽️')} {r['category']}"
        for r in recipes_sorted
    ])

    for tab, recipe in zip(tabs, recipes_sorted):
        with tab:
            st.subheader(recipe["title"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Prep", f"{recipe['prep_time_min']} min")
            c2.metric("Cook", f"{recipe['cook_time_min']} min")
            c3.metric("Total", f"{recipe['total_time_min']} min")

            st.markdown(f"**Servings:** {recipe['servings']}")

            st.markdown("**From your fridge:**")
            if recipe["fridge_ingredients_used"]:
                st.markdown(" · ".join(
                    f"`{i}`" for i in recipe["fridge_ingredients_used"]
                ))
            else:
                st.markdown("_None — uses only pantry staples._")

            st.markdown("**Missing ingredients (need to buy):**")
            if recipe["missing_ingredients"]:
                st.markdown(" · ".join(
                    f"`{i}`" for i in recipe["missing_ingredients"]
                ))
            else:
                st.success("Nothing! You have everything you need. 🎉")

            st.markdown("**Steps:**")
            for idx, step in enumerate(recipe["steps"], 1):
                st.markdown(f"{idx}. {step}")
