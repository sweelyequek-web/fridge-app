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
    api_key = os.environ.get("GEMINI_API_KEY")
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
# UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Fridge Recipe Finder",
    page_icon="🍳",
    layout="centered",
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
