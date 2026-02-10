
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import markdown

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def load_interests():
    try:
        with open("interests.md", "r", encoding="utf-8") as f:
            content = f.read()
            html = markdown.markdown(content)
            return html
    except FileNotFoundError:
        return "<p>No interests found. Please run analysis first.</p>"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    interests_html = load_interests()
    return templates.TemplateResponse("index.html", {"request": request, "interests": interests_html})

from recommendation_engine import generate_date_ideas

from datetime import datetime
 
@app.get("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, lat: float = None, long: float = None, city: str = None):
    # If city is provided (fallback), use it. If lat/long provided, use them.
    current_time = datetime.now()
    try:
        with open("interests.md", "r", encoding="utf-8") as f:
            interests_text = f.read()
    except:
        interests_text = "General cute things."

    recommendations_html = generate_date_ideas(lat, long, city, interests_text, current_time)
    # Convert markdown to html if the engine returned markdown, 
    # but I asked for HTML in the prompt. Let's assume HTML.
    # Actually, it's safer to run it through markdown just in case Gemini ignored instructions,
    # but the prompt asked for specific DIVs.
    
    import markdown
    # If Gemini returns markdown formatted HTML (like wrapped in ```html), strip it.
    cleaned_html = recommendations_html.replace("```html", "").replace("```", "")
    
    location_label = city if city else f"{lat}, {long}"
    
    return templates.TemplateResponse("recommend.html", {"request": request, "city": location_label, "recommendations": cleaned_html})
