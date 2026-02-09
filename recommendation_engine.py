
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

def generate_date_ideas(city, interests_summary):
    """
    1. Uses Gemini to brainstorm 3 specific search queries based on interests + city.
    2. (Simulator) In a real app we'd run these queries. Here we will ask Gemini to 
       hallucinate plausible plans or we could try to use a search tool if available.
       
    Wait, I HAVE a search tool! I should use it.
    But I can't call tools from this script running in the server.
    
    Alternative: The server will just use Gemini to generate *generic* ideas tailored to the city,
    OR I can try to use a simple search library if I installed one (like duckduckgo-search).
    
    Let's stick to Gemini generation for now as per the "Simple Web App" scope, 
    but if the user wants *actual* events, I really need real data.
    
    Since I cannot easily invoke the `search_web` tool from inside the user's running python process 
    (unless I build a bridge), I will use `googlesearch-python` or similar if I can install it.
    
    Let's check if I can genericize it.
    
    Hypothesis: The user wants *specific* places. 
    I will write a prompt that asks Gemini to recommend places in that city 
    (it has training data up to a certain point).
    """
    
    prompt = f"""
    Based on the following user interests:
    {interests_summary}
    
    And the city: {city}
    
    Create 3 distinct, cute, and personalized date itineraries. 
    For each itinerary:
    1. Give it a cute title.
    2. Suggest SPECIFIC real places in {city} if you know them (e.g., specific highly rated parks, restaurants).
    3. Explain why it fits our interests.
    
    Format the output as HTML cards. 
    Use <div class="date-card">...</div> for each.
    Inside, use <h3>Title</h3>, <p>Description...</p>.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"<p>Error generating dates: {e}</p>"
