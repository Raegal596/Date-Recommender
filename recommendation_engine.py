
import os
import concurrent.futures
from google import genai
from google.genai import types
from dotenv import load_dotenv
from googlesearch import search
from googleapiclient.discovery import build

# LangChain / LangGraph imports
from typing import TypedDict, List, Dict, Any, Optional
import json
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import re

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def extract_json(text: Any) -> Any:
    """
    Robustly extracts JSON from a string, handling Markdown code blocks and surrounding text.
    """
    if isinstance(text, list):
        text_parts = []
        for p in text:
            if isinstance(p, str):
                text_parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                text_parts.append(p["text"])
            elif hasattr(p, "text"):
                text_parts.append(getattr(p, "text"))
            else:
                text_parts.append(str(p))
        text = "".join(text_parts)
    elif not isinstance(text, str):
        text = str(text)
        
    text = text.strip()
    
    # Try to find JSON block in markdown
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1)
    
    # Try to clean up if it's just a raw list or dict with extra text around it
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # If failed, try to find the substring
    # This regex looks for the outermost {} or [] pair
    # It's a heuristic and might fail on nested structures if not careful, but works for most LLM outputs
    
    # Find first '[' or '{'
    start_idx = -1
    
    first_brace = text.find('{')
    first_bracket = text.find('[')
    
    if first_brace == -1 and first_bracket == -1:
         # No JSON structure found
         raise ValueError(f"No JSON structure found in text: {text[:50]}...")
         
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_char = '{'
        end_char = '}'
        start_idx = first_brace
    else:
        start_char = '['
        end_char = ']'
        start_idx = first_bracket
        
    # Find the matching closing bracket
    depth = 0
    for i in range(start_idx, len(text)):
        char = text[i]
        j = i
        if char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                json_str = text[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                     # Continue searching if this one failed
                     pass 
    
    # Fallback: simple trimming to last occurrence
    last_brace = text.rfind(end_char)
    if last_brace != -1:
         json_str = text[start_idx:last_brace+1]
         try:
             return json.loads(json_str)
         except:
             pass

    raise ValueError(f"Could not extract JSON from text: {text[:100]}...")

def search_web(query: str) -> str:
    """
    Performs a web search for the given query and returns the top results.
    """
    try:
        results = []
        # advanced=True yields objects with title, description, url
        for result in search(query, num_results=5, advanced=True):
            results.append(f"Title: {result.title}\nURL: {result.url}\nDescription: {result.description}\n")
        final_result = "\n---\n".join(results)
        
        import time
        time.sleep(1) # Prevent tight loops
        
        if not final_result.strip():
            return "No results found for this query."

        if len(final_result) > 2000:
             # print("DEBUG: Truncating search result...")
             final_result = final_result[:2000] + "... (truncated)"
        return final_result
    except Exception as e:
        print(f"Error executing search: {e}")
        return f"Error performing search: {e}"

def search_web_official(query: str) -> str:
    """
    Performs a web search using the official Google Custom Search JSON API.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if not api_key or not cse_id:
            # Fallback to the old search if keys are missing, or return error?
            # User specifically asked for official API, so let's error or warn.
            return "Error: GOOGLE_API_KEY or GOOGLE_CSE_ID not found in environment variables. Please add them to .env."

        service = build("customsearch", "v1", developerKey=api_key)
        # cse() returns a Resource, list() executes the search
        res = service.cse().list(q=query, cx=cse_id, num=5).execute()
        
        search_items = res.get("items", [])
        if not search_items:
            return "No results found for this query."
            
        formatted_results = []
        for item in search_items:
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet")
            formatted_results.append(f"Title: {title}\nURL: {link}\nDescription: {snippet}\n")
            
        return "\n---\n".join(formatted_results)

    except Exception as e:
        print(f"Error executing official search: {e}")
        return f"Error performing search: {e}"

def generate_date_ideas(lat, long, city, interests_summary, current_time):
    """
    Uses Gemini with a search tool to plan and generate date ideas.
    """
    
    # print("DEBUG: generate_date_ideas called")
    location_str = f"Latitude: {lat}, Longitude: {long}" if lat and long else f"City: {city}"
    
    prompt = f"""
    Based on the following user interests:
    {interests_summary}
    
    User Location: {location_str}
    Current Time: {current_time}
    
    GOAL: Create 3 distinct, cute, and personalized date itineraries.
    
    PROCESS:
    1.  **Analyze & Plan**: Analyze the interests, location, and time. Plan 3 potential date concepts.
    2.  **Search**: YOU MUST USE the `search_web_official` tool to find SPECIFIC, REAL places/events.
        -   Search for "events in {city} today" or "best [INTEREST] in {city}".
        -   Verify they are open at {current_time}.
    3.  **Synthesize**: Generate the final HTML output.
    
    CONSTRAINTS:
    -   Do NOT make up places.
    -   If you don't use the search tool, your answer will be rejected.
    
    OUTPUT FORMAT:
    Return ONLY the HTML string starting with <div class="date-card">...
    """

    # Create a chat session with the tool
    # We configure the chat with the tool, but we will override the config for the first message
    chat = client.chats.create(
        model='gemini-3-flash-preview',
        config=types.GenerateContentConfig(
            tools=[search_web_official],
            temperature=0.7,
            system_instruction="You are a helpful date planning assistant. You have access to a Google Search tool. You MUST use it to verify information and find real places. Do not guess.",
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode='ANY'
                )
            )
        )
    )

    # Initial request - FORCE the model to use the search tool
    # print("DEBUG: Sending initial message to model...")
    response = chat.send_message(prompt)
    
    try:
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
             pass 
    except Exception as e:
        pass

    # Simple tool execution loop

    # Simple tool execution loop (up to 5 turns)
    for i in range(5):
        try:
            if not response.function_calls:
                break
                
            # Execute all function calls
            function_responses = []
            for call in response.function_calls:
                if call.name == "search_web_official":
                    args = call.args
                    query = args.get("query") 
                    
                    result = search_web_official(query)
                    function_responses.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name="search_web_official",
                                response={"result": result}
                            )
                        )
                    )

            # Send results back to the model
            if function_responses:
                response = chat.send_message(function_responses)
            else:
                break
        except Exception as e:
             print(f"DEBUG: Error in tool loop: {e}", flush=True)
             import traceback
             traceback.print_exc()
             break
    
    # If the model is still trying to call functions after the loop, force it to summarize
    if response.function_calls:
        try:
            response = chat.send_message("You have performed enough searches. Please generate the final date itineraries now based on the information you have. Do not search anymore.")
        except Exception as e:
            pass

    text_response = response.text
    if text_response is None:
        # Try to extract text manually if possible, or return a fallback
        try:
             if response.candidates and response.candidates[0].content.parts:
                 text_parts = []
                 for part in response.candidates[0].content.parts:
                     if part.text:
                         text_parts.append(part.text)
                 if text_parts:
                     text_response = "".join(text_parts)
        except:
            pass

    if text_response is None:
        return "<div class='error'>Sorry, I couldn't generate recommendations at this time. Please try again.</div>"
    
    return text_response

# -------------------------------------------------------------------------
# Agentic Pipeline Implementation
# -------------------------------------------------------------------------

# Load Tavily API Key
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

# Initialize the model for the agent
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key)

class AgentState(TypedDict):
    city: str
    lat: Optional[float]
    long: Optional[float]
    current_time: str
    interests_summary: str
    weather_info: str
    local_event_sources: Dict[str, List[str]] # City -> List of URLs
    search_plan: List[str]
    date_ideas: List[Dict[str, Any]]
    final_html: str
    messages: List[Any]

# --- Tools ---

@tool
def check_weather(city: str) -> str:
    """Checks the current weather for a given city."""
    try:
        # Simple search for weather
        response = tavily_client.search(query=f"current weather in {city}", search_depth="basic")
        results = response.get("results", [])
        if results:
            return results[0].get("content", "Weather data not found.")
        return "Weather data not found."
    except Exception as e:
        return f"Error checking weather: {e}"

@tool
def find_local_event_sites(city: str) -> List[str]:
    """Finds websites that list local events for a given city."""
    try:
        query = f"best websites for events in {city} today"
        response = tavily_client.search(query=query, search_depth="basic")
        results = response.get("results", [])
        urls = [r.get("url") for r in results[:3]]
        return urls
    except Exception as e:
        print(f"Error finding event sites: {e}")
        return []

@tool
def search_dates_tavily(query: str) -> str:
    """Searches for date ideas using Tavily."""
    try:
        response = tavily_client.search(query=query, search_depth="advanced")
        context = "\n".join([r.get("content", "") for r in response.get("results", [])])
        return context
    except Exception as e:
        return f"Error searching dates: {e}"

# --- Nodes ---

def load_interests(state: AgentState):
    """Loads interests from the markdown file."""
    try:
        with open("interests.md", "r") as f:
            interests = f.read()
    except:
        interests = "No interests file found."
    return {"interests_summary": interests}

def analyze_interests_node(state: AgentState):
    """Analyzes interests to generate a search plan."""
    print("--- Analyzing Interests ---")
    prompt = f"""
    Analyze the following interests and the current context to generate 20 specific search queries for date ideas.
    Consider the user's location ({state['city']}) and the current time ({state['current_time']}).
    
    Interests:
    {state['interests_summary']}
    
    Output a JSON list of strings, e.g., ["query 1", "query 2", "query 3"].
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    
    # Handle list content from LangChain (can be list of parts or strings)
    if isinstance(content, list):
        text_parts = []
        for p in content:
            if isinstance(p, str):
                text_parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                text_parts.append(p["text"])
            elif hasattr(p, "text"):
                text_parts.append(p.text)
            else:
                 # Fallback: try to guess or just stringify
                text_parts.append(str(p))
        content = "".join(text_parts)
        
    content = content.strip()

    # Clean up markdown code blocks if present
    # if content.startswith("```json"):
    #     content = content[7:-3]
    # elif content.startswith("```"):
    #     content = content[3:-3]
    
    print(f"Draft Search Plan: {content}")

    try:
        # import json # Already imported globally
        search_plan = extract_json(content)
        # Verify it's a list
        if not isinstance(search_plan, list):
             # If it's a dict like {"plan": [...]}, try to extract extraction
             if isinstance(search_plan, dict):
                 search_plan = list(search_plan.values())[0] # Very naive fallback
             if not isinstance(search_plan, list):
                 raise ValueError("Extracted JSON is not a list")
    except Exception as e:
        print(f"Error parsing search plan: {e}")
        search_plan = [f"events in {state['city']} today", f"restaurants in {state['city']}", f"activities in {state['city']}"]
        
    print(f"Search Plan: {search_plan}")
    return {"search_plan": search_plan}

def check_weather_node(state: AgentState):
    """Checks weather and updates state."""
    print("--- Checking Weather ---")
    city = state["city"]
    # We call the tool function directly or via invoke if it's a Tool object
    # Since we decorated with @tool, check_weather is a StructuredTool
    try:
        weather = check_weather.invoke({"city": city})
    except:
        weather = "Weather check failed."
        
    print(f"Weather: {weather}")
    return {"weather_info": weather}

def manage_local_sources_node(state: AgentState):
    """Manages local event sources JSON file."""
    print("--- Managing Local Sources ---")
    city = state["city"]
    file_path = "local_events.json"
    
    sources = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                sources = json.load(f)
        except:
            pass
            
    city_sources = sources.get(city, [])
    
    if not city_sources:
        print(f"No sources found for {city}, searching...")
        # Search for sources
        try:
            new_sources = find_local_event_sites.invoke({"city": city})
            if new_sources:
                sources[city] = new_sources
                city_sources = new_sources
                # Save back to file
                with open(file_path, "w") as f:
                    json.dump(sources, f, indent=2)
        except Exception as e:
            print(f"Error finding sources: {e}")
    else:
        print(f"Found cached sources for {city}: {city_sources}")
        
    return {"local_event_sources": sources}

def search_dates_node(state: AgentState):
    """Executes the search plan."""
    print("--- Searching for Dates ---")
    search_plan = state["search_plan"]
    city = state["city"]
    
    results = []
    
    def fetch_result(query):
        full_query = f"{query} in {city}"
        print(f"Searching: {full_query}")
        try:
            search_result = search_dates_tavily.invoke({"query": full_query})
            return f"Query: {query}\nResult: {search_result}"
        except:
            return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_result, query) for query in search_plan]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
        
    combined_results = "\n\n".join(results)
    return {"messages": [HumanMessage(content=f"Search Results:\n{combined_results}")]}

def assessment_node(state: AgentState):
    """Assesses results and generates final output."""
    print("--- Assessing Results ---")
    weather = state["weather_info"]
    current_time = state["current_time"]
    try:
        search_results = state["messages"][-1].content
    except:
        search_results = "No search results."
        
    interests = state["interests_summary"]
    
    prompt = f"""
    You are a romantic date planner.
    
    Context:
    - Location: {state['city']}
    - Time: {current_time}
    - Weather: {weather}
    - User Interests: {interests}
    
    Search Results:
    {search_results}
    
    Task:
    1. Select the best 3 date ideas based on the interests, weather, and practicality.
    2. If the weather is bad (rain/snow), prioritize indoor activities.
    3. Ensure the places are likely open at {current_time}.
    
    Output:
    You MUST output ONLY a valid JSON object with the following structure:
    {{
        "final_html": "<div class='date-card'>...</div>",
        "date_ideas": [
            {{"title": "Idea Name", "description": "Brief summary", "reasoning": "Why it fits"}}
        ]
    }}
    
    The 'final_html' should contain 3 <div class="date-card"> elements. 
    Each card should have a title, description, and reasoning why it fits.
    Style it beautifully with inline CSS if needed, but the class "date-card" is expected.
    
    If you cannot find 3 good distinct ideas, you can generate some new ideas to search 
    and call the search_dates node again. Call search_dates_node() a maximum
    of 2 times before generating the final output.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    
    # Handle list content from LangChain (can be list of parts or strings)
    if isinstance(content, list):
        text_parts = []
        for p in content:
            if isinstance(p, str):
                text_parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                text_parts.append(p["text"])
            elif hasattr(p, "text"):
                text_parts.append(p.text)
            else:
                 # Fallback: try to guess or just stringify
                text_parts.append(str(p))
        content = "".join(text_parts)
    
    # Try parsing JSON
    try:
        parsed = extract_json(content)
        final_html = parsed.get("final_html", "")
        date_ideas = parsed.get("date_ideas", [])
    except Exception as e:
        print(f"Error parsing assessment output: {e}")
        # Fallback
        if "```html" in content:
            final_html = content.split("```html")[1].split("```")[0].strip()
        elif "```" in content:
            final_html = content.replace("```", "").strip()
        else:
            final_html = content
        date_ideas = []
        
    return {"final_html": final_html, "date_ideas": date_ideas}

# --- Graph ---

workflow = StateGraph(AgentState)

workflow.add_node("load_interests", load_interests)
workflow.add_node("analyze_interests", analyze_interests_node)
workflow.add_node("check_weather", check_weather_node)
workflow.add_node("manage_local_sources", manage_local_sources_node)
workflow.add_node("search_dates", search_dates_node)
workflow.add_node("assessment", assessment_node)

workflow.set_entry_point("load_interests")
workflow.add_edge("load_interests", "check_weather")
workflow.add_edge("check_weather", "manage_local_sources")
workflow.add_edge("manage_local_sources", "analyze_interests")
workflow.add_edge("analyze_interests", "search_dates")
workflow.add_edge("search_dates", "assessment")
#workflow.add_edge("assessment", "search_dates")
workflow.add_edge("assessment", END)

app_agent = workflow.compile()

from langsmith import traceable

@traceable(name="date_recommender_agent")
def generate_date_ideas_agentic(lat, long, city, current_time, return_full_state=False):
    """
    Wrapper function to run the agentic pipeline.
    """
    initial_state = {
        "city": city,
        "lat": lat,
        "long": long,
        "current_time": current_time
    }
    
    try:
        result = app_agent.invoke(initial_state)
        if return_full_state:
            return result
        return result["final_html"]
    except Exception as e:
        print(f"Error in agentic pipeline: {e}")
        if return_full_state:
            return {"final_html": f"<div class='error'>Error: {e}</div>", "error": str(e)}
        return f"<div class='error'>Error generating recommendations: {e}</div>"


