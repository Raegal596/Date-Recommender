
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from googlesearch import search

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

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
    2.  **Search**: YOU MUST USE the `search_web` tool to find SPECIFIC, REAL places/events.
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
        model='gemini-2.0-flash',
        config=types.GenerateContentConfig(
            tools=[search_web],
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
                if call.name == "search_web":
                    args = call.args
                    query = args.get("query") 
                    
                    result = search_web(query)
                    function_responses.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name="search_web",
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
                 for part in response.candidates[0].content.parts:
                     if part.text:
                         text_response = part.text
                         break
        except:
            pass

    if text_response is None:
        return "<div class='error'>Sorry, I couldn't generate recommendations at this time. Please try again.</div>"
    
    return text_response

