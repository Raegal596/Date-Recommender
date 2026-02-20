import os
import sys
import json
from dotenv import load_dotenv
from typing import Dict, Any

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

# Load env vars
load_dotenv()

from recommendation_engine import generate_date_ideas_agentic, extract_json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Initialize Evaluator LLM
api_key = os.getenv("GEMINI_API_KEY")
eval_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key)

LOCATIONS = [
    {"city": "Edmonton", "lat": 53.5461, "long": -113.4938},
    {"city": "Vancouver", "lat": 49.2827, "long": -123.1207},
    {"city": "New York", "lat": 40.7128, "long": -74.0060},
    {"city": "Tokyo", "lat": 35.6762, "long": 139.6503},
    {"city": "London", "lat": 51.5074, "long": -0.1278},
    {"city": "Paris", "lat": 48.8566, "long": 2.3522},
    {"city": "Sydney", "lat": -33.8688, "long": 151.2093},
    {"city": "Austin", "lat": 30.2672, "long": -97.7431},
    {"city": "Berlin", "lat": 52.5200, "long": 13.4050},
    {"city": "Toronto", "lat": 43.6532, "long": -79.3832}
]

def evaluate_run(location: Dict[str, Any], result_state: Dict[str, Any]):
    """
    Evaluates a single run for faithfulness and relevance.
    """
    city = location["city"]
    interests = result_state.get("interests_summary", "N/A")
    output_html = result_state.get("final_html", "")
    
    # Extract search context from messages
    search_context = ""
    messages = result_state.get("messages", [])
    for msg in messages:
        if isinstance(msg, HumanMessage) and "Search Results" in str(msg.content):
             search_context += msg.content
    
    if not search_context:
        search_context = "No search results found in state."

    prompt = f"""
    You are an expert evaluator for a date recommendation agent.
    
    Input Data:
    - User Interests: {interests[:500]}... (truncated)
    - Location: {city}
    - Search Context (Ground Truth): 
      {search_context[:2000]}... (truncated)
    
    Agent Output:
    {output_html}
    
    Task:
    Evaluate the Agent Output on two metrics:
    1. **Faithfulness**: Are the recommended places/events real and supported by the Search Context? Do the details (opening times, locations) match? (Score 1-5)
    2. **Relevance**: Do the recommendations actually align with the User Interests and the Location? (Score 1-5)
    
    Output JSON ONLY:
    {{
        "faithfulness_score": int,
        "faithfulness_reasoning": "string",
        "relevance_score": int,
        "relevance_reasoning": "string"
    }}
    """
    
    try:
        response = eval_llm.invoke(prompt)
        content = response.content
        return extract_json(content)
    except Exception as e:
        return {
            "faithfulness_score": 0,
            "faithfulness_reasoning": f"Error: {e}",
            "relevance_score": 0,
            "relevance_reasoning": f"Error: {e}"
        }

def run_evaluation():
    print(f"Starting Evaluation on {len(LOCATIONS)} locations...")
    current_time = "2026-02-15 12:00:00" # Fixed time for consistency
    
    results = []
    
    for loc in LOCATIONS:
        print(f"\nProcessing {loc['city']}...")
        try:
            # Run Agent
            state = generate_date_ideas_agentic(
                loc["lat"], loc["long"], loc["city"], current_time, return_full_state=True
            )
            
            # Evaluate
            eval_result = evaluate_run(loc, state)
            
            result_entry = {
                "location": loc["city"],
                "scores": eval_result
            }
            results.append(result_entry)
            
            print(f"Scores for {loc['city']}: Faithfulness={eval_result['faithfulness_score']}, Relevance={eval_result['relevance_score']}")
            
        except Exception as e:
            print(f"Error processing {loc['city']}: {e}")

    # Calculate Average
    avg_faith = sum(r["scores"]["faithfulness_score"] for r in results) / len(results) if results else 0
    avg_rel = sum(r["scores"]["relevance_score"] for r in results) / len(results) if results else 0
    
    print("\n--- Evaluation Summary ---")
    print(f"Average Faithfulness: {avg_faith:.2f}/5")
    print(f"Average Relevance: {avg_rel:.2f}/5")
    
    # Save detailed results
    with open("eval_results_gemmini_3_flash_20_search.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()
