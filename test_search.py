from recommendation_engine import search_web_official
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing search_web_official...")
result = search_web_official("cute date ideas in San Francisco")
print("Result:")
print(result)

if "Error: GOOGLE_API_KEY" in result:
    print("\nSUCCESS: Code correctly identified missing API keys.")
elif "Error performing search" in result:
    print("\nFAILURE: Unexpected error occurred.")
else:
    print("\nSUCCESS: Search returned results (or keys were actually present).")
