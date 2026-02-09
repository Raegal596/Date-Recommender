
import os
import zipfile
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)
# Using the newer model
model = genai.GenerativeModel('gemini-2.5-flash')

def extract_chat_text(chat_dir):
    """
    Extracts text from all _chat.txt files within zip files in the chat_dir.
    Returns a combined string of chat history.
    """
    combined_text = ""
    zip_files = [f for f in os.listdir(chat_dir) if f.endswith('.zip')]
    
    for zip_file in zip_files:
        print(f"Processing {zip_file}...")
        try:
            with zipfile.ZipFile(os.path.join(chat_dir, zip_file), 'r') as z:
                for file_info in z.infolist():
                    if file_info.filename.endswith('.txt'):
                        with z.open(file_info) as f:
                            # WhatsApp exports are usually UTF-8
                            text = f.read().decode('utf-8', errors='ignore')
                            combined_text += text + "\n"
        except Exception as e:
            print(f"Error processing {zip_file}: {e}")
            
    return combined_text

def analyze_interests(chat_text):
    """
    Uses Gemini to analyze the chat text and extract shared interests.
    """
    print("Analyzing chat history with Gemini...")
    
    # Check if chat text is empty
    if not chat_text.strip():
        return "No chat history found to analyze."

    # Prompt Engineering
    prompt = f"""
    Analyze the following WhatsApp chat history between "Me" and my partner.
    Identify and list our shared interests, hobbies, food preferences, and preferred activities.
    Focus on extracting potential date ideas or things we enjoy doing together.
    
    Format the output as a Markdown list with categories (e.g., ## Food, ## Activities, ## Hobbies).
    Be specific (e.g., instead of just "Food", specify "Sushi", "Italian", "Spicy food").
    Keep it concise and relevant for a date recommendation app.
    
    Chat History:
    """ 
    

    try:
        # 2.5 flash has a limit. Truncate to the last 500,000 characters to be safe and stay well under 1M tokens/minute limits
        # effectively getting the most recent conversation history.
        truncated_text = chat_text[-500000:] 
        print(f"Sending {len(truncated_text)} characters to Gemini...")
        
        response = model.generate_content(prompt + "\n\n" + truncated_text)
        return response.text
    except Exception as e:
        return f"Error gathering interests: {e}"

def main():
    chat_dir = 'chat_history'
    if not os.path.exists(chat_dir):
        print(f"Directory {chat_dir} not found.")
        return

    chat_text = extract_chat_text(chat_dir)
    print(f"Extracted {len(chat_text)} characters of chat history.")
    
    interests = analyze_interests(chat_text)
    
    output_file = 'interests.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(interests)
    
    print(f"Analysis complete. Interests saved to {output_file}")
    # Print a preview
    print("-" * 20)
    print(interests[:500] + "...")
    print("-" * 20)

if __name__ == "__main__":
    main()
