import json
import os
import time
import google.generativeai as genai
from pydantic import BaseModel, Field
from composio import Composio
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

class AppResearch(BaseModel):
    app_name: str
    category: str
    description_one_liner: str
    auth_method: str
    access_type: str
    api_surface: str
    webhooks_supported: bool
    buildability_verdict: str
    evidence_url: str

def research_app(app_name: str, hint_url: str) -> dict:
    """Uses Gemini 1.5/Pro to analyze the requested app integration"""
    model = genai.GenerativeModel('gemini-pro-latest')
    
    prompt = f"""
    You are an AI Product Ops Researcher for Composio. Research the app '{app_name}' (hints: {hint_url}).
    Output strictly a RAW JSON object using this schema:
    {{
      "app_name": "appname",
      "category": "category string",
      "description_one_liner": "1 line description",
      "auth_method": "OAuth2 / API Key / Gated / etc",
      "access_type": "Self-serve or Gated",
      "api_surface": "REST / GraphQL",
      "webhooks_supported": true/false,
      "buildability_verdict": "Your buildability score/roadblock",
      "evidence_url": "Docs URL"
    }}
    Do not output markdown block ticks, just raw JSON text.
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        print(f"✅ Processed {app_name}")
        return data
    except Exception as e:
        print(f"❌ Failed on {app_name}: {e}")
        return {
            "app_name": app_name,
            "category": "Requires Human Verification",
            "description_one_liner": "Failed parsing or searching.",
            "auth_method": "Needs Human", "access_type": "Unknown", "api_surface": "Unknown",
            "webhooks_supported": False, "buildability_verdict": "Manual verification needed.",
            "evidence_url": hint_url
        }

def main():
    print("🚀 Initializing Composio Agent Architecture...")
    # Composio architecture integration point
    composio_client = Composio(api_key=COMPOSIO_API_KEY)
    
    with open("apps_list.json", "r") as f:
        apps = json.load(f)
        
    results = []
    print(f"Processing {len(apps)} applications...")
    
    for app in apps:
        result = research_app(app["name"], app["hint_url"])
        results.append(result)
        time.sleep(1) # Protect Gemini Rate limits!

    with open("agent_results.json", "w") as out:
        json.dump(results, out, indent=2)
        
    print(f"\n🎉 Successfully saved {len(results)} structured results to agent_results.json!")

if __name__ == "__main__":
    main()
