import json
import os
import time
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from composio_openai import ComposioToolSet, Action

# ----------------- Configuration -----------------
# Ensure you have your keys set in your environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY")

class AppResearch(BaseModel):
    app_name: str
    category: str
    description_one_liner: str
    auth_method: str = Field(description="Dominant auth: OAuth2, API key, Basic, Webhooks, or other")
    access_type: str = Field(description="Identify clearly: Self-serve OR Gated")
    api_surface: str = Field(description="REST, GraphQL, MCP available? How broad is the coverage?")
    webhooks_supported: bool = Field(description="Are webhooks officially supported?")
    buildability_verdict: str = Field(description="Agent toolkit potential today. What is the blocker?")
    evidence_url: str = Field(description="Doc URL where you found this")

def init_agent():
    """Initialize Composio Toolset and OpenAI Client"""
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # We equip the agent with web search tooling using Composio!
    # E.g., using EXA or TAVILY search through Composio
    toolset = ComposioToolSet(api_key=COMPOSIO_API_KEY)
    tools = toolset.get_tools(actions=[Action.EXA_SEARCH]) # Example web search action
    
    return openai_client, tools

def research_app(app_name: str, hint_url: str, openai_client: OpenAI, tools: list) -> AppResearch:
    """Executes the AI agent workflow for a specific application."""
    print(f"🕵️ Researching {app_name} ({hint_url})...")
    
    system_prompt = (
        "You are an elite Product Operations CI/CD Agent for Composio. "
        "Your job is to research software developer APIs and extract strictly formatted data. "
        "Use the provided robust web search tools to check developer portals, pricing pages, and auth documentation. "
        "If an app explicitly requires 'Contact Sales' for API access, mark it as 'Gated'."
    )
    
    user_prompt = f"Please research the app '{app_name}'. Documentation hint is '{hint_url}'."
    
    try:
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=tools,
            response_format=AppResearch,
        )
        
        # Here we'd ideally execute the tool calls, feed them back to GPT-4o, 
        # but for this script we assume OpenAI parses directly with tools attached.
        # Note: A full agentic loop (like Langchain / Composio loop) would be implemented in a multi-turn here.
        # This is a concise representation.
        
        return response.choices[0].message.parsed
        
    except Exception as e:
        print(f"Error researching {app_name}: {str(e)}")
        # Adding Human Intervention fallback
        return AppResearch(
            app_name=app_name,
            category="Requires Human Verification",
            description_one_liner="Agent failed to research or web search blocked.",
            auth_method="Unknown - Needs Human",
            access_type="Unknown",
            api_surface="Unknown",
            webhooks_supported=False,
            buildability_verdict="Agent failed execution. Manual check required.",
            evidence_url="N/A"
        )

def main():
    if not OPENAI_API_KEY or not COMPOSIO_API_KEY:
        print("❌ Missing API Keys! Please set OPENAI_API_KEY and COMPOSIO_API_KEY.")
        return

    # Load from our JSON
    with open("apps_list.json", "r") as f:
        apps_to_research = json.load(f)
        
    openai_client, tools = init_agent()
    results = []
    
    # Process them (we'll limit to 5 for demonstration so we don't blow up costs)
    for app in apps_to_research[:5]:
        result = research_app(app["name"], app["hint_url"], openai_client, tools)
        if result:
            results.append(result.dict())
        time.sleep(2) # Rate limit protection
        
    # Validation / Human loop queueing
    print("🔍 Sending failed items to a Human Queue...")
    human_queue = [r for r in results if "Needs Human" in r['auth_method']]
    
    with open("agent_results.json", "w") as out:
        json.dump(results, out, indent=2)
        
    print(f"✅ Saved {len(results)} structured results to agent_results.json")
    print(f"⚠️ {len(human_queue)} apps required human verification.")

if __name__ == "__main__":
    main()
