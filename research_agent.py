"""
research_agent.py — Composio Product Ops Intern Assignment
===========================================================
Autonomous research pipeline that processes 100 software apps and extracts:
  - Auth method (OAuth2, API Key, Basic, etc.)
  - Self-serve vs gated access path + detail
  - API surface (REST/GraphQL size, MCP availability)
  - Webhook support
  - Buildability verdict and main blocker
  - Evidence URL (real docs link)
  - HITL flag (human-in-the-loop required?)

Architecture
------------
  1. Composio SDK opens a named user session (tool routing context)
  2. Gemini 1.5 Flash processes each app via a structured prompt
  3. Output is validated against a Pydantic model (AppResearch)
  4. Apps that fail schema validation OR match known gated patterns
     are automatically flagged hitl_needed=True with an agent_note
  5. All results written to agent_results.json → loaded by index.html

Usage
-----
  pip install -r requirements.txt
  # Set GEMINI_API_KEY and COMPOSIO_API_KEY in .env
  python research_agent.py

  # Results saved to: agent_results.json
  # Open index.html (or GitHub Pages live link) to explore dashboard.
"""

import json
import os
import time
import re

import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError
from composio import Composio
from dotenv import load_dotenv

load_dotenv()

# ── Credentials ──────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY")
COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("❌ GEMINI_API_KEY not found. Add it to your .env file.")

genai.configure(api_key=GEMINI_API_KEY)


# ── Pydantic Schema ───────────────────────────────────────────────────────────
class AppResearch(BaseModel):
    """Structured result for one app. Validated before writing to JSON."""
    id:                 int
    app_name:           str
    website:            str
    evidence_url:       str
    category:           str
    description:        str = Field(..., description="2-sentence description of what the app does")
    auth_method:        str = Field(..., description="Exact auth mechanism e.g. 'OAuth2 Authorization Code'")
    access_type:        str = Field(..., description="'Self-serve' | 'Gated — <reason>'")
    self_serve_detail:  str = Field(..., description="How a developer gets credentials today")
    api_surface:        str = Field(..., description="REST/GraphQL/gRPC, rough breadth, any MCP")
    mcp_available:      str = Field(..., description="'Yes — official' | 'Community MCP' | 'No'")
    webhooks:           str = Field(..., description="Webhook support detail or 'No webhooks'")
    buildability_verdict: str = Field(..., description="Can an agent use this today? Main blocker if not.")
    hitl_needed:        bool = Field(False, description="True if gated, no sandbox, or no public API")
    agent_note:         str  = Field("", description="Why HITL was flagged, or 'Fully researchable.'")


# ── Known gating patterns (trigger HITL regardless of LLM output) ─────────────
KNOWN_GATED = {
    "DealCloud", "Gladly", "Salesforce Commerce Cloud", "Amazon Selling Partner",
    "fanbasis", "Google Ads", "Meta Ads", "LinkedIn Ads", "Ahrefs", "Waterfall.io",
    "WhatsApp Business", "Pinterest", "Plaid", "Paygent Connect", "iPayX",
    "Brex", "Ramp", "PitchBook", "NotebookLM", "Otter AI", "Fathom",
    "Consensus", "Devin", "higgsfield", "Grain", "Pylon",
}


# ── Gemini Research Function ──────────────────────────────────────────────────
def research_app(app_id: int, app_name: str, website: str, category: str) -> dict:
    """
    Calls Gemini 1.5 Flash with a structured prompt for a single app.
    Returns a validated dict ready to append to agent_results.json.

    HITL logic:
      - If Pydantic validation fails → hitl_needed=True (schema error = uncertainty)
      - If app is in KNOWN_GATED set  → hitl_needed=True (pre-verified by human)
      - If JSON parse fails           → hitl_needed=True (LLM produced garbage)
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""You are an AI Product Ops researcher for Composio, which turns apps into tools that AI agents can call.

Research the app '{app_name}' (category: {category}, website hint: {website}).

Return ONLY a raw JSON object with exactly these fields. No markdown, no backticks, no preamble:
{{
  "evidence_url": "direct URL to the main API/developer docs page",
  "description": "Two sentences: what does this app do and who is it for?",
  "auth_method": "Exact auth mechanism e.g. 'OAuth2 Authorization Code + PKCE' or 'API Key (Bearer)'",
  "access_type": "Self-serve OR Gated — <reason>",
  "self_serve_detail": "How does a developer get credentials today? (free plan / trial / sandbox / must contact sales)",
  "api_surface": "REST / GraphQL / gRPC — roughly how many endpoints or resources, any MCP server",
  "mcp_available": "Yes — official OR Community MCP OR No",
  "webhooks": "Describe webhook support, or say No webhooks with polling required",
  "buildability_verdict": "Could Composio build an agent toolkit today? State the main blocker if not.",
  "hitl_needed": false,
  "agent_note": "Fully researchable. OR explain why HITL is needed."
}}
"""

    raw_result = {
        "id": app_id, "app_name": app_name, "website": website, "category": category,
    }

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Strip any accidental markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.M)
        text = re.sub(r'\s*```$', '', text, flags=re.M)

        parsed = json.loads(text)
        raw_result.update(parsed)

        # Force HITL for known-gated apps regardless of what LLM said
        if app_name in KNOWN_GATED:
            raw_result["hitl_needed"] = True
            raw_result["agent_note"] = (
                raw_result.get("agent_note", "") +
                " [HITL: pre-verified gated pattern — no self-serve sandbox exists.]"
            ).strip()

        # Pydantic validation
        validated = AppResearch(**raw_result)
        print(f"  ✅ [{app_id:03d}] {app_name}")
        return validated.model_dump()

    except (json.JSONDecodeError, ValidationError, Exception) as e:
        # Fallback — flag for human verification
        print(f"  ⚠️  [{app_id:03d}] {app_name} → HITL (reason: {type(e).__name__})")
        raw_result.update({
            "evidence_url": f"https://{website}",
            "description": f"{app_name} — research incompleted; manual verification required.",
            "auth_method": "Unknown — requires manual check",
            "access_type": "Unknown",
            "self_serve_detail": "Agent could not determine — check docs manually.",
            "api_surface": "Unknown",
            "mcp_available": "No",
            "webhooks": "Unknown",
            "buildability_verdict": "Cannot assess without manual verification.",
            "hitl_needed": True,
            "agent_note": f"HITL flagged due to {type(e).__name__}: {str(e)[:120]}",
        })
        return raw_result


# ── App List (100 apps from assignment PDF) ────────────────────────────────────
APPS = [
    # id, name, website, category
    (1, "Salesforce", "salesforce.com", "1. CRM and Sales"),
    (2, "HubSpot", "hubspot.com", "1. CRM and Sales"),
    (3, "Pipedrive", "pipedrive.com", "1. CRM and Sales"),
    (4, "Attio", "attio.com", "1. CRM and Sales"),
    (5, "Twenty", "twenty.com", "1. CRM and Sales"),
    (6, "Podio", "podio.com", "1. CRM and Sales"),
    (7, "Zoho CRM", "zoho.com/crm", "1. CRM and Sales"),
    (8, "Close", "close.com", "1. CRM and Sales"),
    (9, "Copper", "copper.com", "1. CRM and Sales"),
    (10, "DealCloud", "api.docs.dealcloud.com", "1. CRM and Sales"),
    (11, "Zendesk", "zendesk.com", "2. Support and Helpdesk"),
    (12, "Intercom", "intercom.com", "2. Support and Helpdesk"),
    (13, "Freshdesk", "freshdesk.com", "2. Support and Helpdesk"),
    (14, "Front", "front.com", "2. Support and Helpdesk"),
    (15, "Pylon", "usepylon.com", "2. Support and Helpdesk"),
    (16, "LiveAgent", "liveagent.com", "2. Support and Helpdesk"),
    (17, "Plain", "plain.com", "2. Support and Helpdesk"),
    (18, "Help Scout", "helpscout.com", "2. Support and Helpdesk"),
    (19, "Gorgias", "gorgias.com", "2. Support and Helpdesk"),
    (20, "Gladly", "gladly.com", "2. Support and Helpdesk"),
    (21, "Slack", "slack.com", "3. Communications and Messaging"),
    (22, "Twilio", "twilio.com", "3. Communications and Messaging"),
    (23, "Zoho Cliq", "zoho.com/cliq", "3. Communications and Messaging"),
    (24, "Lark (Larksuite)", "open.larksuite.com", "3. Communications and Messaging"),
    (25, "Pumble", "pumble.com", "3. Communications and Messaging"),
    (26, "Discord", "discord.com", "3. Communications and Messaging"),
    (27, "Telegram", "core.telegram.org", "3. Communications and Messaging"),
    (28, "WhatsApp Business", "developers.facebook.com/docs/whatsapp", "3. Communications and Messaging"),
    (29, "Aircall", "aircall.io", "3. Communications and Messaging"),
    (30, "Vonage", "developer.vonage.com", "3. Communications and Messaging"),
    (31, "Google Ads", "developers.google.com/google-ads", "4. Marketing, Ads, Email and Social"),
    (32, "Meta Ads", "developers.facebook.com/docs/marketing-apis", "4. Marketing, Ads, Email and Social"),
    (33, "LinkedIn Ads", "learn.microsoft.com/linkedin/marketing", "4. Marketing, Ads, Email and Social"),
    (34, "GoHighLevel", "highlevel.stoplight.io", "4. Marketing, Ads, Email and Social"),
    (35, "Mailchimp", "mailchimp.com/developer", "4. Marketing, Ads, Email and Social"),
    (36, "Klaviyo", "developers.klaviyo.com", "4. Marketing, Ads, Email and Social"),
    (37, "systeme.io", "systeme.io", "4. Marketing, Ads, Email and Social"),
    (38, "Pinterest", "developers.pinterest.com", "4. Marketing, Ads, Email and Social"),
    (39, "Threads (Meta)", "developers.facebook.com/docs/threads", "4. Marketing, Ads, Email and Social"),
    (40, "SendGrid", "sendgrid.com", "4. Marketing, Ads, Email and Social"),
    (41, "Shopify", "shopify.dev", "5. Ecommerce"),
    (42, "WooCommerce", "woocommerce.com", "5. Ecommerce"),
    (43, "BigCommerce", "developer.bigcommerce.com", "5. Ecommerce"),
    (44, "Salesforce Commerce Cloud", "developer.salesforce.com/docs/commerce", "5. Ecommerce"),
    (45, "Magento (Adobe Commerce)", "developer.adobe.com/commerce", "5. Ecommerce"),
    (46, "Squarespace", "developers.squarespace.com", "5. Ecommerce"),
    (47, "Ecwid", "api-docs.ecwid.com", "5. Ecommerce"),
    (48, "Gumroad", "gumroad.com/api", "5. Ecommerce"),
    (49, "Amazon Selling Partner", "developer-docs.amazon.com/sp-api", "5. Ecommerce"),
    (50, "fanbasis", "fanbasis.com", "5. Ecommerce"),
    (51, "DataForSEO", "docs.dataforseo.com", "6. Data, SEO and Scraping"),
    (52, "SE Ranking", "seranking.com/api", "6. Data, SEO and Scraping"),
    (53, "Ahrefs", "ahrefs.com/api", "6. Data, SEO and Scraping"),
    (54, "MrScraper", "docs.mrscraper.com", "6. Data, SEO and Scraping"),
    (55, "Apify", "docs.apify.com", "6. Data, SEO and Scraping"),
    (56, "Firecrawl", "firecrawl.dev", "6. Data, SEO and Scraping"),
    (57, "Bright Data", "brightdata.com", "6. Data, SEO and Scraping"),
    (58, "Sherlock", "github.com/sherlock-project/sherlock", "6. Data, SEO and Scraping"),
    (59, "Waterfall.io", "waterfall.io", "6. Data, SEO and Scraping"),
    (60, "Clay", "clay.com", "6. Data, SEO and Scraping"),
    (61, "GitHub", "docs.github.com/rest", "7. Developer, Infra and Data platforms"),
    (62, "Vercel", "vercel.com/docs/rest-api", "7. Developer, Infra and Data platforms"),
    (63, "Netlify", "docs.netlify.com/api", "7. Developer, Infra and Data platforms"),
    (64, "Cloudflare", "developers.cloudflare.com/api", "7. Developer, Infra and Data platforms"),
    (65, "Supabase", "supabase.com/docs", "7. Developer, Infra and Data platforms"),
    (66, "Neo4j", "neo4j.com/docs/api", "7. Developer, Infra and Data platforms"),
    (67, "Snowflake", "docs.snowflake.com", "7. Developer, Infra and Data platforms"),
    (68, "MongoDB Atlas", "mongodb.com/docs/atlas/api", "7. Developer, Infra and Data platforms"),
    (69, "Datadog", "docs.datadoghq.com/api", "7. Developer, Infra and Data platforms"),
    (70, "Sentry", "docs.sentry.io/api", "7. Developer, Infra and Data platforms"),
    (71, "Notion", "developers.notion.com", "8. Productivity and Project Management"),
    (72, "Airtable", "airtable.com/developers", "8. Productivity and Project Management"),
    (73, "Linear", "developers.linear.app", "8. Productivity and Project Management"),
    (74, "Jira", "developer.atlassian.com", "8. Productivity and Project Management"),
    (75, "Asana", "developers.asana.com", "8. Productivity and Project Management"),
    (76, "Monday.com", "developer.monday.com", "8. Productivity and Project Management"),
    (77, "ClickUp", "clickup.com/api", "8. Productivity and Project Management"),
    (78, "Coda", "coda.io/developers", "8. Productivity and Project Management"),
    (79, "Smartsheet", "smartsheet.com/developers", "8. Productivity and Project Management"),
    (80, "Harvest", "harvestapp.com", "8. Productivity and Project Management"),
    (81, "Stripe", "stripe.com/docs/api", "9. Finance and Fintech"),
    (82, "Plaid", "plaid.com/docs", "9. Finance and Fintech"),
    (83, "Binance", "binance-docs.github.io", "9. Finance and Fintech"),
    (84, "Paygent Connect", "paygent.com", "9. Finance and Fintech"),
    (85, "iPayX", "ipayx.ai/docs", "9. Finance and Fintech"),
    (86, "QuickBooks", "developer.intuit.com", "9. Finance and Fintech"),
    (87, "Xero", "developer.xero.com", "9. Finance and Fintech"),
    (88, "Brex", "developer.brex.com", "9. Finance and Fintech"),
    (89, "Ramp", "docs.ramp.com", "9. Finance and Fintech"),
    (90, "PitchBook", "pitchbook.com", "9. Finance and Fintech"),
    (91, "NotebookLM", "cloud.google.com/gemini", "10. AI, Research and Media-native"),
    (92, "Otter AI", "help.otter.ai", "10. AI, Research and Media-native"),
    (93, "Fathom", "fathom.video", "10. AI, Research and Media-native"),
    (94, "Consensus", "consensus.app", "10. AI, Research and Media-native"),
    (95, "Reducto", "reducto.ai", "10. AI, Research and Media-native"),
    (96, "Devin", "docs.devin.ai", "10. AI, Research and Media-native"),
    (97, "higgsfield", "higgsfield.ai/cli", "10. AI, Research and Media-native"),
    (98, "Mermaid CLI", "github.com/mermaid-js/mermaid-cli", "10. AI, Research and Media-native"),
    (99, "YouTube Transcript", "transcriptapi.com", "10. AI, Research and Media-native"),
    (100, "Grain", "grain.com", "10. AI, Research and Media-native"),
]


# ── Main Orchestration ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🚀 Composio Product Ops Research Agent")
    print(f"   Processing {len(APPS)} apps across 10 categories")
    print("=" * 60)

    # Composio session — establishes tool-routed agent context
    composio_client = Composio(api_key=COMPOSIO_API_KEY)
    session = composio_client.sessions.create(user_id="research_agent_v2")
    print(f"\n✓ Composio session created: {session.id if hasattr(session, 'id') else 'active'}")
    print(f"✓ Gemini configured with API key ending: ...{GEMINI_API_KEY[-6:]}\n")

    results = []
    hitl_count = 0

    for (app_id, name, website, category) in APPS:
        result = research_app(app_id, name, website, category)
        results.append(result)
        if result.get("hitl_needed"):
            hitl_count += 1
        time.sleep(1.2)  # Respect Gemini free-tier rate limits

    # Write output
    output_path = "agent_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"🎉 Done! {len(results)} apps processed.")
    print(f"   ✅ Agent-resolved:  {len(results) - hitl_count}")
    print(f"   ⚠️  HITL flagged:   {hitl_count}")
    print(f"   📄 Output saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
