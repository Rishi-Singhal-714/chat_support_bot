import csv
import json
import re
import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

# ================== CONFIG ==================
NVIDIA_API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "deepseek-ai/deepseek-v4-pro"

client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)

app = Flask(__name__)

# ================== LOAD DATA FROM CSV ==================
DATA_DIR = "data"

def load_csv(filename, required_cols):
    """Load a CSV from data/ and return a list of dicts."""
    data = []
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Clean BOM and whitespace
                row = {k.strip().lstrip('\ufeff'): v.strip() for k, v in row.items()}
                if all(col in row for col in required_cols):
                    data.append(row)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Starting with empty dataset.")
    return data

# In‑memory datasets (reload on server restart)
FAQ = load_csv("faq.csv", ["question", "answer"])
POLICIES = load_csv("policies.csv", ["policy_name", "content"])
PRODUCTS = load_csv("products.csv", ["product_id", "name", "price", "stock", "category"])
ORDERS = load_csv("orders.csv", ["order_id", "customer_name", "status", "tracking_id", "expected_delivery"])
TICKETS = load_csv("tickets.csv", ["ticket_id", "customer_name", "issue", "status", "priority"])

# Ticket counter (increments for new tickets)
if TICKETS:
    existing_ids = [int(t["ticket_id"][1:]) for t in TICKETS if t["ticket_id"].startswith("T") and t["ticket_id"][1:].isdigit()]
    TICKET_COUNTER = max(existing_ids) + 1 if existing_ids else 2000
else:
    TICKET_COUNTER = 2000

# ================== LLM HELPERS ==================
def call_llm(messages, temperature=0.2, max_tokens=2000):
    """Call NVIDIA LLM and return text."""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=False
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return None

def extract_json(text):
    """Try to parse JSON from LLM output."""
    try:
        return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None

# ================== INTENT CLASSIFICATION ==================
def classify_intent(query):
    system = """You are an intent classifier. Output ONLY valid JSON:
{
  "intent": "faq|orders|products|policies|tickets",
  "entities": {"order_id": "...", "product_name": "...", "ticket_id": "...", "issue_description": "..."},
  "confidence": 0.0-1.0,
  "action": "view|raise_ticket|none"
}
Set "raise_ticket" if the user wants to create a new ticket (e.g., report problem, request faster delivery, complaint). Use null for missing entities."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    raw = call_llm(messages, 0.1, 300)
    parsed = extract_json(raw) or {}
    return {
        "intent": parsed.get("intent", "faq"),
        "entities": parsed.get("entities", {}),
        "confidence": parsed.get("confidence", 0.8),
        "action": parsed.get("action", "view")
    }

# ================== ANSWER HANDLERS ==================
def answer_faq(query):
    if not FAQ:
        return "FAQ database is empty."
    faq_text = "\n".join([f"Q: {e['question']}\nA: {e['answer']}" for e in FAQ])
    prompt = f"""You are a support agent. Answer using only the FAQ below. If not found, politely say you'll connect them to a human agent.
FAQ:
{faq_text}
User: {query}
Answer:"""
    return call_llm([{"role": "system", "content": prompt}], 0.3, 300) or "Sorry, I couldn't process that."

def answer_policy(query):
    if not POLICIES:
        return "Policy database is empty."
    pol_text = "\n".join([f"{p['policy_name']}: {p['content']}" for p in POLICIES])
    prompt = f"""Answer the user's question based on these company policies. Be concise.
Policies:
{pol_text}
User: {query}
Answer:"""
    return call_llm([{"role": "system", "content": prompt}], 0.3, 300) or "Error fetching policy."

def handle_order(order_id, action, entities):
    if order_id:
        order = next((o for o in ORDERS if o["order_id"] == order_id), None)
        if not order:
            return f"Order {order_id} not found."
        if action == "raise_ticket":
            global TICKET_COUNTER
            TICKET_COUNTER += 1
            new_ticket = {
                "ticket_id": f"T{TICKET_COUNTER}",
                "customer_name": order["customer_name"],
                "issue": entities.get("issue_description", "Fast delivery requested"),
                "status": "Open",
                "priority": "High"
            }
            TICKETS.append(new_ticket)
            return f"I've created ticket **{new_ticket['ticket_id']}** for faster delivery of order {order_id}. Our team will contact you soon."
        else:
            return (f"Order **{order_id}** – Status: {order['status']}, "
                    f"Tracking: {order.get('tracking_id','N/A')}, "
                    f"ETA: {order.get('expected_delivery','N/A')}.")
    return "Please provide your order ID (e.g., ORD1001)."

def handle_product(prod_name):
    if prod_name:
        matches = [p for p in PRODUCTS if prod_name.lower() in p["name"].lower()]
        if matches:
            p = matches[0]
            return (f"**{p['name']}** (ID: {p['product_id']}) — "
                    f"₹{p['price']}, Stock: {p['stock']}, Category: {p['category']}.")
        return f"No product matching '{prod_name}' found."
    return "Which product are you looking for?"

def handle_ticket(ticket_id, action, entities):
    if action == "raise_ticket":
        issue = entities.get("issue_description", "Issue reported via chat")
        global TICKET_COUNTER
        TICKET_COUNTER += 1
        new_ticket = {
            "ticket_id": f"T{TICKET_COUNTER}",
            "customer_name": "Guest",
            "issue": issue,
            "status": "Open",
            "priority": "Medium"
        }
        TICKETS.append(new_ticket)
        return f"New ticket **{new_ticket['ticket_id']}** created: '{issue}'. We'll respond shortly."
    elif ticket_id:
        ticket = next((t for t in TICKETS if t["ticket_id"] == ticket_id), None)
        if ticket:
            return (f"Ticket **{ticket_id}** – Status: {ticket['status']}, "
                    f"Priority: {ticket['priority']}, Issue: {ticket['issue']}.")
        return f"Ticket {ticket_id} not found."
    return "Please provide a ticket ID (e.g., T1001) or describe the issue for a new ticket."

# ================== API ENDPOINT ==================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "user_query" not in data:
        return jsonify({"error": "Missing user_query"}), 400
    query = data["user_query"].strip()

    intent_res = classify_intent(query)
    intent = intent_res["intent"]
    entities = intent_res["entities"] or {}
    action = intent_res.get("action", "view")

    # Route to appropriate handler
    if intent == "faq":
        answer = answer_faq(query)
    elif intent == "policies":
        answer = answer_policy(query)
    elif intent == "orders":
        answer = handle_order(entities.get("order_id"), action, entities)
    elif intent == "products":
        answer = handle_product(entities.get("product_name"))
    elif intent == "tickets":
        answer = handle_ticket(entities.get("ticket_id"), action, entities)
    else:
        answer = answer_faq(query)  # fallback

    response = {
        "response": answer,
        "intent": intent,
        "confidence": intent_res["confidence"],
        "action_taken": {"type": action, "details": entities} if action != "view" else {}
    }
    return jsonify(response)

# ================== FRONTEND ==================
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)