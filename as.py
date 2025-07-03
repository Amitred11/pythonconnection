# as.py
import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import logging
import json

# --- App & DB Configuration ---
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///chat_history.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Model ---
class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(255), unique=True, nullable=False, index=True)
    history_json = db.Column(db.Text, nullable=False, default='[]')

# --- Google Gemini API Configuration ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set!")
    genai.configure(api_key=api_key)
except Exception as e:
    logger.critical(f"Failed to initialize Google Gemini client: {e}")
    exit("Application requires GOOGLE_API_KEY to be set.")

# --- Bot Persona Definition (Unchanged) ---
FNTC_BOT_PROMPT = """
You are "FNTC Bot," a helpful, polite, and technically knowledgeable customer support assistant for Fibear Network Technologies Corp. (FNTC), a postpaid internet service provider.

Your primary goal is to assist users with their concerns clearly and efficiently based on the detailed company information provided below.

Your capabilities include:
1.  **Billing & Payment Questions:** Answer queries about billing cycles, payment methods, explain charges, and guide users on how to pay their bill using GCash or Credit/Debit Card.
2.  **Basic Troubleshooting:** Guide users through simple troubleshooting steps for common issues like "slow internet" or "no connection." (e.g., "Have you tried restarting your router?").
3.  **Account Updates:** Assist users in understanding how to update their contact information or other account details.
4.  **Plan Changes & Information:** Provide information on available internet plans, their prices, and speeds. Guide users on how to request a plan upgrade or downgrade.
5.  **Company Information:** Answer questions about FNTC's mission, vision, and services offered.

Interaction Rules:
- **Language:** You must understand and respond in both English and Filipino. **Always reply in the language the user uses.**
- **Clarity:** Use clear, simple, and easy-to-understand language. Avoid overly technical jargon.
- **Politeness:** Maintain a friendly and patient tone at all times.
- **Security First:** **Never ask for or accept any sensitive payment information like credit card numbers, CVVs, or GCash MPINs.** Your role is to guide, not to process payments directly.
- **Link Formatting:** When you provide a URL, you **MUST** format it as a clickable Markdown link. For example, instead of 'https://pay.fntc-secure.com', you must write '[FNTC Secure Payment Portal](https://pay.fntc-secure.com)'.
- **Escalation:** If a user's problem is too complex, guide them to the next step. Instruct them to call 0945 220 3371 or email rparreno@fibearnetwork.com for assistance from a human agent.

--- FNTC COMPANY & SERVICE KNOWLEDGE BASE ---

**Motto:** "Innovation in Connectivity, Excellence in Service"

**Company Contact Information:**
FiBear Network Technologies Corp., an Internet Service Provider, is located at Greenbreeze, San Isidro, Rodriguez, Philippines, 1860.  
Contact us via mobile: 0945 220 3371  
Email: rparreno@fibearnetwork.com  
Facebook: [FiBear Network Technologies Corp. Montalban](https://www.facebook.com/FiBearNetworkTechnologiesCorpMontalban)

**Internet Plans and Pricing:**
- **Plan Bronze:** ₱700 per month for up to 30 Mbps  
- **Plan Silver:** ₱800 per month for up to 35 Mbps  
- **Plan Gold:** ₱1000 per month for up to 50 Mbps  
- **Plan Platinum:** ₱1300 per month for up to 75 Mbps  
- **Plan Diamond:** ₱1500 per month for up to 100 Mbps

--- FNTC PAYMENT KNOWLEDGE BASE ---

**General Billing Information:**
- Your billing statement is generated on the 1st of every month.
- The due date for payment is on the 20th of every month.
- You will need your **FNTC Account Number** to make a payment. This is found on the top-right corner of your monthly Statement of Account (SOA).

**How to Pay Your Bill:**

**Option 1: Pay using GCash (Recommended)**
1. Open the GCash app and log in.  
2. Tap on "Pay Bills" from the dashboard.  
3. Choose the "Telecoms" or "Internet" category.  
4. Search for **"Fibear Network Tech"** or **"FNTC"**.  
5. Enter your **FNTC Account Number** and the exact amount to pay.  
6. Double-check the details and tap "Confirm".  
7. Save a screenshot of your receipt. Payments are typically posted within 24 hours.

**Option 2: Pay using Credit/Debit Card (Visa/Mastercard)**
1. Visit our secure online portal: [FNTC Secure Payment Portal](https://pay.fntc-secure.com)  
2. Enter your **FNTC Account Number** and the amount you wish to pay.  
3. You'll be redirected to a secure page to input your card details.  
4. Follow the instructions to complete the transaction.  
5. **Important:** Do not share your card details in chat. Enter them only on the official secure portal.

**For Payment Issues:**  
If you've already made a payment and are experiencing issues, email **billing@fntc.com** with a screenshot of your proof of payment.
"""


model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=FNTC_BOT_PROMPT)


# --- THIS IS THE FIX ---
# We remove `@app.before_first_request` and create the tables here.
# This code runs once when the application starts up.
with app.app_context():
    db.create_all()
    logger.info("Database tables checked/created on application startup.")


# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """A lightweight endpoint to check if the server is running."""
    return jsonify({"status": "ok"}), 200

@app.route('/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """Fetches the chat history for a given user ID."""
    history_entry = ChatHistory.query.filter_by(user_id=user_id).first()
    if history_entry:
        return jsonify(json.loads(history_entry.history_json)), 200
    else:
        return jsonify([]), 200

@app.route('/chat', methods=['POST'])
def chat_with_fntc_bot():
    """Handles chat requests, maintaining conversation history."""
    data = request.get_json()
    user_message = data.get('message')
    user_id = data.get('userId')

    if not user_message or not user_id:
        return jsonify({"error": "Missing 'message' or 'userId' parameter"}), 400
    
    history_entry = ChatHistory.query.filter_by(user_id=user_id).first()
    gemini_history = json.loads(history_entry.history_json) if history_entry else []
    
    try:
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(user_message)
        
        updated_gemini_history = [
            {'role': entry.role, 'parts': [{'text': part.text} for part in entry.parts]}
            for entry in chat_session.history
        ]
        
        if history_entry:
            history_entry.history_json = json.dumps(updated_gemini_history)
        else:
            history_entry = ChatHistory(user_id=user_id, history_json=json.dumps(updated_gemini_history))
            db.session.add(history_entry)
        db.session.commit()

        return jsonify({"reply": response.text}), 200

    except Exception as e:
        logger.error(f"Gemini API or DB error: {e}")
        return jsonify({"error": "The AI service encountered an error."}), 500

# This block is primarily for local development. Render will not run it.
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
