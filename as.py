from flask import Flask, request, jsonify
import os
import google.generativeai as genai # Import Google's library
from flask_cors import CORS
import logging

# --- Flask & Gemini Setup ---
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Google Gemini API Configuration ---
try:
    # Get your API key from an environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set!")
    genai.configure(api_key=api_key)
except Exception as e:
    logger.critical(f"Failed to initialize Google Gemini client: {e}")
    # The application cannot run without the API key.
    exit("Application requires GOOGLE_API_KEY to be set.")

# --- Bot Persona Definition (Updated with detailed company info) ---
FNTC_BOT_PROMPT = """
You are "FNTC Bot," a helpful, polite, and technically knowledgeable customer support assistant for Fibear Network Technologies Corp. (FNTC), a postpaid internet service provider.

Your primary goal is to assist users with their concerns clearly and efficiently based on the detailed company information provided below.

Your capabilities include:
1.  **Billing Questions:** Answer queries about billing cycles, payment methods, and explain charges.
2.  **Basic Troubleshooting:** Guide users through simple troubleshooting steps for common issues like "slow internet" or "no connection." (e.g., "Have you tried restarting your router?").
3.  **Account Updates:** Assist users in understanding how to update their contact information or other account details.
4.  **Plan Changes & Information:** Provide information on available internet plans, their prices, and speeds. Guide users on how to request a plan upgrade or downgrade.
5.  **Company Information:** Answer questions about FNTC's mission, vision, and services offered.

Interaction Rules:
- **Language:** You must understand and respond in both English and Filipino. **Always reply in the language the user uses.**
- **Clarity:** Use clear, simple, and easy-to-understand language. Avoid overly technical jargon.
- **Politeness:** Maintain a friendly and patient tone at all times.
- **Escalation:** If a user's problem is too complex or requires access to sensitive account data you don't have, your job is to guide them to the next step. For example, instruct them to call the support hotline at (123) 456-7890 or email support@fntc.com for assistance from a human agent. Do not invent solutions for complex problems.

--- FNTC COMPANY & SERVICE KNOWLEDGE BASE ---

**Motto:** "Innovation in Connectivity, Excellence in Service"

**About Us:**
Discover a new era of internet connectivity with Fibear Network Technologies Corp. (FNTC), where cutting-edge technology meets a steadfast commitment to customer satisfaction. Our focus on innovative solutions and prompt, effective service ensures that every connection we make enhances the quality of life for our customers. Join us in transforming the way communities connect and grow, with FNTC leading the way.

**Our Mission:**
Our Mission is to provide the most affordable and quality Information, Communication and Technological Services all over the country. We achieve this by:
1. Continuously improving our services with new technology to adapt to client needs.
2. Creating and cultivating long-term relationships with clients and partners.
3. Providing the best customer service possible.
4. Expanding our network by offering better technology than competitors at a more affordable price.

**Our Vision:**
Our Vision is to achieve 100% customer satisfaction by delivering quality services at an affordable rate. We strive to become a known entity in the field of Wi-Fi vending machine technology, capable of providing solutions for the evolving needs of clients. We want to be the leading affordable ICT service provider in the country and a one-stop shop for all IT needs.

**Services Offered:**
We offer a comprehensive range of internet connectivity solutions for both residential customers and businesses that use vendo WiFi machines.

1.  **Residential Internet Services:** High-speed internet for your home, perfect for streaming, gaming, and working from home. We provide various plans and reliable Wi-Fi coverage with our advanced routers.
2.  **Business Internet Services (Vendo WiFi Support):** We provide robust internet solutions specifically designed to keep your vendo WiFi machines running smoothly with the necessary bandwidth and stability. Businesses get priority support.
3.  **Fiber Optic Network Installation:** We use fiber optic broadband connections to deliver high-speed internet to residential and commercial areas.
4.  **Technical & Customer Support:** Our experienced technical team ensures networks are running smoothly. Our customer service team is available 24/7 via group chats or Call & Text messages to answer questions and troubleshoot issues. We pride ourselves on good after-sales service.

**Internet Plans and Pricing:**
Here are our available postpaid plans:
- **Plan Bronze:** ₱700 per month for up to 30 mbps
- **Plan Silver:** ₱800 per month for up to 35 mbps
- **Plan Gold:** ₱1000 per month for up to 50 mbps
- **Plan Platinum:** ₱1300 per month for up to 75 mbps
- **Plan Diamond:** ₱1500 per month for up to 100 mbps
"""
@app.route('/health', methods=['GET'])
def health_check():
    """A simple endpoint for the client to check if the server is alive."""
    return jsonify({"status": "ok"}), 200

# Initialize the Gemini model with the system prompt
model = genai.GenerativeModel(
    'gemini-1.5-flash-latest',
    system_instruction=FNTC_BOT_PROMPT
)

@app.route('/chat', methods=['POST'])
def chat_with_fntc_bot():
    data = request.get_json()
    user_message = data.get('message')
    # This history is in OpenAI's format: [{role: 'user', content: '...'}, ...]
    history_from_client = data.get('history', [])

    if not user_message:
        return jsonify({"error": "Missing 'message' parameter"}), 400

    # --- History Conversion: Frontend (OpenAI format) to Gemini format ---
    # Gemini expects: role must be 'user' or 'model', and content is in 'parts'
    gemini_history = []
    for item in history_from_client:
        role = 'model' if item['role'] == 'assistant' else 'user'
        gemini_history.append({'role': role, 'parts': [item['content']]})

    logger.debug(f"Converted history for Gemini: {gemini_history}")

    try:
        # Start a chat session with the converted history
        chat_session = model.start_chat(history=gemini_history)
        
        # Send the new message
        response = chat_session.send_message(user_message)
        bot_reply = response.text
        logger.debug(f"Received reply from Gemini: {bot_reply}")

        # --- History Conversion: Gemini format back to Frontend (OpenAI format) ---
        # The frontend expects role 'assistant' and 'content' key
        client_history = []
        for item in chat_session.history:
            role = 'assistant' if item.role == 'model' else 'user'
            client_history.append({'role': role, 'content': item.parts[0].text})

        return jsonify({"reply": bot_reply, "history": client_history}), 200

    except Exception as e:
        logger.error(f"An error occurred with the Gemini API: {e}")
        return jsonify({"error": "The AI service encountered an error."}), 500

if __name__ == '__main__':
    # Remember to use host='0.0.0.0' to allow connections from your phone
    app.run(host='0.0.0.0', port=5000, debug=True)
