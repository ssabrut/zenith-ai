import uuid
import gradio as gr
import requests
import os
import json
from typing import List

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/api/v1/chat/"

def interact_with_agent(message: str, history: List[dict], thread_id: str):
    """
    Sends the user message to the FastAPI backend via POST and streams the response.
    """
    if not message:
        yield history, ""
        return

    # 1. Update history with User Message and a placeholder
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "⏳ *Thinking...*"})
    yield history, ""

    # 2. Prepare Payload
    payload = {
        "query": message,
        "thread_id": thread_id
    }

    try:
        # 3. Make the POST request with stream=True
        with requests.post(CHAT_ENDPOINT, json=payload, stream=True) as response:
            print("Response:", response)
            
            # Check for HTTP errors (4xx, 5xx)
            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get('detail', error_detail)
                except:
                    pass
                raise Exception(f"Server Error {response.status_code}: {error_detail}")

            # 4. Process the Stream
            partial_response = ""
            
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    partial_response += chunk
                    
                    # Update the latest bot message in history
                    history[-1]["content"] = partial_response
                    
                    # Yield to Gradio to update UI
                    yield history, ""

    except requests.exceptions.ConnectionError:
        history[-1]["content"] = f"❌ Connection Error: Could not reach backend at {CHAT_ENDPOINT}. Is the server running?"
        yield history, ""
        
    except Exception as e:
        history[-1]["content"] = f"❌ Error: {str(e)}"
        yield history, ""

def init_session():
    """Generates a unique session ID for the user"""
    new_id = str(uuid.uuid4())
    print(f"Starting new session: {new_id}")
    return new_id

def clear_data():
    """Resets UI and generates new session ID"""
    return [], "", str(uuid.uuid4())

# --- GRADIO UI LAYOUT ---
with gr.Blocks(title="ERHA/Dermies AI Agent") as demo:
    
    # Store the unique thread_id
    session_state = gr.State(value=init_session)
    
    gr.Markdown("# 🏥 ERHA/Dermies Smart Assistant")
    gr.Markdown("Ask about treatments, prices, or manage your appointments.")
    
    # Use type="messages" for dictionary format
    chatbot = gr.Chatbot(
        max_height=500,
        avatar_images=(None, "🤖")
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Contoh: 'Berapa harga peeling wajah?'",
            show_label=False,
            scale=4
        )

    clear_btn = gr.Button("Clear Conversation", variant="secondary")

    # --- EVENT WIRING ---
    msg.submit(
        interact_with_agent, 
        inputs=[msg, chatbot, session_state], 
        outputs=[chatbot, msg]
    )

    clear_btn.click(
        clear_data, 
        outputs=[chatbot, msg, session_state]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())