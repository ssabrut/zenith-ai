import uuid
import gradio as gr
import asyncio
from typing import List, Tuple

# --- IMPORT YOUR BACKEND ---
# Adjust the import path based on your folder structure
# We assume 'app' is the compiled StateGraph from core/graph/workflow.py
try:
    from core.graph.workflow import app
except ImportError:
    # Fallback for testing if backend isn't ready
    print("⚠️ Backend not found. Using dummy mock.")
    app = None

async def interact_with_agent(message: str, history: List[List], thread_id: str):
    """
    Core logic to talk to LangGraph and stream results to Gradio.
    """
    if not message:
        yield history, ""
        return

    # 1. Update history with User Message and a placeholder for Bot
    # history format: [[user_msg, bot_msg], ...]
    history.append([message, "⏳ *Thinking...*"])
    yield history, "" # Update UI immediately

    # 2. Config for LangGraph (Memory)
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"question": message}

    if app is None:
        # Mock response if backend is missing
        await asyncio.sleep(1)
        history[-1][1] = "Backend is not connected. Please check imports."
        yield history, ""
        return

    # 3. Stream from LangGraph
    # We use astream to get updates node-by-node
    try:
        async for event in app.astream(inputs, config=config):
            for node_name, state_update in event.items():
                
                # OPTIONAL: Show status updates based on active node
                if node_name == "router":
                    history[-1][1] = "🔄 *Analyzing your request...*"
                    yield history, ""
                elif node_name == "rag":
                    history[-1][1] = "📚 *Searching knowledge base...*"
                    yield history, ""
                elif node_name == "booking":
                    history[-1][1] = "📅 *Checking appointment system...*"
                    yield history, ""
                
                # FINAL ANSWER: When 'generate' node finishes
                elif node_name == "generate":
                    # Extract the AIMessage content
                    # State structure: {'messages': [AIMessage(content='...'), ...]}
                    messages = state_update.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        final_text = last_message.content
                        
                        # Update the placeholder with the real answer
                        history[-1][1] = final_text
                        yield history, ""
                        
    except Exception as e:
        history[-1][1] = f"❌ Error: {str(e)}"
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
with gr.Blocks(title="ERHA/Dermies AI Agent", theme=gr.themes.Soft()) as demo:
    
    # Store the unique thread_id in a hidden state component
    session_state = gr.State(value=init_session)
    
    gr.Markdown("# 🏥 ERHA/Dermies Smart Assistant")
    gr.Markdown("Ask about treatments, prices, or manage your appointments.")
    
    chatbot = gr.Chatbot(
        height=500,
        avatar_images=(None, "🤖"), # Add user icon if desired
        bubble_full_width=False,
        show_copy_button=True,
        type="messages" # Newer Gradio versions prefer 'messages' format, but 'tuples' is safer for older
    )
    # Note: If you get a 'format' error, remove type="messages" to default to tuples.
    # The code below assumes TUPLES format [[user, bot]] which works on almost all versions.
    chatbot.type = "tuples" 

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Contoh: 'Berapa harga peeling wajah?' atau 'Saya mau booking konsultasi'",
            show_label=False,
            scale=4
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")

    clear_btn = gr.Button("Clear Conversation", variant="secondary")

    # --- EVENT WIRING ---
    
    # 1. User sends message (Enter key or Button click)
    # We pass: [message, chatbot history, session_id]
    # We output: [chatbot history, message (cleared)]
    msg.submit(
        interact_with_agent, 
        inputs=[msg, chatbot, session_state], 
        outputs=[chatbot, msg]
    )
    
    submit_btn.click(
        interact_with_agent, 
        inputs=[msg, chatbot, session_state], 
        outputs=[chatbot, msg]
    )

    # 2. Clear button resets history and creates NEW thread_id
    clear_btn.click(
        clear_data, 
        outputs=[chatbot, msg, session_state]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)