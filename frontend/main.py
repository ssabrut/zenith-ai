import gradio as gr

def chat_response(message, history):
    """
    Process user message and return bot response.
    
    Args:
        message: Current user message
        history: List of [user_msg, bot_msg] pairs
    
    Returns:
        Empty string (to clear input box)
    """
    # Simple echo bot example - replace with your actual logic
    bot_message = f"You said: {message}"
    
    # Add to history
    history.append([message, bot_message])
    
    return history, ""

def clear_chat():
    """Clear the chat history"""
    return [], ""

# Create the Gradio interface
with gr.Blocks(title="Chatbot Interface") as demo:
    gr.Markdown("# 💬 Conversational Chatbot")
    gr.Markdown("Start chatting with the bot below!")
    
    # Chatbot component
    chatbot = gr.Chatbot(
        value=[],
        height=500,
        avatar_images=(None, "🤖")
    )
    
    # Input row
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            scale=4,
            container=False
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")
    
    # Clear button
    clear_btn = gr.Button("Clear Chat", variant="secondary")
    
    # Event handlers
    msg.submit(chat_response, inputs=[msg, chatbot], outputs=[chatbot, msg])
    submit_btn.click(chat_response, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])

# Launch the app
if __name__ == "__main__":
    demo.launch()