import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def chatbot_response(message, history):
    conversation = ""
    
    if history:
        recent = history[-2:] if len(history) > 2 else history
        for human_msg, bot_msg in recent:
            conversation += f"{human_msg}\n{bot_msg}\n"
    
    conversation += message
    
    inputs = tokenizer(conversation, return_tensors="pt", max_length=128, truncation=True)
    
    reply_ids = model.generate(
        inputs["input_ids"],
        max_length=128,
        min_length=20,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        num_beams=1,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
    )
    
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response.strip()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💬 AI Chatbot")
    
    chatbot = gr.Chatbot(height=450, bubble_full_width=False, avatar_images=(None, "🤖"))
    
    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message...", show_label=False, scale=4)
        send = gr.Button("Send", scale=1)
    
    clear = gr.Button("Clear Chat")
    
    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        bot_message = chatbot_response(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
