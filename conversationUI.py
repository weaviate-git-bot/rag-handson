from conversationService import get_llm_response
import gradio as gr
import readchar
import signal


def exit_ui():
    gr.close_all()
    exit(0)


def handler(signum, frame):
    msg = "Você realmente deseja encerrar a interface gráfica? s/n"
    print(msg, end="", flush=True)
    res = readchar.readchar()
    if res == "s":
        print("")
        exit_ui()
    else:
        print("", end="\r", flush=True)
        print(" " * len(msg), end="", flush=True)  # clear the printed line
        print("    ", end="\r", flush=True)

signal.signal(signal.SIGINT, handler)

text_box = gr.Textbox("Pergunta: ")
demo = gr.ChatInterface(
    get_llm_response,
    title="RAG Hands On",
    textbox=text_box,
    submit_btn=gr.Button("Perguntar"),
    retry_btn=gr.Button("Tentar novamente"),
    clear_btn=gr.Button("Limpar"),
    undo_btn=gr.Button("Desfazer"),
)

if __name__ == "__main__":
    demo.launch()
