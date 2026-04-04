import gradio as gr

def add_sentences(Sen1, Sen2):
    return Sen1 + " " + Sen2

# Define the interface
demo = gr.Interface(
    fn=add_sentences, 
	inputs = [
		gr.Textbox(label="Input 1"),
		gr.Textbox(label="Input 2")
	],
	outputs = gr.Textbox(label="Output")
)

# Launch the interface
demo.launch()

