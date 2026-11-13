import os

import boto3
import gradio as gr

# ---------------------------------------------------------------------------
# Amazon Bedrock configuration (set as CML project / application env vars)
#   AWS_BEARER_TOKEN_BEDROCK  Bedrock API key (or use AWS_ACCESS_KEY_ID +
#                             AWS_SECRET_ACCESS_KEY instead)
#   AWS_REGION                Region where the model is enabled, e.g. us-east-1
#   BEDROCK_MODEL_ID          Optional, defaults to Llama 3.1 70B
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "").strip() or "us.meta.llama3-1-70b-instruct-v1:0"
REGION = os.environ.get("AWS_REGION", "").strip() or "us-east-1"
MODEL_NAME = "Llama 3.1 70B"

has_api_key = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip())
has_iam_keys = bool(os.environ.get("AWS_ACCESS_KEY_ID", "").strip()) and bool(
    os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
)
if not (has_api_key or has_iam_keys):
    raise RuntimeError(
        "No AWS credentials found. Set AWS_BEARER_TOKEN_BEDROCK (Bedrock API key), "
        "or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, under Project Settings > "
        "Advanced > Environment Variables, then restart."
    )

client = boto3.client("bedrock-runtime", region_name=REGION)


def format_prompt(message, history):
    """Convert Gradio chat history into the Bedrock Converse messages format."""
    turns = []
    for turn in history:
        if isinstance(turn, dict):  # Gradio "messages" format
            turns.append((turn["role"], turn["content"]))
        else:  # Gradio "tuples" format: [user, assistant]
            user_prompt, bot_response = turn
            turns.append(("user", user_prompt))
            turns.append(("assistant", bot_response))
    turns.append(("user", message))

    # Converse needs non-empty text and alternating user/assistant roles.
    messages = []
    for role, text in turns:
        if not text:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"][0]["text"] += "\n" + text
        else:
            messages.append({"role": role, "content": [{"text": text}]})
    return messages


def generate(
    prompt,
    history,
    temperature=0.9,
    max_new_tokens=256,
    top_p=0.95,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2

    response = client.converse_stream(
        modelId=MODEL_ID,
        messages=format_prompt(prompt, history),
        inferenceConfig={
            "temperature": temperature,
            "topP": float(top_p),
            "maxTokens": int(max_new_tokens),
        },
    )

    output = ""
    for event in response["stream"]:
        delta = event.get("contentBlockDelta", {}).get("delta", {})
        if "text" in delta:
            output += delta["text"]
            yield output

    return output


additional_inputs = [
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=64,
        maximum=1048,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
]


css = """
  #mkd {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""


with gr.Blocks(css=css) as demo:

    gr.HTML(
        f"<h1><center>{MODEL_NAME} Instruct</center></h1>"
    )

    gr.HTML(
        "<h3><center>"
        f"In this demo, you can chat with the {MODEL_NAME} model "
        "on Amazon Bedrock. 💬"
        "</center></h3>"
    )

    gr.HTML(
        "<h3><center>"
        "Learn more about the model "
        "<a href='https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html'>"
        "here"
        "</a>. 📚"
        "</center></h3>"
    )

    gr.ChatInterface(
        generate,
        additional_inputs=additional_inputs,
        examples=[
            ["What is the secret to life?"],
            ["Write me a recipe for pancakes."]
        ]
    )


demo.queue(
    default_concurrency_limit=75,
    max_size=100
).launch(debug=True)
