# Llama 3.1 70B on Amazon Bedrock (CML)

A Gradio chatbot that streams responses from Meta's **Llama 3.1 70B Instruct** model on **Amazon Bedrock**, plus a notebook that walks through the API step by step. No GPU is needed: the model runs on Bedrock and is billed per token (well under $1 per 1M tokens).

| File | What it is |
|---|---|
| `Llama_3.1_70B_Bedrock.ipynb` | Lab notebook: curl request, Python client, streaming, multi-turn chat, Gradio demo |
| `app.py` | The chatbot, run as a CML Application |
| `requirements.txt` | Python dependencies |

---

## Part 1: AWS setup (done once, by the lab admin)

### 1. Region
Use **us-east-1**. The lab calls Llama 3.1 70B Instruct through the US cross-region inference profile `us.meta.llama3-1-70b-instruct-v1:0`, which Bedrock may route to us-east-1, us-east-2 or us-west-2.

### 2. Make sure the model works in your account
1. In the AWS console, switch to that region and open **Amazon Bedrock**.
2. Open the **Model catalog**, search for **Llama 3.1 70B**, and open it in the **Playground**.
3. Send one test prompt. If the console asks you to request access to the model, do so.

### 3. Create credentials for the lab user
Create a dedicated IAM user with only the policy below, and generate an **access key** for it. Rotate or delete the key after each course.

### IAM policy for the lab identity
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
    "Resource": [
      "arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:inference-profile/us.meta.llama3-1-70b-instruct-v1:0",
      "arn:aws:bedrock:us-east-1::foundation-model/meta.llama3-1-70b-instruct-v1:0",
      "arn:aws:bedrock:us-east-2::foundation-model/meta.llama3-1-70b-instruct-v1:0",
      "arn:aws:bedrock:us-west-2::foundation-model/meta.llama3-1-70b-instruct-v1:0"
    ]
  }]
}
```
If you use a Bedrock API key, also allow `bedrock:CallWithBearerToken`.

### 4. Recommended
- **Budget**: add an AWS Budgets alert so usage can't run away.
- **Quotas**: a class of ~20 learners stays well within default Bedrock quotas. If you see `ThrottlingException`, request a quota increase for the model.

---

## Part 2: CML setup

When you create the project from this AMP, it asks for:

| Variable | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Access key ID of the lab IAM user |
| `AWS_SECRET_ACCESS_KEY` | Secret access key of the lab IAM user |
| `AWS_REGION` | `us-east-1` |
| `BEDROCK_MODEL_ID` | Leave the default, `us.meta.llama3-1-70b-instruct-v1:0` |

A Bedrock API key also works: set `AWS_BEARER_TOKEN_BEDROCK` under Project Settings > Advanced > Environment Variables instead of the two access-key variables. Don't set both, because the API key takes priority.

If you change a variable later, restart the session or application: environment variables are read only at start-up.

To install manually in a session:
```bash
pip3 install -r requirements.txt
pip3 install "gradio[oauth]==4.44.1" "uvicorn>=0.14.0" "fastapi==0.115.3"
GRADIO_SERVER_PORT=$CDSW_APP_PORT python3 app.py
```

---

## Troubleshooting

| Error | Cause / fix |
|---|---|
| `No AWS credentials found` | Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, then restart the session or application. |
| `Unable to locate credentials` | boto3 is older than 1.39.0 and ignores the API key. Run `pip3 install -U "boto3>=1.39.0"`. |
| `UnrecognizedClientException` / `403` | The access key or API key is wrong, expired or deleted. |
| `AccessDeniedException` | The model isn't enabled for the account, or a policy blocks Bedrock or API keys. See step 2. |
| `ValidationException ... model identifier is invalid` | Wrong `BEDROCK_MODEL_ID`, or the model isn't offered in `AWS_REGION`. |
| `ThrottlingException` | Too many requests at once. Wait and retry, or request a quota increase. |
| Connection timeout | The CML cluster can't reach `bedrock-runtime.<region>.amazonaws.com`. Check the cluster's egress/proxy. |
