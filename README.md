# GPT-2 Discord Bot

### Setup

- Install dependencies with:

```bash
pip install -r requirements.txt
```

- Run the script `download_model.sh` by:
```
sh download_model.sh 117M
```
_This should download the gpt-2 model_


- Create `auth.json`, and place it inside the `config` folder. Its content should be:

```json
{
   "token": "<your_token>",
   "client_id": "<client_id>"
}
```

### How to run

- Run the script with:

```bash
python gpt-chatbot-client.py
```

### Improvements

- Batch the message sends into `1990` chars per batch.
- Send typing indicator every 10 sec when generating text.
- Enable finetuning.