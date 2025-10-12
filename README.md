## ProtoPrompt

## Requirements

```bash
pip install -r requirements.txt
```

**Note**: Before running, you need to put your OpenAI api key to the .env file
```bash
# create .env
OPENAI_API_KEY = "YOUR-API-KEY"
```

## Datasets
Some datasets are provided, while others will be automatically downloaded via Hugging Face.
```bash
# log in
pip install huggingface_hub
huggingface-cli login
```

## Run
Run
```bash
python src/main.py --config_dir configs/main_penguin.yaml 
```
or
```bash
uv run python src/main.py --config_dir configs/main_penguin.yaml 
```