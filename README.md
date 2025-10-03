## ACL 2025 Submission--Code Appendix

## Requirements
If you are using conda environment:
```bash
conda create -n arr
conda activate arr
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

Or, if you use uv:
```bash
uv run python src/main.py --config_dir configs/main_penguin.yaml 
```