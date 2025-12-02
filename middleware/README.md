# Frame Analysis with GPT-4 Vision

This script extracts frames from video at change detection points and optionally analyzes them with GPT-4 Vision.

## Setup

1. Install required packages:
```bash
pip install openai python-dotenv
```

2. Configure your OpenAI API key:

**Create a `.env` file in the middleware directory:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
nano .env
```

**Or set it directly:**
```bash
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env
echo "GPT_MODEL=gpt-4o" >> .env
```

## Getting Your API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)
5. Paste it in your `.env` file

## Usage

### Without GPT-4 Analysis (Free)
Set `USE_GPT4 = False` in the script, then run:
```bash
python test.py
```

### With GPT-4 Vision Analysis
Set `USE_GPT4 = True` and provide your API key, then run:
```bash
python test.py
```

## Output Files

- `extracted_frames/event_*.jpg` - Extracted frame images
- `extracted_frames/gpt4_analyses.json` - Machine-readable analysis results
- `extracted_frames/gpt4_analyses.txt` - Human-readable analysis report

## Cost Estimate

GPT-4 Vision API costs approximately:
- **gpt-4o**: ~$0.005 per image (cheaper, recommended)
- **gpt-4-turbo**: ~$0.01 per image
- **gpt-4o-mini**: ~$0.001 per image (cheapest)

For 14 frames ≈ $0.07 with gpt-4o or $0.014 with gpt-4o-mini

## Configuration

### .env File
```bash
OPENAI_API_KEY=sk-your-actual-key-here
GPT_MODEL=gpt-4o  # Options: gpt-4o, gpt-4o-mini, gpt-4-turbo
```

### Script Settings (test.py)
- `VIDEO_PATH`: Path to your video file
- `JSON_PATH`: Path to the change log JSON
- `USE_GPT4`: Enable/disable GPT-4 analysis (True/False)
