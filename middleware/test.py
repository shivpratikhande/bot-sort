import cv2
import json
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def timestamp_to_seconds(ts: str) -> int:
    minutes, seconds = ts.split(":")
    return int(minutes) * 60 + int(seconds)

def extract_frame(video_path: str, timestamp_seconds: int):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_seconds * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Failed to read frame at {timestamp_seconds}s")

    return frame


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_model_pricing(model: str) -> dict:
    """Get pricing information for different GPT models (per 1M tokens)"""
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
    }
    return pricing.get(model, {"input": 2.50, "output": 10.00})  # Default to gpt-4o


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate cost based on token usage"""
    pricing = get_model_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def analyze_frame_with_gpt4(image_path: str, event_info: dict, api_key: str, model: str = "gpt-4o") -> tuple:
    """Send frame to GPT-4 Vision for analysis. Returns (description, input_tokens, output_tokens, cost)"""
    try:
        client = OpenAI(api_key=api_key)
        
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        
        # Create prompt with context
        objects_detected = ", ".join([f"{count} {obj}" for obj, count in event_info["objects"].items()])
        changes_str = ", ".join(event_info["changes"])
        
        prompt = f"""Analyze this video frame captured at timestamp {event_info['timestamp']}.

Detected objects: {objects_detected}
Recent changes: {changes_str}

Please provide:
1. A detailed description of what's happening in the scene
2. The context and setting (indoor/outdoor, location type, time of day if visible)
3. Any notable activities or interactions
4. Overall scene summary

Keep it concise but informative."""

        # Call GPT-4 Vision API
        response = client.chat.completions.create(
            model=model,  # Uses model from .env or parameter
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Extract token usage and calculate cost
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cost = calculate_cost(input_tokens, output_tokens, model)
        
        return response.choices[0].message.content, input_tokens, output_tokens, cost
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}", 0, 0, 0.0


def process_frame(frame, event_info):
    print("Processing frame for event:", event_info["timestamp"])
    print("Objects in frame:", event_info["objects"])
    print("Changes:", event_info["changes"])
    print("Frame shape:", frame.shape)
    print("------")

def extract_frames_from_events(video_path: str, json_path: str, save_frames=False, use_gpt4=False, api_key=None, model=None):
    # Load JSON file
    with open(json_path, "r") as f:
        events = json.load(f)

    save_dir = "extracted_frames"
    os.makedirs(save_dir, exist_ok=True)
    
    # Store all analyses for final report
    all_analyses = []
    
    # Get model from parameter or environment
    gpt_model = model or os.getenv("GPT_MODEL", "gpt-4o")
    
    # Cost tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    for i, event in enumerate(events):
        timestamp = event["timestamp"]
        current_seconds = timestamp_to_seconds(timestamp)
        
        # Calculate middle timestamp between current and next event
        if i < len(events) - 1:
            next_timestamp = events[i + 1]["timestamp"]
            next_seconds = timestamp_to_seconds(next_timestamp)
            # Take the average (middle) between current and next
            middle_seconds = (current_seconds + next_seconds) // 2
            print(f"\n📌 Extracting frame for event {i} at timestamp {timestamp} (using middle frame at {middle_seconds}s between {current_seconds}s and {next_seconds}s)")
        else:
            # For the last event, just use the current timestamp
            middle_seconds = current_seconds
            print(f"\n📌 Extracting frame for event {i} at timestamp {timestamp} ({current_seconds}s - last event)")

        # Extract frame at middle timestamp
        frame = extract_frame(video_path, middle_seconds)

        # Pass to your custom function
        process_frame(frame, event)

        # Optionally save the frame as an image
        if save_frames:
            save_path = os.path.join(save_dir, f"event_{i}_{timestamp.replace(':', '_')}_mid{middle_seconds}s.jpg")
            cv2.imwrite(save_path, frame)
            print(f"💾 Saved frame to: {save_path}")
            
            # Analyze with GPT-4 Vision if enabled
            if use_gpt4 and api_key:
                print(f"🤖 Analyzing frame with GPT-4 Vision ({gpt_model})...")
                description, input_tokens, output_tokens, cost = analyze_frame_with_gpt4(save_path, event, api_key, gpt_model)
                
                # Update totals
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_cost += cost
                
                print(f"📝 GPT-4 Analysis:\n{description}")
                print(f"💰 Cost: ${cost:.6f} (Input: {input_tokens} tokens, Output: {output_tokens} tokens)\n")
                
                # Store analysis
                all_analyses.append({
                    "event_number": i,
                    "timestamp": timestamp,
                    "frame_time": middle_seconds,
                    "objects": event["objects"],
                    "changes": event["changes"],
                    "gpt4_description": description,
                    "tokens": {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": input_tokens + output_tokens
                    },
                    "cost": cost
                })

    print("\n✅ All event frames processed.")
    
    # Display cost summary
    if use_gpt4 and all_analyses:
        print("\n" + "=" * 80)
        print("💰 COST SUMMARY")
        print("=" * 80)
        print(f"Model: {gpt_model}")
        print(f"Total frames analyzed: {len(all_analyses)}")
        print(f"Total input tokens: {total_input_tokens:,}")
        print(f"Total output tokens: {total_output_tokens:,}")
        print(f"Total tokens: {total_input_tokens + total_output_tokens:,}")
        print(f"Total cost: ${total_cost:.6f}")
        print(f"Average cost per frame: ${total_cost/len(all_analyses):.6f}")
        print("=" * 80 + "\n")
    
    # Save all analyses to a JSON file
    if use_gpt4 and all_analyses:
        # Add cost summary to JSON
        analysis_data = {
            "model": gpt_model,
            "summary": {
                "total_frames": len(all_analyses),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "total_cost": total_cost,
                "average_cost_per_frame": total_cost / len(all_analyses)
            },
            "analyses": all_analyses
        }
        
        analysis_path = os.path.join(save_dir, "gpt4_analyses.json")
        with open(analysis_path, "w") as f:
            json.dump(analysis_data, f, indent=2)
        print(f"📊 GPT-4 analyses saved to: {analysis_path}")
        
        # Also save as readable text file
        text_path = os.path.join(save_dir, "gpt4_analyses.txt")
        with open(text_path, "w") as f:
            f.write("GPT-4 Vision Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {gpt_model}\n")
            f.write(f"Total Frames Analyzed: {len(all_analyses)}\n")
            f.write(f"Total Input Tokens: {total_input_tokens:,}\n")
            f.write(f"Total Output Tokens: {total_output_tokens:,}\n")
            f.write(f"Total Cost: ${total_cost:.6f}\n")
            f.write(f"Average Cost Per Frame: ${total_cost/len(all_analyses):.6f}\n")
            f.write("=" * 80 + "\n\n")
            
            for analysis in all_analyses:
                f.write(f"Event {analysis['event_number']} | Timestamp: {analysis['timestamp']}\n")
                f.write(f"Objects: {analysis['objects']}\n")
                f.write(f"Changes: {', '.join(analysis['changes'])}\n")
                f.write(f"Tokens: {analysis['tokens']['total']} (Input: {analysis['tokens']['input']}, Output: {analysis['tokens']['output']})\n")
                f.write(f"Cost: ${analysis['cost']:.6f}\n")
                f.write(f"\nGPT-4 Description:\n{analysis['gpt4_description']}\n")
                f.write("-" * 80 + "\n\n")
        print(f"📄 Human-readable report saved to: {text_path}")


if __name__ == "__main__":
    VIDEO_PATH = "../test3.mp4"        
    JSON_PATH = "../runs/detect/exp13/object_change_log.json"
    
    # Load API key from .env file
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Load model from .env file (defaults to gpt-4o)
    GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")
    
    # Enable GPT-4 Vision analysis
    USE_GPT4 = True  # Set to False to skip GPT-4 analysis
    
    # Check if API key is set
    if USE_GPT4 and not OPENAI_API_KEY:
        print("⚠️  Warning: OPENAI_API_KEY not found in .env file!")
        print("Please edit the .env file and add your API key.")
        print("Continuing without GPT-4 analysis...\n")
        USE_GPT4 = False

    extract_frames_from_events(
        video_path=VIDEO_PATH,
        json_path=JSON_PATH,
        save_frames=True,
        use_gpt4=USE_GPT4,
        api_key=OPENAI_API_KEY,
        model=GPT_MODEL
    )
