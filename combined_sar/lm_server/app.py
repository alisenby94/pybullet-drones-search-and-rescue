"""
LM Server for Drone Mission Planning

FastAPI server using Qwen2.5-1.5B-Instruct for generating mission plans.
"""

from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
from typing import Dict, Any
import os
import warnings
import gc

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

app = FastAPI(title="Local LLM Server")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "running",
        "model": MODEL_NAME,
        "device": str(device)
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_CONTEXT_TOKENS = 1024  # Qwen 2.5 1.5B context window
MIN_NEW_TOKENS = 32        # minimum generation budget (increased)
MAX_NEW_TOKENS = 384       # cap generation to avoid overrun (increased for complete JSON)

# Determine device with better handling
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using Apple Metal Performance Shaders (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA on device {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print(f"Using CPU")

print(f"Selected device: {device}")
print(f"Torch version: {torch.__version__}")

# Load tokenizer
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer loaded successfully")

# Load model with improved settings
print(f"Loading model {MODEL_NAME}...")
print(f"This may take a minute on first run (downloading ~3GB)...")

try:
    # Try to use device_map with accelerate if available
    try:
        import accelerate
        print(f"accelerate {accelerate.__version__} is available, using device_map='auto'")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,  # Use dtype instead of torch_dtype
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print(f"Model loaded with device_map='auto'")
    except Exception as e:
        print(f"device_map='auto' failed ({type(e).__name__}), trying direct device placement")
        # Fallback to direct device placement
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=None
        )
        model = model.to(device)
        print(f"Model loaded and moved to {device}")
        
except Exception as e:
    print(f"ERROR loading model: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    model = None

if model:
    print(f"Model ready on {device}")
    model.eval()  # Set to evaluation mode
    print(f"Model set to evaluation mode")

def count_tokens(text: str) -> int:
    """Count tokens using raw encode (legacy). Prefer count_prompt_tokens for chat prompts."""
    return len(tokenizer.encode(text))


def count_prompt_tokens(text: str) -> int:
    """Count tokens exactly as the model will see them (includes special tokens)."""
    return int(tokenizer(text, return_tensors="pt").input_ids.shape[-1])


def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """Trim a string to at most max_tokens using the model tokenizer."""
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

TOOL_DEFINITIONS = [
    {
        "name": "explore",
        "description": "Plan exploration in a direction with a distance in meters.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                },
                "distance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 19,
                    "description": "Distance in meters (< 20)"
                }
            },
            "required": ["direction", "distance"]
        }
    },
    {
        "name": "scan_surroundings",
        "description": "Scan detected obstacle for details.",
        "parameters": {
            "type": "object",
        }
    },
    {
        "name": "explore_obstacle",
        "description": "Explore and navigate around obstacles in a specific direction.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                }
            },
            "required": ["direction"]
        }
    }
]

TOOLS_JSON = json.dumps(TOOL_DEFINITIONS, indent=2)

def build_system_prompt() -> str:
    system_prompt = f"""You are a drone mission planner. Respond with ONLY valid JSON.

IMPORTANT actions (use only these names):
- "explore"
- "scan_surroundings"
- "explore_obstacle"

Available tools:
{TOOLS_JSON}

Planning rules:
- Always produce 3-5 actions.
- If obstacles are present, include at least one "explore_obstacle" facing the nearest obstacle direction. Do not return only "explore" actions when obstacles exist.
- Prefer starting with a scan ("scan_surroundings" or "explore_obstacle") before moving.
- Avoid repeating the exact same action+direction pair.

JSON Structure:
{{
  "reasoning": "<brief reasoning for plan>",
  "actions": [
    {{
      "action": "<action_name>",
      "parameters": {{<action_parameters>}}
    }},
    ...
  ]
}}
Valid directions: N, NE, E, SE, S, SW, W, NW
Valid distances: any integer meters < 30 (e.g., 5-29; default to 10/20/30 if unsure)

Return ONLY the JSON."""
    return system_prompt

def build_user_prompt(observations: str, history: str) -> str:
    return f"""Environment observations:
{observations}

Mission history:
{history}

Provide a JSON mission plan:"""


def _strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments from a JSON-like string."""
    # Remove /* block comments */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Remove // line comments
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return text

@app.post("/plan")
async def plan_route(request: Dict[str, Any]) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
    
    try:
        print(f"\n{'='*60}")
        print(f"Received planning request")
        print(f"{'='*60}")
        
        observations = request.get("observations", "")
        history = request.get("history", "")
        
        print(f"Observations length: {len(observations)} chars")
        print(f"History length: {len(history)} chars")

        system_prompt = build_system_prompt()

        # Iteratively truncate observations/history to fit the 1024-token window
        observations_trimmed = observations
        history_trimmed = history
        for _ in range(3):  # up to 3 tightening passes
            user_prompt = build_user_prompt(observations_trimmed, history_trimmed)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            token_count = count_prompt_tokens(full_prompt)
            print(f"Total prompt tokens: {token_count}")

            if token_count <= MAX_CONTEXT_TOKENS:
                break

            # Compute a rough budget split for obs/history after accounting for fixed parts
            # Reserve 200 tokens for system + formatting, split remaining equally.
            budget = max(100, MAX_CONTEXT_TOKENS - 200)
            obs_budget = budget // 2
            hist_budget = budget - obs_budget
            observations_trimmed = truncate_text_by_tokens(observations_trimmed, obs_budget)
            history_trimmed = truncate_text_by_tokens(history_trimmed, hist_budget)
        else:
            raise HTTPException(status_code=400, detail=f"Prompt exceeds token limit after truncation: {token_count}/{MAX_CONTEXT_TOKENS}")

        # Rebuild final prompt after truncation pass
        user_prompt = build_user_prompt(observations_trimmed, history_trimmed)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        token_count = count_prompt_tokens(full_prompt)
        print(f"Total prompt tokens (final): {token_count}")
        
        if token_count > MAX_CONTEXT_TOKENS:
            raise HTTPException(status_code=400, detail=f"Prompt exceeds token limit: {token_count}/{MAX_CONTEXT_TOKENS}")
        
        print(f"Generating response with {MODEL_NAME} (sampling enabled)...")
        
        # Move to device and generate
        with torch.no_grad():  # Disable gradient computation for inference
            # Pre-emptively clear cache to avoid fragmentation on small GPUs
            if device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            # Tokenize with attention mask to avoid warnings
            inputs = tokenizer(full_prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Ensure generation + prompt stays within context window
            available_gen = max(MIN_NEW_TOKENS, MAX_CONTEXT_TOKENS - token_count - 2)
            gen_tokens = min(MAX_NEW_TOKENS, available_gen)

            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_tokens,
                do_sample=True,   # Enable sampling to reduce repetitive plans
                temperature=0.7,  # Slightly higher for more variation
                top_p=0.9,
                repetition_penalty=1.1,  # Prevent repetitive empty output
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode only the generated tokens (not the input prompt)
        # output_ids shape is [1, total_tokens], input_ids shape is [1, prompt_tokens]
        prompt_len = input_ids.shape[1]
        generated_ids = output_ids[0, prompt_len:]  # Get only the new tokens
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        # Clean up any special tokens manually if needed
        output_text = output_text.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
        
        # Clear CUDA cache to free memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        # Debug: Print raw output
        print(f"\n=== RAW MODEL OUTPUT (generated tokens only) ===")
        print(f"Input shape: {input_ids.shape}, Output shape: {output_ids[0].shape}")
        print(f"First 200 chars: {repr(output_text[:200])}")
        print(f"Full output:\n{output_text}")
        print(f"=== END RAW OUTPUT ===\n")
        
        # Extract JSON from output (may have extra text)
        response_json_str = None
        
        # Remove common markdown wrappers
        cleaned_output = output_text.replace('```json', '|||JSONSTART|||').replace('```', '').strip()
        
        # Look for JSON after markdown marker or at the start
        search_text = cleaned_output
        if '|||JSONSTART|||' in cleaned_output:
            # Find JSON after the markdown marker
            parts = cleaned_output.split('|||JSONSTART|||')
            if len(parts) > 1:
                search_text = parts[-1]  # Get the part after the last marker
        
        # Find outermost braces in the search text
        start = search_text.find("{")
        if start != -1:
            # Find matching closing brace
            brace_count = 0
            end = -1
            for i in range(start, len(search_text)):
                if search_text[i] == '{':
                    brace_count += 1
                elif search_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if end != -1:
                response_json_str = search_text[start:end]
            elif brace_count > 0:
                # Incomplete JSON - try to close it
                print(f"WARNING: Incomplete JSON detected, attempting to complete...")
                incomplete = search_text[start:]
                # Try to salvage by closing unclosed structures
                while brace_count > 0:
                    # Check if we're in an array
                    if incomplete.rstrip().endswith(','):
                        incomplete = incomplete.rstrip()[:-1]  # Remove trailing comma
                    incomplete += ']}' if '"actions"' in incomplete else '}'
                    brace_count -= 1
                response_json_str = incomplete
                print(f"Completed JSON: {response_json_str[:200]}...")
        
        if not response_json_str:
            print(f"ERROR: Could not find complete JSON in output")
            # Try to salvage with fallback
            print(f"Attempting fallback plan generation...")
            response_json_str = '{"reasoning": "exploration", "actions": [{"action": "explore", "parameters": {"direction": "N", "distance": 10}}]}'
        
        print(f"\n=== EXTRACTED JSON ===")
        print(response_json_str[:500] if len(response_json_str) > 500 else response_json_str)
        print(f"=== END EXTRACTED JSON ===\n")
        
        response_json = None
        try:
            # Remove JavaScript-style comments before parsing
            response_json_str_clean = _strip_json_comments(response_json_str)
            response_json = json.loads(response_json_str_clean)
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON decode failed: {e}")
            print(f"Attempted to parse: {response_json_str[:500]}")
            # Try multiple cleanup strategies
            for cleanup_attempt in range(3):
                try:
                    if cleanup_attempt == 0:
                        # Remove trailing commas
                        cleaned = response_json_str_clean.replace(",}", "}").replace(",]", "]")
                    elif cleanup_attempt == 1:
                        # Fix single quotes to double quotes
                        cleaned = response_json_str_clean.replace("'", '"')
                        cleaned = cleaned.replace(",}", "}").replace(",]", "]")
                    else:
                        # Remove all whitespace and newlines, fix quotes
                        import re
                        cleaned = re.sub(r'\s+', ' ', response_json_str_clean)
                        cleaned = cleaned.replace(",}", "}").replace(",]", "]")
                        cleaned = cleaned.replace("'", '"')
                    
                    response_json = json.loads(cleaned)
                    print(f"SUCCESS: Parsed after cleanup attempt {cleanup_attempt + 1}")
                    break
                except:
                    if cleanup_attempt == 2:
                        # All attempts failed, use fallback
                        print("All cleanup attempts failed, using fallback plan")
                        response_json = {
                            "reasoning": "fallback due to parse error",
                            "actions": [{"action": "explore", "parameters": {"direction": "N", "distance": 10}}]
                        }
                    continue

        # Ensure we always have a response
        if response_json is None:
            response_json = {
                "reasoning": "fallback - no valid JSON produced",
                "actions": [{"action": "explore", "parameters": {"direction": "N", "distance": 10}}]
            }

        return response_json
    except HTTPException:
        # Re-raise the custom handled HTTP exceptions
        raise 
    except Exception as e:
        print(f"An unhandled error occurred: {e}") 
        # This is what's causing your silent 500.
        # You can raise a specific 500 here if you want to provide a
        # better message, but printing the error is key for debugging.
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {type(e).__name__}")
