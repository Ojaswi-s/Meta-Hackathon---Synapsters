"""
inference.py
Baseline inference script for the Meeting Notes Action Item Extraction environment.

MANDATORY environment variables:
  API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     The model identifier (e.g. meta-llama/Llama-3.3-70B-Instruct)
  HF_TOKEN       Your Hugging Face / API key

Usage:
  API_BASE_URL=https://router.huggingface.co/v1 \
  MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  HF_TOKEN=hf_your_token_here \
  python inference.py

Expected runtime: < 5 minutes on all 3 tasks.
"""

import os
import json
import time
import sys
import functools
from typing import List, Dict, Any, Optional

# Force all print statements to flush immediately, preventing validator log truncation
print = functools.partial(print, flush=True)

from openai import OpenAI
from server.models import MeetingAction, ExtractedActionItem
from server.client import MeetingEnv

# ── Required env vars ─────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# ── Inference config ──────────────────────────────────────────────────────────
MAX_STEPS = 5          # Synced with environment (agent gets up to 5 attempts to refine based on feedback)
TEMPERATURE = 0.2      # Slight temperature to prevent deterministic loops during refinement
MAX_TOKENS = 2000      # Expanded to handle detailed Chain of Thought reasoning

TASK_IDS = ["easy", "medium", "hard"]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert meeting analyst. Your job is to extract all 
action items from meeting transcripts.

For each action item you must identify:
- owner: the person responsible (use their first name or role)
- task: a concise description of what they will do, using the EXACT keywords spoken in the transcript where possible
- deadline: when it's due (null if not mentioned)
- priority: "high", "medium", or "low"

You MUST respond with ONLY a valid JSON object in this exact format. You must output the "reasoning" key FIRST so you can think step-by-step before making your final list:
{
  "reasoning": "First, evaluate the transcript step-by-step. Identify tasks, resolve assignment conflicts, and pinpoint deadlines.",
  "action_items": [
    {
      "owner": "string",
      "task": "string",
      "deadline": "string or null",
      "priority": "high|medium|low"
    }
  ],
  "is_final": true
}

Rules:
- Include EVERY action item, even implicit ones ("we should", "someone needs to").
- If responsibility changes, assign it to whoever accepted it LAST.
- If an existing task is reassigned mid-meeting, discard the old owner.
- DO NOT invent action items that aren't in the transcript.
- Use explicit terminology and acronyms exactly as found in the transcript.
- is_final should be true on your last attempt.
- Respond with JSON only — no markdown, no preamble.
"""


def call_llm(
    client: OpenAI,
    transcript: str,
    task_description: str,
    previous_feedback: Optional[str],
    step: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Call the LLM and parse its JSON response."""

    user_content = f"""TASK INSTRUCTIONS:
{task_description}

MEETING TRANSCRIPT:
{transcript}
"""

    if previous_feedback:
        user_content += f"\nFEEDBACK FROM LAST ATTEMPT:\n{previous_feedback}\n"

    is_last = step >= max_steps - 1
    user_content += f"\nThis is attempt {step + 1} of {max_steps}."
    if is_last:
        user_content += " Set is_final to true in your response."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=30,  # Prevent hanging on a slow router (20min total runtime limit)
    )

    raw = response.choices[0].message.content or "{}"

    # Indestructible JSON parser: extract from first { to last }
    start_idx = raw.find('{')
    end_idx = raw.rfind('}')
    if start_idx != -1 and end_idx != -1:
        raw = raw[start_idx:end_idx + 1]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [warn] JSON parse failed, using empty response. Raw: {raw[:200]}")
        parsed = {"action_items": [], "is_final": is_last, "reasoning": "parse error"}

    # Enforce is_final on last step
    if is_last:
        parsed["is_final"] = True

    return parsed


def run_task(client: OpenAI, env_client, task_id: str) -> Dict[str, Any]:
    """Run a single task episode and return results."""
    print(f"[START] task={task_id}")
    print(f"\n{'='*60}")
    print(f"Task: {task_id.upper()}")
    print(f"{'='*60}")

    episode_rewards = []
    final_f1 = 0.0
    final_score = 0.0
    
    try:
        # Reset environment
        try:
            result = env_client.reset(task_id=task_id)
        except Exception as exc:
            print(f"  [ERROR] Could not reset environment for task '{task_id}': {exc}")
            print(f"  Ensure the server is running at: {ENV_BASE_URL}")
            return {
                "task_id": task_id, "final_f1": 0.0, "best_f1": 0.0,
                "total_reward": 0.0, "steps_taken": 0, "rewards_per_step": [],
            }
        obs = result.observation

        print(f"Transcript length: {len(obs.transcript)} chars")
        print(f"Total items to find: {obs.total_items_in_task}")

        previous_feedback = None
        done = False  # track done from obs, not result (reset() wraps differently)

        for step in range(MAX_STEPS):
            if done:
                break

            print(f"\n  Step {step + 1}/{MAX_STEPS}")

            try:
                # Get LLM response
                action_data = call_llm(
                    client=client,
                    transcript=obs.transcript,
                    task_description=obs.task_description,
                    previous_feedback=previous_feedback,
                    step=step,
                    max_steps=MAX_STEPS,
                )
            except Exception as e:
                print(f"  [ERROR] LLM call failed: {e}")
                break

            items = action_data.get("action_items", [])
            print(f"  Agent extracted {len(items)} items")

            # Build action
            action = MeetingAction(
                action_items=[
                    ExtractedActionItem(
                        owner=item.get("owner", ""),
                        task=item.get("task", ""),
                        deadline=item.get("deadline"),
                        priority=item.get("priority", "medium"),
                    )
                    for item in items
                ],
                is_final=action_data.get("is_final", False),
                reasoning=action_data.get("reasoning"),
            )

            # Step environment
            try:
                result = env_client.step(action)
            except Exception as exc:
                print(f"  [ERROR] Environment step {step + 1} failed: {exc}")
                break
            obs = result.observation
            done = getattr(obs, 'done', False) or getattr(result, 'done', False)

            reward = result.reward or 0.0
            episode_rewards.append(reward)
            final_f1 = obs.current_f1
            final_score = reward
            previous_feedback = obs.step_feedback

            print(f"  Reward: {reward:.4f} | F1: {obs.current_f1:.4f} | "
                  f"Found: {obs.items_found_count}/{obs.total_items_in_task}")
            if obs.step_feedback:
                print(f"  Feedback: {obs.step_feedback}")

            print(f"[STEP] step={step + 1} reward={reward}")

            if done:
                break
    finally:
        try:
            state = env_client.state()
            best_f1 = state.best_f1
            steps_taken = state.step_count
        except Exception:
            best_f1 = 0.0
            steps_taken = len(episode_rewards)
            
        print(f"\n  Final best F1: {best_f1:.4f}")
        print(f"  Total reward: {sum(episode_rewards):.4f}")

        print(f"[END] task={task_id} score={best_f1} steps={steps_taken}")

    return {
        "task_id": task_id,
        "final_f1": final_f1,
        "best_f1": best_f1,
        "total_reward": sum(episode_rewards),
        "steps_taken": steps_taken,
        "rewards_per_step": episode_rewards,
    }


def main():
    print("Meeting Notes Action Item Extraction — Baseline Inference")
    print(f"Model:    {MODEL_NAME}")
    print(f"Env URL:  {ENV_BASE_URL}")
    print(f"API URL:  {API_BASE_URL}")

    if not API_KEY:
        raise ValueError("HF_TOKEN or API_KEY environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    start_time = time.time()

    connected = False
    max_retries = 10
    env_client = None
    
    # Retry logic for initial connection (Phase 2 env startup delay)
    for attempt in range(max_retries):
        try:
            env_client = MeetingEnv(base_url=ENV_BASE_URL).sync()
            with env_client:
                connected = True
                for task_id in TASK_IDS:
                    try:
                        task_result = run_task(client, env_client, task_id)
                        results.append(task_result)
                    except Exception as e:
                        print(f"  [ERROR] Task {task_id} raised exception: {e}")
                break  # Successful execution, break retry loop
        except Exception as e:
            print(f"[Attempt {attempt + 1}/{max_retries}] Exception during env connection/execution: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                print("Max retries reached. Could not complete execution.")

    if not connected:
        print("Exiting gracefully due to connection failure.")
        # Print fallback blocks to satisfy the validator parser and let us see the logs!
        for task_id in TASK_IDS:
            print(f"[START] task={task_id}")
            print(f"[END] task={task_id} score=0.0 steps=0")
        return {"error": "Failed to connect to environment server"}

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<10} {'Best F1':<12} {'Total Reward':<15} {'Steps'}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['task_id']:<10} "
            f"{r['best_f1']:<12.4f} "
            f"{r['total_reward']:<15.4f} "
            f"{r['steps_taken']}"
        )

    avg_f1 = sum(r["best_f1"] for r in results) / len(results)
    total_reward = sum(r["total_reward"] for r in results)
    print("-" * 50)
    print(f"{'AVERAGE':<10} {avg_f1:<12.4f} {total_reward:<15.4f}")
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Save results
    output = {
        "model": MODEL_NAME,
        "tasks": results,
        "avg_f1": avg_f1,
        "total_reward": total_reward,
        "runtime_seconds": round(elapsed, 1),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to baseline_results.json")

    return output


if __name__ == "__main__":
    main()
