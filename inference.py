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
from typing import List, Dict, Any, Optional

from openai import OpenAI
from server.models import MeetingAction, ExtractedActionItem
from server.client import MeetingEnv

# ── Required env vars ─────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# ── Inference config ──────────────────────────────────────────────────────────
MAX_STEPS = 3          # Max refinement steps per task
TEMPERATURE = 0.1      # Low temperature for deterministic extraction
MAX_TOKENS = 1200

TASK_IDS = ["easy", "medium", "hard"]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert meeting analyst. Your job is to extract all 
action items from meeting transcripts.

For each action item you must identify:
- owner: the person responsible (use their first name or role)
- task: a clear, specific description of what they will do
- deadline: when it's due (null if not mentioned)
- priority: "high", "medium", or "low"

You MUST respond with ONLY a valid JSON object in this exact format:
{
  "action_items": [
    {
      "owner": "string",
      "task": "string",
      "deadline": "string or null",
      "priority": "high|medium|low"
    }
  ],
  "is_final": true,
  "reasoning": "brief explanation of your extraction decisions"
}

Rules:
- Include EVERY action item, even implicit ones
- If two people discuss taking the same task, assign it to whoever accepted it LAST
- Do not invent action items that aren't in the transcript
- is_final should be true on your last attempt
- Respond with JSON only — no markdown, no preamble
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

    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

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
    print(f"\n{'='*60}")
    print(f"Task: {task_id.upper()}")
    print(f"{'='*60}")

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

    episode_rewards = []
    final_f1 = 0.0
    final_score = 0.0
    previous_feedback = None

    for step in range(MAX_STEPS):
        if result.done:
            break

        print(f"\n  Step {step + 1}/{MAX_STEPS}")

        # Get LLM response
        action_data = call_llm(
            client=client,
            transcript=obs.transcript,
            task_description=obs.task_description,
            previous_feedback=previous_feedback,
            step=step,
            max_steps=MAX_STEPS,
        )

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

        reward = result.reward or 0.0
        episode_rewards.append(reward)
        final_f1 = obs.current_f1
        final_score = reward
        previous_feedback = obs.step_feedback

        print(f"  Reward: {reward:.4f} | F1: {obs.current_f1:.4f} | "
              f"Found: {obs.items_found_count}/{obs.total_items_in_task}")
        if obs.step_feedback:
            print(f"  Feedback: {obs.step_feedback}")

        if obs.done:
            break

    state = env_client.state()
    print(f"\n  Final best F1: {state.best_f1:.4f}")
    print(f"  Total reward: {sum(episode_rewards):.4f}")

    return {
        "task_id": task_id,
        "final_f1": final_f1,
        "best_f1": state.best_f1,
        "total_reward": sum(episode_rewards),
        "steps_taken": state.step_count,
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

    env_client = MeetingEnv(base_url=ENV_BASE_URL).sync()

    results = []
    start_time = time.time()

    with env_client:
        for task_id in TASK_IDS:
            task_result = run_task(client, env_client, task_id)
            results.append(task_result)

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
