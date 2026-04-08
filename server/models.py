"""
meeting_env/models.py
Typed Pydantic models for the Meeting Notes Action Item Extraction environment.
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


class ExtractedActionItem(BaseModel):
    """A single action item extracted from the meeting transcript."""
    owner: str = Field(..., description="Full name or role of the responsible person")
    task: str = Field(..., description="Clear description of what needs to be done")
    deadline: Optional[str] = Field(
        None, description="Deadline string e.g. 'Friday', 'end of sprint', '2024-12-01'"
    )
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Priority level: 'high', 'medium', or 'low'"
    )


# ── Agent Action ────────────────────────────────────────────────────────────

class MeetingAction(Action):
    """What the agent submits each step — a list of extracted action items."""
    action_items: List[ExtractedActionItem] = Field(
        ...,
        description="All action items the agent has extracted from the transcript"
    )
    is_final: bool = Field(
        default=False,
        description="Set True to end the episode and lock in the final score"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Optional: agent's reasoning about its extraction decisions"
    )


# ── Environment Observation ──────────────────────────────────────────────────

class MeetingObservation(Observation):
    """What the agent receives after each step."""
    transcript: str = Field(..., description="The full meeting transcript to analyze")
    task_description: str = Field(
        ..., description="Instructions describing what the agent must extract"
    )
    step_feedback: Optional[str] = Field(
        None,
        description="Feedback from the last submission — hints without revealing answers"
    )
    items_found_count: int = Field(
        default=0,
        description="Number of ground-truth action items correctly identified so far"
    )
    total_items_in_task: int = Field(
        default=0,
        description="Total number of action items that exist in this task"
    )
    current_f1: float = Field(
        default=0.0,
        description="Current F1 score (0.0–1.0) based on last submission"
    )
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: Optional[float] = Field(
        default=None, description="Reward signal from last step"
    )
    step_count: int = Field(
        default=0,
        description="Current step number (0 on reset, increments with each step call)"
    )
    steps_remaining: int = Field(
        default=5,
        description="Steps remaining in this episode — set is_final=True on your last attempt"
    )


# ── Episode State ────────────────────────────────────────────────────────────

class MeetingState(State):
    """Episode-level metadata tracked server-side."""
    task_id: str = Field(default="", description="Which task is active: easy/medium/hard")
    task_name: str = Field(default="", description="Human-readable task name")
    best_f1: float = Field(default=0.0, description="Best F1 score achieved this episode")
    attempts: int = Field(default=0, description="Number of submission attempts")
    is_complete: bool = Field(default=False, description="Whether episode ended")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated")
