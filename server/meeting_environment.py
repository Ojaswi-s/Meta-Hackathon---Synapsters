"""
meeting_env/server/meeting_environment.py
Core environment logic implementing the OpenEnv Environment interface.
"""
import uuid
from typing import Optional

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

from .models import MeetingAction, MeetingObservation, MeetingState
from .tasks import TASKS
from .grader import grade, generate_feedback

MAX_STEPS = 5  # Agent gets up to 5 attempts to refine its extraction


class MeetingEnvironment(Environment):
    """
    Meeting Notes Action Item Extraction environment.

    Episode flow:
      1. reset(task_id="easy"|"medium"|"hard") → returns transcript + instructions
      2. Agent calls step() with extracted action items (can refine multiple times)
      3. Each step returns F1-based reward + non-answer feedback
      4. Episode ends when agent sets is_final=True OR max steps reached
    """

    def __init__(self):
        super().__init__()
        self._state = MeetingState(
            episode_id="",
            step_count=0,
            task_id="",
            task_name="",
            best_f1=0.0,
            attempts=0,
            is_complete=False,
            cumulative_reward=0.0,
        )
        self._current_task: Optional[dict] = None
        self._last_reward: float = 0.0
        self._last_done: bool = False
        self._last_obs: Optional[MeetingObservation] = None

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> MeetingObservation:
        """
        Start a new episode.
        Args:
            task_id: "easy", "medium", or "hard"
        Returns:
            Initial observation with the transcript and task instructions.
        """
        if task_id not in TASKS:
            task_id = "easy"

        task = TASKS[task_id]
        self._current_task = task

        self._state = MeetingState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            task_name=task["name"],
            best_f1=0.0,
            attempts=0,
            is_complete=False,
            cumulative_reward=0.0,
        )
        self._last_reward = 0.0
        self._last_done = False

        obs = MeetingObservation(
            transcript=task["transcript"],
            task_description=task["description"],
            step_feedback=None,
            items_found_count=0,
            total_items_in_task=len(task["ground_truth"]),
            current_f1=0.0,
            done=False,
            reward=None,
            step_count=0,
            steps_remaining=MAX_STEPS,
        )
        self._last_obs = obs
        return obs

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: MeetingAction) -> MeetingObservation:
        """
        Submit extracted action items for grading.
        Returns an observation with feedback and the current score.
        """
        if self._current_task is None:
            return MeetingObservation(
                transcript="No task loaded. Call reset() first.",
                task_description="",
                step_feedback="Environment not initialized.",
                items_found_count=0,
                total_items_in_task=0,
                current_f1=0.0,
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1
        self._state.attempts += 1

        # Grade the submission
        agent_items = [item.model_dump() for item in action.action_items]
        ground_truth = self._current_task["ground_truth"]

        score, details = grade(agent_items, ground_truth)

        # Track best score (for partial progress reward shaping)
        improvement = max(0.0, score - self._state.best_f1)
        self._state.best_f1 = max(self._state.best_f1, score)

        # Reward signal: score + improvement bonus - step cost
        step_cost = 0.02  # Small penalty per step to encourage efficiency
        reward = round(score + (improvement * 0.3) - step_cost, 4)
        reward = max(0.0, min(1.5, reward))  # Clamp to reasonable range

        self._state.cumulative_reward += reward
        self._last_reward = reward

        # Determine done
        done = (
            action.is_final
            or self._state.step_count >= MAX_STEPS
        )
        self._state.is_complete = done
        self._last_done = done

        # Generate feedback
        feedback = generate_feedback(details, MAX_STEPS, self._state.step_count)

        obs = MeetingObservation(
            transcript=self._current_task["transcript"],
            task_description=self._current_task["description"],
            step_feedback=feedback,
            items_found_count=details["items_found"],
            total_items_in_task=len(ground_truth),
            current_f1=details["f1"],
            done=done,
            reward=reward,
            step_count=self._state.step_count,
            steps_remaining=max(0, MAX_STEPS - self._state.step_count),
        )
        self._last_obs = obs
        return obs

    # ── state ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> MeetingState:
        return self._state

    # ── reward ────────────────────────────────────────────────────────────────

    def reward(self, action: MeetingAction, obs: MeetingObservation) -> float:
        """Return the reward from the last step (computed inside step())."""
        return self._last_reward

    # ── done ──────────────────────────────────────────────────────────────────

    def is_done(self) -> bool:
        return self._last_done
