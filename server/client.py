"""
meeting_env/client.py
Typed EnvClient for the Meeting Notes environment.
"""
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from .models import MeetingAction, MeetingObservation, MeetingState


class MeetingEnv(EnvClient[MeetingAction, MeetingObservation, MeetingState]):
    """
    Client for the Meeting Notes Action Item Extraction environment.

    Sync usage (for inference.py):
        with MeetingEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task_id="easy")
            print(result.observation.transcript)
            action = MeetingAction(action_items=[...], is_final=True)
            result = env.step(action)
            print(result.reward)

    Async usage:
        async with MeetingEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task_id="hard")
            result = await env.step(action)
    """

    def _step_payload(self, action: MeetingAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[MeetingObservation]:
        obs_data = payload.get("observation", {})
        obs = MeetingObservation(
            transcript=obs_data.get("transcript", ""),
            task_description=obs_data.get("task_description", ""),
            step_feedback=obs_data.get("step_feedback"),
            items_found_count=obs_data.get("items_found_count", 0),
            total_items_in_task=obs_data.get("total_items_in_task", 0),
            current_f1=obs_data.get("current_f1", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            step_count=obs_data.get("step_count", 0),
            steps_remaining=obs_data.get("steps_remaining", 5),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> MeetingState:
        return MeetingState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            task_name=payload.get("task_name", ""),
            best_f1=payload.get("best_f1", 0.0),
            attempts=payload.get("attempts", 0),
            is_complete=payload.get("is_complete", False),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
