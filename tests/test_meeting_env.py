"""
tests/test_meeting_env.py
Unit tests for grader logic and environment state management.
Run: pytest tests/ -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.grader import grade, _owner_matches, _keyword_score
from server.tasks import TASKS


# ── Grader unit tests ──────────────────────────────────────────────────────

class TestOwnerMatching:
    def test_exact_match(self):
        assert _owner_matches("Maya", "maya") is True

    def test_first_name_match(self):
        assert _owner_matches("Maya (Eng)", "maya") is True

    def test_wrong_owner(self):
        assert _owner_matches("Jordan", "maya") is False

    def test_partial_match_role(self):
        assert _owner_matches("Chris", "chris") is True


class TestKeywordScore:
    def test_all_keywords_present(self):
        score = _keyword_score("push the auth token fix to staging", ["auth", "fix", "staging"])
        assert score == 1.0

    def test_partial_keywords(self):
        score = _keyword_score("fix the auth issue", ["auth", "fix", "staging"])
        assert score == pytest.approx(2/3, rel=0.01)

    def test_no_keywords(self):
        score = _keyword_score("something unrelated", ["auth", "fix", "staging"])
        assert score == 0.0

    def test_empty_keywords(self):
        score = _keyword_score("anything", [])
        assert score == 1.0


class TestGrader:
    def test_perfect_submission(self):
        ground_truth = TASKS["easy"]["ground_truth"]
        agent_items = [
            {"owner": "Maya", "task": "push auth fix to staging", "deadline": "Thursday", "priority": "high"},
            {"owner": "Daniel", "task": "send figma mockups onboarding designs", "deadline": "today", "priority": "medium"},
            {"owner": "Raj", "task": "update q4 roadmap board in notion", "deadline": "Friday", "priority": "medium"},
            {"owner": "Sam", "task": "follow up with analytics vendor email tracking", "deadline": "Wednesday", "priority": "medium"},
            {"owner": "Priya", "task": "write post-mortem outage report for slack", "deadline": "Thursday", "priority": "medium"},
        ]
        score, details = grade(agent_items, ground_truth)
        assert score >= 0.80, f"Perfect submission should score >= 0.80, got {score}"
        assert details["items_found"] == 5

    def test_empty_submission(self):
        ground_truth = TASKS["easy"]["ground_truth"]
        score, details = grade([], ground_truth)
        assert score == 0.0
        assert details["precision"] == 0.0
        assert details["recall"] == 0.0
        assert details["f1"] == 0.0

    def test_wrong_owner_scores_zero(self):
        ground_truth = [
            {"id": "gt_1", "owner": "maya", "keywords": ["auth", "fix"], "deadline": "thursday", "priority": "high"}
        ]
        agent_items = [
            {"owner": "Daniel", "task": "push auth fix to staging", "deadline": "Thursday", "priority": "high"}
        ]
        score, details = grade(agent_items, ground_truth)
        assert score == 0.0

    def test_partial_credit_missing_deadline(self):
        ground_truth = [
            {"id": "gt_1", "owner": "maya", "keywords": ["auth", "fix", "staging"], "deadline": "thursday", "priority": "high"}
        ]
        agent_items = [
            {"owner": "Maya", "task": "push auth fix to staging", "deadline": None, "priority": "high"}
        ]
        score, details = grade(agent_items, ground_truth)
        # Should get credit for owner + keywords + priority, but not deadline
        assert 0.3 < score < 0.95

    def test_score_range(self):
        """Score must always be in [0.0, 1.0]."""
        ground_truth = TASKS["hard"]["ground_truth"]
        agent_items = [
            {"owner": "wei", "task": "database migration plan", "deadline": "end of next week", "priority": "high"},
        ]
        score, details = grade(agent_items, ground_truth)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        """Same inputs must always produce same output."""
        ground_truth = TASKS["medium"]["ground_truth"]
        agent_items = [
            {"owner": "Jordan", "task": "analyze cohort retention data hypothesis", "deadline": "Monday", "priority": "high"},
            {"owner": "Taylor", "task": "refresh onboarding email copy sequence", "deadline": None, "priority": "medium"},
        ]
        score1, _ = grade(agent_items, ground_truth)
        score2, _ = grade(agent_items, ground_truth)
        assert score1 == score2


# ── Task structure tests ───────────────────────────────────────────────────

class TestTasks:
    def test_all_tasks_present(self):
        assert "easy" in TASKS
        assert "medium" in TASKS
        assert "hard" in TASKS

    def test_ground_truth_has_required_fields(self):
        for task_id, task in TASKS.items():
            for item in task["ground_truth"]:
                assert "id" in item, f"{task_id}: missing id"
                assert "owner" in item, f"{task_id}: missing owner"
                assert "keywords" in item, f"{task_id}: missing keywords"
                assert "priority" in item, f"{task_id}: missing priority"
                assert item["priority"] in ("high", "medium", "low"), \
                    f"{task_id}: invalid priority {item['priority']}"

    def test_transcripts_are_non_empty(self):
        for task_id, task in TASKS.items():
            assert len(task["transcript"]) > 100, f"{task_id}: transcript too short"

    def test_difficulty_progression(self):
        """Hard task should have the most complex ground truth."""
        easy_n = len(TASKS["easy"]["ground_truth"])
        hard_n = len(TASKS["hard"]["ground_truth"])
        # Both have items, hard ones require conflict resolution
        assert easy_n >= 3
        assert hard_n >= 3


# ── Environment integration tests ────────────────────────────────────────

class TestEnvironmentLogic:
    def setup_method(self):
        from server.meeting_environment import MeetingEnvironment
        self.env = MeetingEnvironment()

    def test_reset_returns_observation(self):
        from server.models import MeetingObservation
        obs = self.env.reset("easy")
        assert isinstance(obs, MeetingObservation)
        assert len(obs.transcript) > 0
        assert obs.total_items_in_task == 5
        assert obs.current_f1 == 0.0
        assert obs.done is False

    def test_reset_clears_state(self):
        self.env.reset("easy")
        from server.models import MeetingAction, ExtractedActionItem
        action = MeetingAction(action_items=[], is_final=False)
        self.env.step(action)
        assert self.env.state.step_count == 1

        self.env.reset("medium")
        assert self.env.state.step_count == 0
        assert self.env.state.task_id == "medium"

    def test_step_increments_counter(self):
        self.env.reset("easy")
        from server.models import MeetingAction
        action = MeetingAction(action_items=[], is_final=False)
        self.env.step(action)
        assert self.env.state.step_count == 1
        self.env.step(action)
        assert self.env.state.step_count == 2

    def test_is_final_ends_episode(self):
        self.env.reset("easy")
        from server.models import MeetingAction
        action = MeetingAction(action_items=[], is_final=True)
        obs = self.env.step(action)
        assert obs.done is True

    def test_reward_is_non_negative(self):
        self.env.reset("easy")
        from server.models import MeetingAction
        action = MeetingAction(action_items=[], is_final=True)
        obs = self.env.step(action)
        assert obs.reward is not None
        assert obs.reward >= 0.0

    def test_good_submission_scores_high(self):
        self.env.reset("easy")
        from server.models import MeetingAction, ExtractedActionItem
        action = MeetingAction(
            action_items=[
                ExtractedActionItem(owner="Maya", task="push auth fix to staging", deadline="Thursday", priority="high"),
                ExtractedActionItem(owner="Daniel", task="send figma mockups onboarding", deadline="today", priority="medium"),
                ExtractedActionItem(owner="Raj", task="update q4 roadmap board notion", deadline="Friday", priority="medium"),
                ExtractedActionItem(owner="Sam", task="email vendor about analytics tracking bug", deadline="Wednesday", priority="medium"),
                ExtractedActionItem(owner="Priya", task="write post-mortem about outage slack", deadline="Thursday", priority="medium"),
            ],
            is_final=True
        )
        obs = self.env.step(action)
        assert obs.current_f1 >= 0.70, f"Good submission should score >= 0.70 F1, got {obs.current_f1}"
