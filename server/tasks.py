"""
meeting_env/server/tasks.py
Three task definitions with meeting transcripts and ground truth action items.
Difficulty: easy → medium → hard
"""
from typing import Dict, Any, List

# ── Ground truth schema ───────────────────────────────────────────────────────
# Each ground truth item has:
#   id:        unique identifier (used by grader)
#   owner:     canonical owner name/role (normalized lowercase for comparison)
#   task:      keywords that must appear in agent's task description
#   deadline:  canonical deadline string or None
#   priority:  "high" | "medium" | "low"


TASKS: Dict[str, Dict[str, Any]] = {

    # ── EASY: Explicit action items ──────────────────────────────────────────
    "easy": {
        "id": "easy",
        "name": "Explicit actions — product standup",
        "description": (
            "Extract all action items from the meeting transcript below. "
            "Action items are explicit commitments made by named individuals. "
            "For each item, identify: the owner (who is responsible), the task "
            "(what they will do), the deadline (when, if mentioned), and the "
            "priority (high/medium/low based on urgency cues in the transcript)."
        ),
        "transcript": """
[Product Team Weekly Standup — Tuesday 10:00 AM]

Priya (PM): Okay, let's move through blockers quickly. Maya, you had the API issue?

Maya (Eng): Yeah, the auth token expiry is causing 401s on mobile. I'll push the fix 
to staging by Thursday and ping QA once it's up.

Priya: Great. That's blocking the release so flag it high priority. Daniel, the 
onboarding flow designs?

Daniel (Design): Almost done. I'll send the final Figma mockups to the team by 
end of day today.

Priya: Perfect. Raj, we need the Q4 roadmap updated before the board meeting.

Raj (PM): I'll have the roadmap doc updated and shared in Notion by Friday EOD.

Priya: Also, Sam — can you follow up with the analytics vendor about the tracking 
bug? Just needs an email this week, medium priority is fine.

Sam (Eng): Sure, I'll send the follow-up email to the vendor by Wednesday.

Priya: And I'll write up the post-mortem on last week's outage and share it in 
Slack by Thursday. That's everything — let's ship.
""".strip(),
        "ground_truth": [
            {
                "id": "gt_1",
                "owner": "maya",
                "keywords": ["auth", "fix", "staging", "401", "token"],
                "deadline": "thursday",
                "priority": "high"
            },
            {
                "id": "gt_2",
                "owner": "daniel",
                "keywords": ["figma", "mockup", "design", "onboarding"],
                "deadline": "today",
                "priority": "medium"
            },
            {
                "id": "gt_3",
                "owner": "raj",
                "keywords": ["roadmap", "q4", "notion", "board"],
                "deadline": "friday",
                "priority": "medium"
            },
            {
                "id": "gt_4",
                "owner": "sam",
                "keywords": ["vendor", "analytics", "email", "tracking"],
                "deadline": "wednesday",
                "priority": "medium"
            },
            {
                "id": "gt_5",
                "owner": "priya",
                "keywords": ["post-mortem", "postmortem", "outage", "slack"],
                "deadline": "thursday",
                "priority": "medium"
            },
        ]
    },

    # ── MEDIUM: Implicit action items ────────────────────────────────────────
    "medium": {
        "id": "medium",
        "name": "Implicit actions — strategy session",
        "description": (
            "Extract all action items from the meeting transcript below. "
            "Many action items are implied through discussion rather than explicitly "
            "stated as 'X will do Y.' Look for decisions that require follow-up, "
            "problems that need to be investigated, and agreements that need to be "
            "formalized. Assign the most likely owner based on role and context. "
            "Deadlines may be implicit (e.g. 'before the launch' means the launch date). "
            "There are 6 action items in total."
        ),
        "transcript": """
[Growth Strategy Session — Thursday 2:00 PM]

Alex (Head of Growth): The retention numbers dropped 8% last month. We need to 
understand why before we can fix anything.

Jordan (Data): Yeah, I've been seeing some weird patterns in the cohort data. 
The drop is concentrated in week-2 users. I can dig into that more.

Alex: Good. Get me something concrete — we should have a hypothesis by next Monday 
before the exec review.

Jordan: I'll have it ready.

Alex: Also, the onboarding email sequence — has anyone looked at open rates recently?

Taylor (Marketing): Not since Q2 honestly. They're probably stale. I should refresh 
the copy.

Alex: It would be good to A/B test as well. Can we get that set up?

Taylor: I'll need the email tool access sorted first — I've been waiting on IT 
for two weeks.

Alex: That's blocking real work. Someone needs to escalate that ticket. 
[to Jordan] Can you loop in IT directly since you have that relationship?

Jordan: Yeah, I'll ping them today.

Alex: [to Taylor] And once access is sorted, the A/B test should be live before 
the end of the month, yeah?

Taylor: Should be doable.

Alex: One more thing — our referral program hasn't been touched in six months. 
The incentive structure is probably outdated given what competitors are doing. 
We should do a competitive analysis.

Sam (Strategy): I can own that. A few days of research should be enough.

Alex: Great. Wrap it up by end of next week so we can act on it in the sprint 
after this one.
""".strip(),
        "ground_truth": [
            {
                "id": "gt_1",
                "owner": "jordan",
                "keywords": ["retention", "cohort", "analysis", "hypothesis", "drop", "week-2"],
                "deadline": "monday",
                "priority": "high"
            },
            {
                "id": "gt_2",
                "owner": "taylor",
                "keywords": ["email", "copy", "onboarding", "refresh", "sequence"],
                "deadline": None,
                "priority": "medium"
            },
            {
                "id": "gt_3",
                "owner": "jordan",
                "keywords": ["it", "email", "access", "escalate", "ticket"],
                "deadline": "today",
                "priority": "high"
            },
            {
                "id": "gt_4",
                "owner": "taylor",
                "keywords": ["a/b", "ab test", "email", "test", "live"],
                "deadline": "end of month",
                "priority": "medium"
            },
            {
                "id": "gt_5",
                "owner": "sam",
                "keywords": ["referral", "competitive", "analysis", "incentive"],
                "deadline": "end of next week",
                "priority": "medium"
            },
            {
                "id": "gt_6",
                "owner": "alex",
                "keywords": ["referral", "sprint", "act", "program"],
                "deadline": "next sprint",
                "priority": "low"
            },
        ]
    },

    # ── HARD: Conflicting assignments ────────────────────────────────────────
    "hard": {
        "id": "hard",
        "name": "Conflicting assignments — engineering planning",
        "description": (
            "Extract all action items from the meeting transcript below. "
            "This is a complex meeting where some action items are implicitly assigned, "
            "some ownership changes during the conversation, and two participants appear "
            "to accept the same task at different points. "
            "You must resolve all conflicts and assign each action item to exactly one "
            "owner based on the final state of the conversation. "
            "Pay close attention to when someone defers, re-assigns, or backs out of a task. "
            "There are 5 distinct action items in total."
        ),
        "transcript": """
[Engineering Planning — Q4 Kickoff — Friday 3:00 PM]

Chris (Eng Manager): We need to sort out three things today: the database migration, 
the new CI pipeline, and the security audit prep.

Wei (Senior Eng): I can take the DB migration. I've done them before. 
I'll have a migration plan drafted by end of next week.

Chris: Good. Nadia, the CI pipeline?

Nadia (Eng): Yeah, I'll set up the new pipeline. I'm thinking Wednesday for a 
first working version.

Chris: Great. Security audit — we need someone to compile the evidence pack 
for the auditors. That means access logs, incident history, config exports.

Wei: I can do that too. I have access to all those systems.

Nadia: Actually, I was going to pick that up — I did it last year and know 
exactly what they need.

Chris: [pause] Nadia, you've got the CI pipeline. That's already a full week's work. 
Wei, can you take security audit prep instead?

Wei: Sure, no problem. I'll reprioritize. I can have the evidence pack ready 
by the Wednesday before the audit, which is... the 18th.

Chris: Perfect. One more thing — the DB migration needs a rollback plan 
documenting what we'd do if something goes wrong. Wei, while you're doing 
the migration plan, include a rollback section.

Wei: Will do — same deadline, end of next week.

Chris: And someone needs to schedule a dry-run of the migration with the 
whole team before we go live. That's on me — I'll set up a calendar invite 
by Monday morning.
""".strip(),
        "ground_truth": [
            {
                "id": "gt_1",
                "owner": "wei",
                "keywords": ["migration", "plan", "database", "db"],
                "deadline": "end of next week",
                "priority": "high"
            },
            {
                "id": "gt_2",
                "owner": "nadia",
                "keywords": ["ci", "pipeline", "build"],
                "deadline": "wednesday",
                "priority": "high"
            },
            {
                "id": "gt_3",
                "owner": "wei",
                "keywords": ["security", "audit", "evidence", "pack", "logs", "config"],
                "deadline": "18th",
                "priority": "high"
            },
            {
                "id": "gt_4",
                "owner": "wei",
                "keywords": ["rollback", "plan", "migration"],
                "deadline": "end of next week",
                "priority": "medium"
            },
            {
                "id": "gt_5",
                "owner": "chris",
                "keywords": ["dry-run", "dry run", "schedule", "calendar", "invite"],
                "deadline": "monday",
                "priority": "medium"
            },
        ]
    }
}
