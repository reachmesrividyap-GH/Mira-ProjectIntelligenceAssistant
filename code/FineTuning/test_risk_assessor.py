"""
DeepEval Test Suite for Mira's Risk Assessor Agent (N16)
=========================================================
Run with: python test_risk_assessor.py

Prerequisites:
  pip install deepeval openai
  Set OPENAI_API_KEY as environment variable (do NOT hardcode)
    Windows PowerShell:  $env:OPENAI_API_KEY = "sk-proj-..."
    Mac/Linux:           export OPENAI_API_KEY="sk-proj-..."

Usage:
  1. Run each test input below through N16 in n8n (chat or webhook).
  2. Paste N16's raw JSON output into PIPELINE_OUTPUTS, matching the
     test ID.
  3. Run this script. Save the console output for your before/after
     comparison.
  4. After fine-tuning N16, re-run the same test inputs and repeat
     steps 2-3 with the new outputs.

NOTE: N16's system prompt was refreshed to make two rules explicit
that the metrics below already assumed: (1) T6_named_person_new-style
requests ("X is leaving, which sprints does this affect?") must be
classified as intent "both" (report + question), not "question" only,
so a new_risks entry is still required; and (2) T9_sprint_scoped-style
requests ("What risks affect Sprint 5?") are explicitly sufficient
on their own (sufficient_detail: true) and must be grounded in that
sprint's real tracked tasks rather than a generic risk substitution.
No metric text below needed to change — they were already written to
expect this behavior — but it's worth knowing the underlying prompt
now states it explicitly rather than implying it.
"""
import sys, os, json
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()  # reads OPENAI_API_KEY from a .env file in this same folder, if present

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

if not os.environ.get('OPENAI_API_KEY'):
    print("ERROR: OPENAI_API_KEY not found. Either create a .env file in this")
    print("folder containing OPENAI_API_KEY=sk-proj-..., or set it directly:")
    print('  PowerShell:  $env:OPENAI_API_KEY = "sk-proj-..."')
    sys.exit(1)

# ============================================================
# TEST INPUTS — mirrors this build's own confirmed bug/fix cases
# ============================================================
TEST_INPUTS = {
    "T1_vague_statement":       {"input": "Generate a risk assessment for: 'New project starting soon.'", "label": "vague"},
    "T2_vague_statement_alt":   {"input": "Risk check for something new we're starting", "label": "vague"},
    "T3_broad_named_project":   {"input": "Generate a risk assessment for the AI Adoption Project at ABCDE Ltd.", "label": "broad"},
    "T4_top_n":                 {"input": "What are the top 3 risks for the ABCDE Ltd project?", "label": "count"},
    "T5_top_n_alt":             {"input": "Give me the top 2 risks for this project", "label": "count"},
    "T6_named_person_new":      {"input": "Priya Nair is leaving the organization. Which sprints does this affect?", "label": "named_new"},
    "T7_named_person_existing": {"input": "Arjun Mehta might be stepping back from the project soon, what's the impact?", "label": "named_existing"},
    "T8_bare_vague":            {"input": "risks?", "label": "vague"},
    "T9_sprint_scoped":         {"input": "What risks affect Sprint 5?", "label": "broad"},
}

# ============================================================
# PASTE YOUR N16 (Risk Assessor Agent) OUTPUTS HERE
# Paste the raw JSON string N16 actually produced for each test ID.
#
# IMPORTANT: each value is a RAW triple-quoted string (r'''...''').
# That means you can paste N16's JSON output exactly as it comes out
# of n8n, double quotes and all, without escaping anything by hand.
# Leave a value as the placeholder string "PASTE_HERE" (no r''' needed)
# for any test you haven't run through N16 yet — the harness skips
# those automatically.
# ============================================================
PIPELINE_OUTPUTS = {
    "T1_vague_statement": r'''
{
  "intent": "question",
  "answer": "Here are the relevant risks for your query.",
  "requested_count": null,
  "risk_list_heading": "Here are the relevant risks for your query.",
  "sufficient_detail": false,
  "missing_info": [
    "Specific project name or details about the new project."
  ],
  "risks": [],
  "new_risks": []
}
''',

    "T2_vague_statement_alt": r'''
{
  "intent": "question",
  "answer": "Here are the relevant risks for your query.",
  "requested_count": null,
  "risk_list_heading": "Here are the relevant risks for your query.",
  "sufficient_detail": false,
  "missing_info": ["specific project name or details about the new initiative"],
  "risks": [],
  "new_risks": []
}
''',

    "T3_broad_named_project": r'''
{
  "intent": "question",
  "answer": "Here are the relevant risks for the AI Adoption Project at ABCDE Ltd (all already tracked in the register).",
  "requested_count": null,
  "risk_list_heading": "Here are the relevant risks for the AI Adoption Project at ABCDE Ltd (all already tracked in the register).",
  "sufficient_detail": true,
  "missing_info": [],
  "risks": [
    {
      "risk_id": "R01",
      "category": "Data Quality & Availability",
      "risk": "Incomplete or unstructured data",
      "impact": "Poor model performance and inaccurate outcomes",
      "mitigation": "Perform early data audits and implement robust data governance",
      "matched_entity": "AI Adoption Project at ABCDE Ltd"
    },
    {
      "risk_id": "R02",
      "category": "Integration Issues",
      "risk": "Difficulty integrating AI with legacy systems",
      "impact": "Delays in deployment and operational disruptions",
      "mitigation": "Plan integration roadmap and use APIs/middleware",
      "matched_entity": "AI Adoption Project at ABCDE Ltd"
    },
    {
      "risk_id": "R03",
      "category": "Talent & Skills Gap",
      "risk": "Lack of internal AI/ML expertise",
      "impact": "Overreliance on external vendors and slow progress",
      "mitigation": "Provide staff training and hire specialized talent",
      "matched_entity": "AI Adoption Project at ABCDE Ltd"
    },
    {
      "risk_id": "R04",
      "category": "Change Resistance",
      "risk": "Employees resist adopting AI-driven workflows",
      "impact": "Low adoption and reduced ROI",
      "mitigation": "Strong change management and early engagement",
      "matched_entity": "AI Adoption Project at ABCDE Ltd"
    },
    {
      "risk_id": "R05",
      "category": "Ethical & Legal Concerns",
      "risk": "Risks of biased models or data privacy violations",
      "impact": "Legal repercussions and reputational damage",
      "mitigation": "Involve legal early and implement fairness checks",
      "matched_entity": "AI Adoption Project at ABCDE Ltd"
    },
    {
      "risk_id": "R06",
      "category": "Cost Overruns",
      "risk": "Underestimation of AI development costs",
      "impact": "Budget constraints and project delays",
      "mitigation": "Set realistic budget and build in contingency funds",
      "matched_entity": "AI Adoption Project at ABCDE Ltd"
    },
    {
      "risk_id": "R08",
      "category": "Model Performance Drift",
      "risk": "AI models lose accuracy over time",
      "impact": "Incorrect decisions and trust erosion",
      "mitigation": "Regular retraining cycles and monitoring dashboards",
      "matched_entity": "AI Adoption Project at ABCDE Ltd"
    },
    {
      "risk_id": "R09",
      "category": "Security Vulnerabilities",
      "risk": "AI models targeted by adversarial attacks",
      "impact": "Data breaches and operational risks",
      "mitigation": "Adopt AI security best practices and penetration testing",
      "matched_entity": "AI Adoption Project at ABCDE Ltd"
    },
    {
      "risk_id": "R10",
      "category": "Vendor Lock-in",
      "risk": "Overdependence on a specific AI platform",
      "impact": "Flexibility loss and future switching costs",
      "mitigation": "Favor open standards and negotiate flexible vendor terms",
      "matched_entity": "AI Adoption Project at ABCDE Ltd"
    },
    {
      "risk_id": "R11",
      "category": "Talent & Skills Gap",
      "risk": "Departure of Arjun Mehta could lead to a critical gap in leadership for ongoing tasks and milestones.",
      "impact": "Potential delays in task execution and project timelines, especially for tasks involving high-level stakeholder engagement and strategic project closure.",
      "mitigation": "Identify and groom a successor from the team, redistribute tasks among existing team members to ensure continuity, and document critical processes and decisions.",
      "matched_entity": "Arjun Mehta"
    }
  ],
  "new_risks": []
}
''',

    "T4_top_n": r'''
{
  "intent": "question",
  "answer": "Here are the top 3 risks for the ABCDE Ltd project, ranked by severity.",
  "requested_count": 3,
  "risk_list_heading": "Here are the top 3 risks, ranked by severity.",
  "sufficient_detail": true,
  "missing_info": [],
  "risks": [
    {
      "risk_id": "R06",
      "category": "Cost Overruns",
      "risk": "Underestimation of AI development costs",
      "impact": "Budget constraints and project delays",
      "mitigation": "Set realistic budget and build in contingency funds",
      "matched_entity": "ABCDE Ltd project"
    },
    {
      "risk_id": "R10",
      "category": "Vendor Lock-in",
      "risk": "Overdependence on a specific AI platform",
      "impact": "Flexibility loss and future switching costs",
      "mitigation": "Favor open standards and negotiate flexible vendor terms",
      "matched_entity": "ABCDE Ltd project"
    },
    {
      "risk_id": "R02",
      "category": "Integration Issues",
      "risk": "Difficulty integrating AI with legacy systems",
      "impact": "Delays in deployment and operational disruptions",
      "mitigation": "Plan integration roadmap and use APIs/middleware",
      "matched_entity": "ABCDE Ltd project"
    }
  ],
  "new_risks": []
}
''',

    "T5_top_n_alt": r'''
{
  "intent": "question",
  "answer": "Here are the top 2 risks for the project, ranked by severity.",
  "requested_count": 2,
  "risk_list_heading": "Here are the top 2 risks, ranked by severity.",
  "sufficient_detail": true,
  "missing_info": [],
  "risks": [
    {
      "risk_id": "R06",
      "category": "Cost Overruns",
      "risk": "Underestimation of AI development costs",
      "impact": "Budget constraints and project delays",
      "mitigation": "Set realistic budget and build in contingency funds",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R07",
      "category": "Unrealistic Expectations",
      "risk": "Stakeholders expect immediate ROI",
      "impact": "Disillusionment and loss of support",
      "mitigation": "Align expectations and show quick wins",
      "matched_entity": "project-wide assessment"
    }
  ],
  "new_risks": []
}
''',

    "T6_named_person_new": r'''
{
  "intent": "both",
  "answer": "Priya Nair's departure will affect Sprint 2 and Sprint 4 due to her involvement in critical tasks and milestones.",
  "requested_count": null,
  "risk_list_heading": "Here are the risks affecting the sprints due to Priya Nair's departure.",
  "sufficient_detail": true,
  "missing_info": [],
  "risks": [],
  "new_risks": [
    {
      "risk_id": "",
      "category": "Talent & Skills Gap",
      "risk": "Priya Nair's departure could lead to a critical gap in leadership for ongoing tasks and milestones.",
      "impact": "Potential delays in task execution and project timelines, especially for tasks involving data analysis and pipeline development.",
      "mitigation": "Identify and groom a successor from the team, redistribute tasks among existing team members to ensure continuity, and document critical processes and decisions."
    }
  ]
}
''',

    "T7_named_person_existing": r'''
{
  "intent": "both",
  "answer": "Arjun Mehta's potential departure could significantly impact ongoing tasks and milestones, particularly in leadership roles. Here are the relevant risks and their implications.",
  "requested_count": null,
  "risk_list_heading": "Here are the risks associated with Arjun Mehta's potential departure.",
  "sufficient_detail": true,
  "missing_info": [],
  "risks": [
    {
      "risk_id": "R11",
      "category": "Talent & Skills Gap",
      "risk": "Departure of Arjun Mehta could lead to a critical gap in leadership for ongoing tasks and milestones.",
      "impact": "Potential delays in task execution and project timelines, especially for tasks involving high-level stakeholder engagement and strategic project closure.",
      "mitigation": "Identify and groom a successor from the team, redistribute tasks among existing team members to ensure continuity, and document critical processes and decisions.",
      "matched_entity": "Arjun Mehta"
    }
  ],
  "new_risks": []
}
''',

    "T8_bare_vague": r'''
{
  "intent": "question",
  "answer": "Here are the relevant risks for your query.",
  "requested_count": null,
  "risk_list_heading": "Here are the relevant risks for your query.",
  "sufficient_detail": true,
  "missing_info": [],
  "risks": [
    {
      "risk_id": "R01",
      "category": "Data Quality & Availability",
      "risk": "Incomplete or unstructured data",
      "impact": "Poor model performance and inaccurate outcomes",
      "mitigation": "Perform early data audits and implement robust data governance",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R02",
      "category": "Integration Issues",
      "risk": "Difficulty integrating AI with legacy systems",
      "impact": "Delays in deployment and operational disruptions",
      "mitigation": "Plan integration roadmap and use APIs/middleware",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R03",
      "category": "Talent & Skills Gap",
      "risk": "Lack of internal AI/ML expertise",
      "impact": "Overreliance on external vendors and slow progress",
      "mitigation": "Provide staff training and hire specialized talent",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R04",
      "category": "Change Resistance",
      "risk": "Employees resist adopting AI-driven workflows",
      "impact": "Low adoption and reduced ROI",
      "mitigation": "Strong change management and early engagement",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R05",
      "category": "Ethical & Legal Concerns",
      "risk": "Risks of biased models or data privacy violations",
      "impact": "Legal repercussions and reputational damage",
      "mitigation": "Involve legal early and implement fairness checks",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R06",
      "category": "Cost Overruns",
      "risk": "Underestimation of AI development costs",
      "impact": "Budget constraints and project delays",
      "mitigation": "Set realistic budget and build in contingency funds",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R07",
      "category": "Unrealistic Expectations",
      "risk": "Stakeholders expect immediate ROI",
      "impact": "Disillusionment and loss of support",
      "mitigation": "Align expectations and show quick wins",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R08",
      "category": "Model Performance Drift",
      "risk": "AI models lose accuracy over time",
      "impact": "Incorrect decisions and trust erosion",
      "mitigation": "Regular retraining cycles and monitoring dashboards",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R09",
      "category": "Security Vulnerabilities",
      "risk": "AI models targeted by adversarial attacks",
      "impact": "Data breaches and operational risks",
      "mitigation": "Adopt AI security best practices and penetration testing",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R10",
      "category": "Vendor Lock-in",
      "risk": "Overdependence on a specific AI platform",
      "impact": "Flexibility loss and future switching costs",
      "mitigation": "Favor open standards and negotiate flexible vendor terms",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R11",
      "category": "Talent & Skills Gap",
      "risk": "Departure of Arjun Mehta could lead to a critical gap in leadership for ongoing tasks and milestones.",
      "impact": "Potential delays in task execution and project timelines, especially for tasks involving high-level stakeholder engagement and strategic project closure.",
      "mitigation": "Identify and groom a successor from the team, redistribute tasks among existing team members to ensure continuity, and document critical processes and decisions.",
      "matched_entity": "Arjun Mehta"
    }
  ],
  "new_risks": []
}
''',

    "T9_sprint_scoped": r'''
{
  "intent": "question",
  "answer": "Here are the risks affecting Sprint 5.",
  "requested_count": null,
  "risk_list_heading": "Here are the risks affecting Sprint 5.",
  "sufficient_detail": true,
  "missing_info": [],
  "risks": [
    {
      "risk_id": "R11",
      "category": "Talent & Skills Gap",
      "risk": "Departure of Arjun Mehta could lead to a critical gap in leadership for ongoing tasks and milestones.",
      "impact": "Potential delays in task execution and project timelines, especially for tasks involving high-level stakeholder engagement and strategic project closure.",
      "mitigation": "Identify and groom a successor from the team, redistribute tasks among existing team members to ensure continuity, and document critical processes and decisions.",
      "matched_entity": "Arjun Mehta"
    },
    {
      "risk_id": "R04",
      "category": "Change Resistance",
      "risk": "Employees resist adopting AI-driven workflows.",
      "impact": "Low adoption and reduced ROI.",
      "mitigation": "Strong change management and early engagement.",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R02",
      "category": "Integration Issues",
      "risk": "Difficulty integrating AI with legacy systems.",
      "impact": "Delays in deployment and operational disruptions.",
      "mitigation": "Plan integration roadmap and use APIs/middleware.",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R01",
      "category": "Data Quality & Availability",
      "risk": "Incomplete or unstructured data.",
      "impact": "Poor model performance and inaccurate outcomes.",
      "mitigation": "Perform early data audits and implement robust data governance.",
      "matched_entity": "project-wide assessment"
    },
    {
      "risk_id": "R10",
      "category": "Vendor Lock-in",
      "risk": "Overdependence on a specific AI platform.",
      "impact": "Flexibility loss and future switching costs.",
      "mitigation": "Favor open standards and negotiate flexible vendor terms.",
      "matched_entity": "project-wide assessment"
    }
  ],
  "new_risks": []
}
''',
}

# ============================================================
# VALIDATE PASTED OUTPUTS — catches a bad paste (invalid JSON or a
# missing required key) with a clean message, instead of a raw
# Python SyntaxError pointing at some unrelated line number.
# ============================================================
def validate_pipeline_outputs():
    print("\nValidating pasted N16 outputs...")
    required_keys = ["intent", "answer", "sufficient_detail", "risks", "new_risks"]
    any_invalid = False

    for test_id, output in PIPELINE_OUTPUTS.items():
        if "PASTE" in output:
            print(f"  {test_id}: (not pasted yet, skipping)")
            continue

        try:
            parsed = json.loads(output)
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                raise ValueError(f"Missing required key(s): {', '.join(missing)}")
            print(f"  {test_id}: OK")
        except Exception as e:
            any_invalid = True
            print(f"  {test_id}: INVALID - {e}")

    if any_invalid:
        print("\nFix the invalid entries above in PIPELINE_OUTPUTS before running the evaluation.")
        sys.exit(1)


# ============================================================
# METRICS — each one restates a rule from N16's own system message
# ============================================================

vague_handling = GEval(
    name="Vague Statement Handling",
    criteria=(
        "The input is a vague statement with NO identifiable project name or "
        "scope attached (e.g., 'new project starting soon', 'something new "
        "we're starting'). A CORRECT response should: "
        "1) Set sufficient_detail to false. "
        "2) NOT return any populated risks or new_risks arrays. "
        "3) NOT invent a project-specific risk from a generic statement. "
        "4) Ask the user for the specific project name or scope in 'answer'. "
        "A broad question asking to SEE existing tracked risks (e.g., 'what "
        "are the risks for the project') is NOT the same as this vague-"
        "statement case and should NOT be flagged as insufficient — only "
        "score this metric on inputs that are genuinely vague statements "
        "with no identifiable project attached. "
        "Correctly flagging as insufficient with empty arrays = PASS. "
        "Fabricating risks from a vague statement = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4.1",
)

groundedness = GEval(
    name="Risk Groundedness",
    criteria=(
        "The input asks about existing, tracked project risks (broadly, by "
        "count, by sprint, or about a named person/system already reflected "
        "in the risk register). Does the output's 'risks' array contain ONLY "
        "risks that are plausible, real entries (each with a risk_id, "
        "category, risk description, impact, and mitigation) rather than "
        "generic, invented, or unrelated content? Are named entities "
        "(people, systems) only matched to a risk if that risk's own text "
        "plausibly names or concerns them? "
        "Grounded, specific risk entries = PASS. Generic or invented risk "
        "content = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4.1",
)

count_enforcement = GEval(
    name="Requested Count Enforcement",
    criteria=(
        "If the input asks for a specific number of risks (e.g., 'top 3', "
        "'top 2'), does the output's 'risks' array contain EXACTLY that "
        "many entries — no more, no fewer? Does 'requested_count' correctly "
        "reflect the number asked for? If no specific count was requested, "
        "this metric should PASS regardless of array length. "
        "Exact count match = PASS. Wrong count, or ignoring the request = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4.1",
)

new_vs_existing = GEval(
    name="New vs Existing Risk Routing",
    criteria=(
        "If the input reports a real, specific event or concern about a "
        "named person or system, did the output correctly decide whether "
        "this is already represented in the risk register (route to "
        "'risks', with a real risk_id and a 'matched_entity' naming the "
        "real overlap) versus genuinely new (route to 'new_risks', with an "
        "EMPTY risk_id, never borrowing an existing risk_id for freshly-"
        "generated content)? "
        "Correct routing with correct risk_id handling = PASS. Borrowing an "
        "existing risk_id for new content, or duplicating an already-"
        "tracked risk into new_risks = FAIL."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5, model="gpt-4.1",
)

# AnswerRelevancyMetric breaks a long, JSON-heavy actual_output into many
# individual statements to verdict on — with gpt-4o's 16384-token output
# cap that generation step can run out of room mid-response and raise
# openai.LengthFinishReasonError. gpt-4.1 has a much higher output cap
# (32,768 tokens), which comfortably covers these verbose N16 outputs.
relevancy = AnswerRelevancyMetric(threshold=0.5, model="gpt-4.1")


def run_evaluation(include_relevancy=False):
    """
    Each block below scores a test group with ONLY the metric(s) that
    directly test the behavior being fine-tuned:
        Vague handling   -> sufficient_detail / empty arrays
        Top-N routing    -> exact count / requested_count
        New vs Existing  -> risk_id / matched_entity routing
        Groundedness     -> are the risks real and relevant
    AnswerRelevancyMetric is a general linguistic-relevance score, not
    specific to any of N16's actual rules, so it's left OUT of these
    targeted runs by default and only added when include_relevancy=True
    (or call run_relevancy_only() separately). This also keeps each run
    smaller/faster and less likely to bump into token-limit issues.

    If a metric's verdicts look wrong and you want to see exactly what
    the judge model produced, set verbose_mode=True on that metric
    (e.g. GEval(..., verbose_mode=True)) — DeepEval prints the judge's
    raw reasoning/verdicts for each test case when this is on.
    """
    def get_cases(label):
        return [
            LLMTestCase(input=d["input"], actual_output=PIPELINE_OUTPUTS[t])
            for t, d in TEST_INPUTS.items()
            if d["label"] == label and "PASTE" not in PIPELINE_OUTPUTS.get(t, "PASTE")
        ]

    extra = [relevancy] if include_relevancy else []

    print("\n" + "=" * 60)
    print("EVALUATING VAGUE STATEMENTS (must flag as insufficient)")
    print("=" * 60)
    vague_cases = get_cases("vague")
    if vague_cases:
        evaluate(test_cases=vague_cases, metrics=[vague_handling] + extra)
    else:
        print("  (no outputs pasted yet)")

    print("\n" + "=" * 60)
    print("EVALUATING BROAD / NAMED-PROJECT REQUESTS")
    print("=" * 60)
    broad_cases = get_cases("broad")
    if broad_cases:
        evaluate(test_cases=broad_cases, metrics=[groundedness] + extra)
    else:
        print("  (no outputs pasted yet)")

    print("\n" + "=" * 60)
    print("EVALUATING COUNT-SPECIFIC REQUESTS (top-N)")
    print("=" * 60)
    count_cases = get_cases("count")
    if count_cases:
        evaluate(test_cases=count_cases, metrics=[count_enforcement, groundedness] + extra)
    else:
        print("  (no outputs pasted yet)")

    print("\n" + "=" * 60)
    print("EVALUATING NEW RISK (named person, not yet tracked)")
    print("=" * 60)
    new_cases = get_cases("named_new")
    if new_cases:
        evaluate(test_cases=new_cases, metrics=[new_vs_existing, groundedness] + extra)
    else:
        print("  (no outputs pasted yet)")

    print("\n" + "=" * 60)
    print("EVALUATING EXISTING RISK MATCH (named person, already tracked)")
    print("=" * 60)
    existing_cases = get_cases("named_existing")
    if existing_cases:
        evaluate(test_cases=existing_cases, metrics=[new_vs_existing, groundedness] + extra)
    else:
        print("  (no outputs pasted yet)")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("\nWhat to look for:")
    print("  Vague statements:  Vague Statement Handling should PASS")
    print("  Broad/named:       Risk Groundedness should PASS")
    print("  Top-N:             Requested Count Enforcement should PASS")
    print("  Named person:      New vs Existing Risk Routing should PASS")
    if include_relevancy:
        print("  All:               Relevancy should PASS")
    print("\nSave a screenshot for before/after comparison.")


def run_relevancy_only():
    """Optional separate pass: general Answer Relevancy across every pasted output."""
    all_cases = [
        LLMTestCase(input=d["input"], actual_output=PIPELINE_OUTPUTS[t])
        for t, d in TEST_INPUTS.items()
        if "PASTE" not in PIPELINE_OUTPUTS.get(t, "PASTE")
    ]
    print("\n" + "=" * 60)
    print("EVALUATING ANSWER RELEVANCY (all test cases)")
    print("=" * 60)
    if all_cases:
        evaluate(test_cases=all_cases, metrics=[relevancy])
    else:
        print("  (no outputs pasted yet)")


if __name__ == "__main__":
    print("=" * 60)
    print("MIRA RISK ASSESSOR AGENT (N16) - DeepEval Evaluation")
    print("=" * 60)
    validate_pipeline_outputs()
    run_evaluation()               # targeted metrics only (fast, focused)
    # run_relevancy_only()         # uncomment to also run Answer Relevancy separately
