"""
Fine-Tuning: Mira Risk Assessor Agent (N16)
=============================================
Run with: python fine_tune_risk_assessor.py

Prerequisites:
  pip install openai
  Set OPENAI_API_KEY as environment variable (do NOT hardcode)
    Windows PowerShell:  $env:OPENAI_API_KEY = "sk-proj-..."
    Mac/Linux:           export OPENAI_API_KEY="sk-proj-..."

This trains ONE model: a fine-tuned version of N16's base model, using
training_data_risk_assessor.jsonl — real corrected examples built from
this build's own confirmed bug/fix pairs (vague-statement handling,
top-N count enforcement, new-vs-existing risk routing).

NOTE: The system message embedded in every example of the JSONL was
refreshed to match N16's current production prompt, which added two
clarifications since the examples were first built: (1) a report of
someone leaving/stepping back phrased as a question — e.g. "X is
leaving, which sprints does this affect?" — must still be classified
as intent "both", not "question", so it still gets a new_risks entry;
and (2) sprint/component-scoped requests (e.g. "What risks affect
Sprint 5?") are explicitly sufficient_detail: true and must be
grounded in that sprint's real tracked tasks, not a generic risk
substitution. The 10 example answers already matched both rules, so
only the system message text itself was swapped in — the schema below
(the required keys this script checks for) did not change.
"""
import sys, os, json, time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()  # reads OPENAI_API_KEY from a .env file in this same folder, if present

from openai import OpenAI

if not os.environ.get('OPENAI_API_KEY'):
    print("ERROR: OPENAI_API_KEY not found. Either create a .env file in this")
    print("folder containing OPENAI_API_KEY=sk-proj-..., or set it directly:")
    print('  PowerShell:  $env:OPENAI_API_KEY = "sk-proj-..."')
    sys.exit(1)

client = OpenAI()

# gpt-4o-mini is confirmed fine-tunable by OpenAI's own docs. Note this
# is a DIFFERENT model family than N16's current live model (gpt-4.1-mini)
# — after fine-tuning succeeds, N16's model will need to be switched to
# this fine-tuned gpt-4o-mini, not just have fine-tuning "added on top"
# of its current gpt-4.1-mini. Re-run the full baseline test suite after
# switching, since this is a genuine model change, not just a tune.
BASE_MODEL = "gpt-4o-mini-2024-07-18"
N_EPOCHS = 3
TRAINING_FILE = "training_data_risk_assessor.jsonl"
SUFFIX = "mira-risk-assessor"


def validate_jsonl(filepath):
    print(f"\n  Validating {filepath}...")
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)
            roles = [m["role"] for m in data["messages"]]
            assert roles[0] == "system", f"Line {i}: first message must be system"
            assert "user" in roles, f"Line {i}: needs a user message"
            assert "assistant" in roles, f"Line {i}: needs an assistant message"
            # Confirm the assistant message is valid JSON matching N16's schema
            assistant_msg = next(m["content"] for m in data["messages"] if m["role"] == "assistant")
            parsed = json.loads(assistant_msg)
            for required_key in ["intent", "answer", "sufficient_detail", "risks", "new_risks"]:
                assert required_key in parsed, f"Line {i}: assistant JSON missing '{required_key}'"
            count += 1
    print(f"  PASSED - {count} examples, all schema-valid")
    return count


def upload_and_train(filepath, suffix):
    print(f"\n  Uploading {filepath}...")
    with open(filepath, "rb") as f:
        file_obj = client.files.create(file=f, purpose="fine-tune")
    print(f"  File ID: {file_obj.id}")
    print(f"  Creating job (suffix: {suffix})...")
    job = client.fine_tuning.jobs.create(
        training_file=file_obj.id,
        model=BASE_MODEL,
        hyperparameters={"n_epochs": N_EPOCHS},
        suffix=suffix,
    )
    print(f"  Job ID: {job.id}")
    return job.id


def wait_for_job(job_id, name):
    print(f"\n  Waiting for {name}...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        if job.status == "succeeded":
            print(f"  {name} COMPLETE: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status in ("failed", "cancelled"):
            print(f"  {name} {job.status.upper()}: {getattr(job, 'error', 'unknown')}")
            return None
        else:
            trained = ""
            if hasattr(job, "trained_tokens") and job.trained_tokens:
                trained = f" | Tokens: {job.trained_tokens}"
            print(f"  [{name}] {job.status}{trained}")
            time.sleep(30)


def test_model(model_id, name, system_prompt, test_input):
    print(f"\n  Testing {name}...")
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_input},
        ],
    )
    output = response.choices[0].message.content
    print(f"  Input: '{test_input}'")
    print(f"  Output:")
    print(f"  {'-' * 50}")
    for line in output.split('\n'):
        print(f"  {line}")
    print(f"  {'-' * 50}")


def main():
    print("=" * 60)
    print("FINE-TUNING: Mira Risk Assessor Agent (N16)")
    print("=" * 60)

    validate_jsonl(TRAINING_FILE)
    job_id = upload_and_train(TRAINING_FILE, SUFFIX)

    print("\n" + "=" * 60)
    print("WAITING FOR FINE-TUNING JOB")
    print("=" * 60)

    fine_tuned_model = wait_for_job(job_id, "Risk Assessor")

    if fine_tuned_model:
        # Read the system message from the JSONL's own first example, so
        # this test uses the exact same prompt N16 runs in production.
        with open(TRAINING_FILE, "r", encoding="utf-8") as f:
            first_example = json.loads(f.readline())
            system_prompt = next(
                m["content"] for m in first_example["messages"] if m["role"] == "system"
            )

        # Re-run the exact case that originally exposed the bug.
        test_model(
            fine_tuned_model,
            "Risk Assessor",
            system_prompt,
            "Generate a risk assessment for: 'New project starting soon.'",
        )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if fine_tuned_model:
        print(f"\n  Fine-tuned model: {fine_tuned_model}")
        print(f"  -> Paste into N16's OpenAI Chat Model node in n8n")

    print(f"\n{'=' * 60}")
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Open the Mira n8n workflow")
    print("2. Click N16's OpenAI Chat Model sub-node -> Model field -> Enter Manually")
    print("3. Paste the fine-tuned model ID above")
    print("4. Save the workflow")
    print("5. Re-run all test inputs from test_risk_assessor.py through N16")
    print("6. Paste the new outputs into test_risk_assessor.py's PIPELINE_OUTPUTS")
    print("7. Run: python test_risk_assessor.py")
    print("8. Compare before vs after results")


if __name__ == "__main__":
    main()
