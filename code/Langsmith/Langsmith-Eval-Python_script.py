from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

DATASET_ID = "0ae436fa-a175-4056-b2f8-8dcfdbae88d4"  # from the check_datasets.py output

# 1. Fetch all examples from the dataset
examples = list(client.list_examples(dataset_id=DATASET_ID))

if not examples:
    raise ValueError(f"No examples found in dataset {DATASET_ID}. Please check your dataset.")

# 2. Dynamically extract the actual dictionary keys present in your dataset
sample_input_keys = list(examples[0].inputs.keys())
sample_output_keys = list(examples[0].outputs.keys()) if examples[0].outputs else []

if not sample_input_keys:
    raise ValueError("The dataset examples do not contain any input keys.")

# Auto-assign the first available key names
INPUT_KEY = sample_input_keys[0]
# Use the first output key if available, otherwise default to a fallback string
OUTPUT_KEY = sample_output_keys[0] if sample_output_keys else "actual_reference_output"

print(f" Detected dataset input key: '{INPUT_KEY}'")
print(f" Detected dataset output key: '{OUTPUT_KEY}'\n")

# 3. Build the lookup dictionary safely using the detected keys
lookup = {
    ex.inputs[INPUT_KEY]: ex.outputs.get(OUTPUT_KEY, "")
    for ex in examples
}

# 4. Define the target pipeline using the dynamic key
def target(inputs: dict) -> dict:
    # If the evaluator passes the whole dictionary, extract the text using our discovered key
    input_text = inputs.get(INPUT_KEY, "")
    return {"output": lookup.get(input_text, "")}

# 5. Run the evaluation
results = evaluate(
    target,
    data=DATASET_ID,
    experiment_prefix="mira-eval-allTsusecases",
)

print("\nEvaluation Results:")
print(results)
