from langsmith import Client

client = Client()

print("Datasets visible to this API key:\n")
for d in client.list_datasets():
    print(f"- {d.name}  (id: {d.id})")