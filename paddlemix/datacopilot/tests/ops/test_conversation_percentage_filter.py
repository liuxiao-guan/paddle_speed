from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.filter._conversation_percentage_filter import conversation_percentage_filter

# Path to the dataset
anno_path = 'random_samples.json'

# Load the dataset
print("Loading the dataset...")
dataset = MMDataset.from_json(anno_path)
print("Initial dataset size:", len(dataset))

# Apply the conversation percentage filter
min_percentile = 5   # Minimum percentile
max_percentile = 95  # Maximum percentile
dataset = dataset.conversation_percentage_filter(min_percentile=min_percentile, max_percentile=max_percentile)

# Print the size of the filtered dataset
print("Filtered dataset size:", len(dataset))
print("Conversation percentage filtering complete.")

# Export the filtered dataset
dataset.export_json(anno_path.replace('.json', '_conversation_percentage_filter.json'))