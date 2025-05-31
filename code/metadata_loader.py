import json

class MetadataLoader:
    def __init__(self, metadata_path='data/rag_metadata.json'):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def get_chunks_by_indices(self, indices):
        return [self.data[i]["text"] for i in indices]
