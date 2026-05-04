import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class CandidateClassifier:
    def __init__(self, data_dir="data"):
        # Load a fast, lightweight embedding model suitable for CPU
        print("Loading embedding model (this takes a few seconds the first time)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load data
        self.rubric = self._load_json(os.path.join(data_dir, "rubric.json"))
        self.candidates = self._load_json(os.path.join(data_dir, "candidates.json"))
        
        # Pre-compute embeddings for tracks and degrees so we don't recalculate them
        self.track_names = [t["id"] for t in self.rubric["tracks"]]
        self.track_texts = [f"{t['name']} {t['description']}" for t in self.rubric["tracks"]]
        self.track_embeddings = self.model.encode(self.track_texts)
        
        self.degree_names = list(self.rubric["degrees"].keys())
        self.degree_texts = list(self.rubric["degrees"].values())
        self.degree_embeddings = self.model.encode(self.degree_texts)

    def _load_json(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _generate_rationale(self, candidate, top_track_id, recommended_degree):
        # Dynamically generate a rationale based on the algorithmic matches
        track_name = next(t["name"] for t in self.rubric["tracks"] if t["id"] == top_track_id)
        
        return (f"Based on holistic vector similarity, the candidate's background in "
                f"[{candidate['degree']}] and expressed interest in [{candidate['sop'][:50]}...] "
                f"mathematically aligns closest with the '{track_name}' track. "
                f"The semantic focus of their profile strongly indicates a {recommended_degree} "
                f"as the optimal degree pathway.")

    def classify_single(self, candidate):
        # Create a rich text representation of the candidate for the model to "read"
        candidate_text = f"{candidate['degree']}. Subjects: {candidate['subjects']}. Experience: {candidate['experience']}. Goal: {candidate['sop']}"
        candidate_vec = self.model.encode([candidate_text])
        
        # 1. Find Top 3 Career Tracks
        track_similarities = cosine_similarity(candidate_vec, self.track_embeddings)[0]
        # Get indices of top 3 scores
        top_3_indices = np.argsort(track_similarities)[::-1][:3]
        top_3_tracks = [self.track_names[i] for i in top_3_indices]
        
        # 2. Find Best Fit Degree
        degree_similarities = cosine_similarity(candidate_vec, self.degree_embeddings)[0]
        best_degree_index = np.argmax(degree_similarities)
        recommended_degree = self.degree_names[best_degree_index]
        
        # 3. Generate Rationale
        rationale = self._generate_rationale(candidate, top_3_tracks[0], recommended_degree)
        
        return {
            "candidate_id": candidate["id"],
            "top_3_career_tracks": top_3_tracks,
            "recommended_degree": recommended_degree,
            "rationale": rationale
        }

    def run_pipeline(self):
        print(f"Processing {len(self.candidates)} candidates...")
        results = []
        for candidate in self.candidates:
            result = self.classify_single(candidate)
            results.append(result)
        return results