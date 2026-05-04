import json
import os
from embedding_engine import CandidateClassifier

def save_output(data, output_dir="outputs", filename="final_recommendations.json"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"\nSuccess! Results saved to {filepath}")

if __name__ == "__main__":
    print("Initializing CARA Pipeline Classification Engine...")
    
    # Initialize our ML engine
    classifier = CandidateClassifier(data_dir="data")
    
    # Run the classification
    final_results = classifier.run_pipeline()
    
    # Save the deliverables
    save_output(final_results)