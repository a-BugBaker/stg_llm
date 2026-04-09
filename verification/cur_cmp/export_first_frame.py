import json
from pathlib import Path
from stg_system.candidate_resolver import get_candidates
from stg_system.config import EngineConfig
from stg_system.models import GraphState

def main():
    # Load first frame
    data = json.loads(Path("data/less_move.json").read_text(encoding="utf-8"))
    frame_0 = data[0]
    objects = frame_0["objects"]
    
    cfg = EngineConfig() # Default config
    graph = GraphState()
    
    results = []
    from dataclasses import asdict
    
    print(f"Processing {len(objects)} objects in first frame...")
    for i, obj in enumerate(objects):
        idx = int(obj["idx"])
        result = get_candidates(idx, objects, 0, graph, cfg)
        results.append({
            "idx": idx,
            "label": obj["label"],
            "result": asdict(result)
        })
        
    output_path = "frame_0_candidates_exported.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results exported to {output_path}")

if __name__ == "__main__":
    main()
