import json
from pathlib import Path

def main():
    input_path = "frame_0_candidates_exported.json"
    output_path = "frame_0_multi_cmp_candidates.json"
    
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found.")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # 筛选 cur_cmp 长度大于 1 的项
    multi_cmp_results = [
        item for item in data 
        if len(item.get("result", {}).get("cur_cmp", [])) > 1
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(multi_cmp_results, f, ensure_ascii=False, indent=2)
        
    print(f"Filter complete. Found {len(multi_cmp_results)} objects with multiple candidates in cur_cmp.")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
