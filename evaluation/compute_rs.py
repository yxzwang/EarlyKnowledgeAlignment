import json
import os
import argparse
from tqdm import tqdm

# Make sure the file `eval_r.py` containing the `cal_rsim` function
# is in the same directory or accessible in your Python path.
from eval_r import cal_rsim

def calculate_rsim_for_item(d):
    """
    Processes a single data item to calculate and add the r-sim score.

    Args:
        d (dict): A dictionary representing one data sample.

    Returns:
        dict: The updated dictionary with the 'r_sim' score.
    """
    try:
        # 1. Deduplicate context from the source data
        context_list = []
        for c in d.get('context', []):
            if c not in context_list:
                context_list.append(c)
        context_str = '\n'.join(context_list)

        # 2. Extract all knowledge snippets from the prediction string
        knowledge_parts = []
        prediction_str = d.get('prediction', '')
        
        # Split by the closing tag and ignore the final part of the string
        ksplit = prediction_str.split("</knowledge>")[:-1]
        for part in ksplit:
            # For each part, extract the content after the opening tag
            content = part.split("<knowledge>")[-1]
            knowledge_parts.append(content.strip())
        
        knowledge_str = '\n'.join(knowledge_parts)

        # 3. Calculate r-sim score
        # The score is 0.0 if no knowledge was extracted.
        # cal_rsim expects a list of contexts and a list of knowledge strings.
        rsim_score = cal_rsim([context_str], [knowledge_str]) if knowledge_str else 0.0

        # 4. Add the calculated score to the dictionary
        d['r_sim'] = rsim_score

    except Exception as e:
        print(f"\n[ERROR] Failed to process item with question: '{d.get('question', 'N/A')}'")
        print(f"Error details: {e}")
        # Add an error value to easily identify failed items
        d['r_sim'] = -1.0
        d['error'] = str(e)
    
    return d

def main():
    """
    Main function to orchestrate reading, processing, and writing the data.
    """
    parser = argparse.ArgumentParser(
        description="Calculate R-Sim score for each item in a JSON file and add it to the data."
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        default= "expr_results/Qwen2.5-7B-Instruct_2WikiMultiHopQA_grpo/results_step40.json",
        help="Path to the input JSON file containing the dataset."
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        help="Path to save the output JSON file. If not provided, a new file is created next to the input file with a '_with_rsim' suffix."
    )
    args = parser.parse_args()

    # --- Load data from the input file ---
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Input file not found at: {args.input_file}")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from file: {args.input_file}")
        return

    if not isinstance(data, list):
        print("[ERROR] The input JSON file must contain a list of objects.")
        return

    # --- Process each item to calculate R-Sim ---
    updated_data = []
    print(f"Processing {len(data)} items from '{os.path.basename(args.input_file)}'...")
    for item in tqdm(data, desc="Calculating R-Sim"):
        updated_item = calculate_rsim_for_item(item)
        updated_data.append(updated_item)

    # --- Save the updated data to the output file ---
    if args.output_file:
        output_path = args.output_file
    else:
        # Create a default output file name if not provided
        base, ext = os.path.splitext(args.input_file)
        output_path = f"{base}_with_rsim{ext}"

    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n[SUCCESS] Processing complete.")
        print(f"Updated data with R-Sim scores saved to: {output_path}")
    except IOError as e:
        print(f"\n[ERROR] Could not write to output file '{output_path}': {e}")

if __name__ == "__main__":
    main()