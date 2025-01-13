import json

def update_json_keys(input_file, output_file, old_prefix, new_prefix):
    """
    Update the keys in a JSON file by replacing the old_prefix with new_prefix.

    :param input_file: Path to the input JSON file.
    :param output_file: Path to save the updated JSON file.
    :param old_prefix: The prefix in the keys to be replaced.
    :param new_prefix: The new prefix to replace the old one.
    """
    try:
        # Load the JSON data from the input file
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        # Update keys by replacing the prefix
        updated_data = {key.replace(old_prefix, new_prefix): value for key, value in data.items()}

        # Save the updated JSON data to the output file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(updated_data, outfile, ensure_ascii=False, indent=4)

        print(f"File has been successfully updated and saved to: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    input_file_path = "/root/autodl-tmp/LLaVA/playground/data/eval/mm-vet/results/llava-v1.6-vicuna-7b_test_1.json"  # Replace with your input file path
    output_file_path = "/root/autodl-tmp/MMVET-trans/llava-v1.6-vicuna-7b_test_1.json"  # Replace with your output file path
    old_prefix = "v1_v1_"
    new_prefix = "v1_"

    update_json_keys(input_file_path, output_file_path, old_prefix, new_prefix)
