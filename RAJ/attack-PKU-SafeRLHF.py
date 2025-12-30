from models import LLMmodel
import pandas as pd
from tqdm import tqdm
import os
import argparse

def process_dataset_with_ids(model_name, dataset_name, start_id, end_id):
    """
    Process different types of questions within specified ID range for a specific model and dataset, and save results.
    """
    print(f"Running model: {model_name}")

    model = LLMmodel(model_name)
    
    # Build file path
    rewrite_data_path = f'data/origin/{dataset_name}.csv'

    # Read dataset and filter specified ID range
    rewrites_df = pd.read_csv(rewrite_data_path)
    rewrites_df = rewrites_df.iloc[start_id:end_id].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Get question list, using original ID as index
    questions_dict = {
        'question': dict(zip(rewrites_df.index.tolist(), rewrites_df['question'].tolist())),
        'concrete': dict(zip(rewrites_df.index.tolist(), rewrites_df['concrete'].tolist())),
    }
    
    # Create results directory (if it doesn't exist)
    results_dir = f'results/{model_name}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Process all three types of data uniformly
    for data_type, questions_with_ids in questions_dict.items():
        # *** Main modification here: add ID range to filename ***
        result_path = f'{results_dir}/{dataset_name}_{data_type}_{start_id}_{end_id}.csv'
        
        # Try to read existing results, create empty DataFrame if not found
        if os.path.exists(result_path):
            df_results = pd.read_csv(result_path, dtype={'id': str})
            processed_ids = set(df_results['id'].tolist())
            results_list = df_results.to_dict('records')
        else:
            df_results = pd.DataFrame(columns=['id', 'question', 'reasoning', 'response'])
            processed_ids = set()
            results_list = []
            
        print(f"Starting to process {data_type} data, ID range: {start_id} to {end_id}...")

        # Iterate through all questions
        for current_id, question in tqdm(questions_with_ids.items()):
            current_id = str(current_id)
            
            # Skip if ID has already been processed
            if current_id in processed_ids:
                continue

            # Generate response
            try:
                reasoning, response = model.generate(question)
                results_list.append({'id': current_id, 'question': question, 'reasoning': reasoning, 'response': response})
                # Save results in real-time
                pd.DataFrame(results_list).to_csv(result_path, index=False)
            except Exception as e:
                print(f"Error processing question for ID {current_id}: {question}, error: {e}")
                continue

    print(f"Model {model_name} finished running.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process dataset within specified ID range.")
    parser.add_argument('--model_name', type=str, default='R1-32B', help='Model name to run.')
    parser.add_argument('--dataset_name', type=str, default='PKU-SafeRLHF-prompts', help='Dataset name to process.')
    parser.add_argument('--start_id', type=int, default=0, help='Start ID (inclusive).')
    parser.add_argument('--end_id', type=int, default=999, help='End ID (exclusive).')
    
    args = parser.parse_args()

    process_dataset_with_ids(args.model_name, args.dataset_name, args.start_id, args.end_id)