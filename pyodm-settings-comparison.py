import pyodm
import os
import json
from typing import List, Dict, Any

class ODMSettingsExplorer:
    def __init__(self, node_url='http://localhost:3000'):
        """
        Initialize connection to ODM node
        
        :param node_url: URL of the ODM processing node
        """
        self.node = pyodm.Node(node_url)
        self.results = []
    
    def generate_processing_configs(self, base_options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a set of configuration variations
        
        :param base_options: Base configuration dictionary
        :return: List of configuration dictionaries
        """
        config_variations = [
            # Variation 1: Default
            base_options.copy(),
            
            # Variation 2: Low resolution
            {**base_options, 'orthophoto-resolution': 10},
            
            # Variation 3: High resolution
            {**base_options, 'orthophoto-resolution': 2},
            
            # Variation 4: Additional processing options
            {**base_options, 
             'dsm': True, 
             'dtm': True}
        ]
        
        return config_variations
    
    def process_with_config(self, images_path: str, config: Dict[str, Any]) -> Dict:
        """
        Process images with a specific configuration
        
        :param images_path: Path to directory with drone images
        :param config: Processing configuration
        :return: Processing result summary
        """
        try:
            # Create task with specific configuration
            task = self.node.create_task(images_path, options=config)
            
            # Wait for processing to complete
            task.wait_for_completion()
            
            # Prepare result summary
            result = {
                'config': config,
                'task_id': task.id,
                'status': task.status,
                'processing_time': task.elapsed_time if hasattr(task, 'elapsed_time') else None
            }
            
            # Add output information if task completed
            if task.status == pyodm.STATUS_COMPLETED:
                result['outputs'] = task.outputs
            
            return result
        
        except Exception as e:
            print(f"Error processing with config {config}: {e}")
            return {
                'config': config,
                'status': 'error',
                'error_message': str(e)
            }
    
    def compare_processing_settings(self, images_path: str):
        """
        Compare different processing settings
        
        :param images_path: Path to directory with drone images
        """
        # Base configuration
        base_options = {
            'orthophoto-resolution': 5,
            'verbose': True
        }
        
        # Generate configuration variations
        configs = self.generate_processing_configs(base_options)
        
        # Process images with each configuration
        self.results = [
            self.process_with_config(images_path, config) 
            for config in configs
        ]
        
        # Print comparison results
        self.print_comparison_results()
    
    def print_comparison_results(self):
        """
        Print a comparison of processing results
        """
        print("\n=== Processing Settings Comparison ===")
        for result in self.results:
            print("\nConfiguration:")
            print(json.dumps(result.get('config', {}), indent=2))
            print(f"Status: {result.get('status', 'N/A')}")
            print(f"Processing Time: {result.get('processing_time', 'N/A')}")
            
            # Print output files if available
            if result.get('outputs'):
                print("Output Files:")
                for output in result['outputs']:
                    print(f"- {output}")
    
    def export_results(self, output_file='odm_comparison_results.json'):
        """
        Export comparison results to a JSON file
        
        :param output_file: Path to output JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults exported to {output_file}")

def main():
    # Path to directory with drone images
    image_directory = '/path/to/drone/images'
    
    # Initialize and run settings explorer
    explorer = ODMSettingsExplorer()
    
    try:
        # Compare processing settings
        explorer.compare_processing_settings(image_directory)
        
        # Export results
        explorer.export_results()
    
    except Exception as e:
        print(f"Error in processing: {e}")

if __name__ == "__main__":
    main()
