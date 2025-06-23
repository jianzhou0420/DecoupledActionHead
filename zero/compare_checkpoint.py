"""Code Generated from Google Gemini, verified by Jian."""
from collections import defaultdict
import torch


def compare_checkpoints(ckpt_path1, ckpt_path2, rtol=1e-05, atol=1e-08):
    """
    Finds all common parameter tensors (weights, biases, etc.) between two checkpoints
    and checks if they are numerically close. (This function is unchanged).
    """
    # Load the checkpoints onto the CPU to avoid potential GPU memory issues.
    checkpoint1 = torch.load(ckpt_path1, map_location='cpu')
    checkpoint2 = torch.load(ckpt_path2, map_location='cpu')

    # The state_dict can be the top-level object or nested under the 'state_dict' key.
    state_dict1 = checkpoint1.get('state_dict', checkpoint1)
    state_dict2 = checkpoint2.get('state_dict', checkpoint2)

    # Get the keys, which are the names of each parameter.
    param_names1 = set(state_dict1.keys())
    param_names2 = set(state_dict2.keys())

    # Find the intersection to get all parameters that exist in both checkpoints.
    common_param_names = param_names1.intersection(param_names2)

    if not common_param_names:
        return None

    # Initialize dictionaries to store the results.
    results = {
        'close_params': [],
        'different_params': [],
        'mismatched_shape_params': []
    }

    # Iterate through each common parameter name to compare the tensors.
    for param_name in sorted(list(common_param_names)):
        tensor1 = state_dict1[param_name]
        tensor2 = state_dict2[param_name]

        if tensor1.shape != tensor2.shape:
            results['mismatched_shape_params'].append({
                'name': param_name,
                'shape1': tensor1.shape,
                'shape2': tensor2.shape
            })
            continue

        are_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
        mae_difference = torch.mean(torch.abs(tensor1 - tensor2)).item()

        if are_close:
            results['close_params'].append({'name': param_name, 'mae': mae_difference})
        else:
            results['different_params'].append({'name': param_name, 'mae': mae_difference})

    return results


def print_friendly_report(results, ckpt_path1, ckpt_path2):
    """
    Takes the results from compare_checkpoints and prints them in a grouped,
    human-friendly format.
    """
    print("=" * 60)
    print("      Checkpoint Comparison Report")
    print("=" * 60)
    print(f"File 1: {ckpt_path1}")
    print(f"File 2: {ckpt_path2}\n")

    # --- Group parameters by layer name ---
    grouped_layers = defaultdict(list)
    all_params = results['close_params'] + results['different_params']

    for param in all_params:
        # Split 'layer.sub_layer.weight' into 'layer.sub_layer' and 'weight'
        parts = param['name'].rsplit('.', 1)
        if len(parts) == 2:
            layer_name, param_type = parts
        else:
            # Handle cases where the parameter name has no '.'
            layer_name, param_type = param['name'], 'tensor'

        status = 'close' if param in results['close_params'] else 'different'
        grouped_layers[layer_name].append({'type': param_type, 'status': status, 'mae': param['mae']})

    # --- Determine overall status for each group ---
    report = defaultdict(list)
    for layer_name, params in grouped_layers.items():
        statuses = {p['status'] for p in params}
        if len(statuses) == 1 and 'close' in statuses:
            overall_status = '[CLOSE]'
        elif len(statuses) == 1 and 'different' in statuses:
            overall_status = '[DIFFERENT]'
        else:
            overall_status = '[MIXED]'
        report[overall_status].append({'layer_name': layer_name, 'params': params})

    # --- Print Summary ---
    total_layers = len(grouped_layers)
    close_count = len(report['[CLOSE]'])
    different_count = len(report['[DIFFERENT]'])
    mixed_count = len(report['[MIXED]'])
    mismatched_count = len(results['mismatched_shape_params'])

    print("--- Summary ---")
    print(f"Found {total_layers} common layer groups with matching shapes.")
    print(f"  ✅ {close_count} layer groups are entirely CLOSE.")
    print(f"  ❌ {different_count} layer groups are entirely DIFFERENT.")
    print(f"  🔄 {mixed_count} layer groups have MIXED results (some params close, some not).")
    print(f"  ⚠️ Found {mismatched_count} parameters with MISMATCHED shapes.\n")

    # --- Print Detailed Report ---
    for status_key, status_label in [('[CLOSE]', '✅ Layer Groups are CLOSE'),
                                     ('[DIFFERENT]', '❌ Layer Groups are DIFFERENT'),
                                     ('[MIXED]', '🔄 Layer Groups have MIXED Results')]:
        if report[status_key]:
            print(f"--- {status_label} ---")
            for item in sorted(report[status_key], key=lambda x: x['layer_name']):
                print(f"  {item['layer_name']}")
                for param in item['params']:
                    symbol = '✅' if param['status'] == 'close' else '❌'
                    print(f"    {symbol} {param['type']:<10} (MAE: {param['mae']:.2e})")
            print()

    if results['mismatched_shape_params']:
        print(f"--- ⚠️ Parameters with MISMATCHED Shapes ---")
        for param in results['mismatched_shape_params']:
            print(f"  - {param['name']} (Shapes: {param['shape1']} vs {param['shape2']})")
        print()


# --- Main execution block ---
if __name__ == '__main__':
    # --- IMPORTANT ---
    # Replace with the actual paths to your checkpoint files
    # stage1_epoch50_ckpt_path = "/media/jian/ssd4t/DP/first/data/outputs/Archive/2025.06.08_00.00.00_pretrain_JPee_stage1_stack_d1_1000/checkpoint_epoch=049.ckpt"
    # stage2_epoch50_ckpt_path = "data/outputs/Archive/2025.06.08_21.00.23_pretrain_JPee_stage2_stack_d1_1000/checkpoint_epoch=049.ckpt"
    stage1_epoch50_ckpt_path = "data/robomimic/Stage1/tmp/DecoupleActhonHead_stage1_None_1000_epoch=049.ckpt"
    stage2_epoch50_ckpt_path = "/media/jian/ssd4t/DP/first/data/outputs/2025.06.22/20.55.19_stage2_H_mug_cleanup_d1_1000_ABC_D/checkpoints/stage2_H_mug_cleanup_d1_1000_ABC_D_epoch=039.ckpt"

    try:
        comparison_results = compare_checkpoints(stage1_epoch50_ckpt_path, stage2_epoch50_ckpt_path)

        if comparison_results is None:
            print("No common parameter names were found between the two checkpoints.")
        else:
            print_friendly_report(comparison_results, stage1_epoch50_ckpt_path, stage2_epoch50_ckpt_path)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the checkpoint file paths are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
