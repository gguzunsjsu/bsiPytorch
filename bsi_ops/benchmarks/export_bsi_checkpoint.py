import os
import json
import argparse
from datetime import datetime

import torch
from transformers import AutoTokenizer, OPTForCausalLM
from verify_accuracy_bsi import quantize_model_bsi, summarize_bsi_model


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Export BSI quantization spec and (optionally) the dense state_dict")
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='Model name or HF repo id')
    parser.add_argument('--weights', type=str, default=None, help='Optional path to a local state_dict .pt to load before export')
    parser.add_argument('--pf', type=int, default=127, help='Precision factor used to build BSI keys')
    parser.add_argument('--scope', type=str, default='all', choices=['all', 'attention', 'mlp'], help='Which Linear layers to quantize')
    parser.add_argument('--out_dir', type=str, default='bsi_export', help='Output directory to write spec and weights')
    parser.add_argument('--no_state_dict', action='store_true', help='Do not write a dense state_dict copy to disk')
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)

    # Load dense model on CPU
    print(f"Loading base model: {args.model_name}")
    model = OPTForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map='cpu')
    if args.weights and os.path.isfile(args.weights):
        print(f"Loading provided state_dict from {args.weights}")
        sd = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(sd, strict=False)

    # Optionally save dense state_dict for reproducibility
    weights_path = None
    if not args.no_state_dict:
        weights_path = os.path.join(out_dir, 'model_state.pt')
        print(f"Saving dense state_dict to {weights_path}")
        torch.save(model.state_dict(), weights_path)

    # Build a quantized copy (CPU), summarize storage
    print(f"Quantizing copy with pf={args.pf}, scope={args.scope} for summary...")
    model_copy = OPTForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map='cpu')
    if args.weights and os.path.isfile(args.weights):
        model_copy.load_state_dict(sd, strict=False)
    model_copy = quantize_model_bsi(model_copy, precision_factor=args.pf, scope=args.scope)
    summary = summarize_bsi_model(model_copy)

    # Persist a spec that can be used to rebuild BSI keys on load
    spec = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'model_name': args.model_name,
        'precision_factor': int(args.pf),
        'scope': args.scope,
        'num_quantized_layers': int(summary['num_quantized_layers']),
        'layers': [
            {
                'name': l['name'],
                'in_features': int(l['in_features']),
                'out_features': int(l['out_features'])
            } for l in summary['layers']
        ],
        'storage_bytes': {
            'bsi_weights': int(summary['total_bsi_bytes']),
            'bias': int(summary['total_bias_bytes']),
            'bsi_total': int(summary['total_static_bytes']),
            'dense_linear_reference': int(summary['total_dense_bytes'])
        },
        'dense_state_dict_path': weights_path,
        'note': (
            'This spec does not include serialized BSI keys. On load, dense weights are needed '
            'to rebuild BSI keys with the same precision factor and scope.'
        ),
    }

    spec_path = os.path.join(out_dir, 'bsi_spec.json')
    with open(spec_path, 'w') as f:
        json.dump(spec, f, indent=2)
    print(f"Wrote BSI spec to {spec_path}")

    # Print a brief report
    mb = lambda b: b / (1024**2)
    print("\nSummary:")
    print(f"  Quantized layers: {summary['num_quantized_layers']}")
    print(f"  BSI weights: {mb(summary['total_bsi_bytes']):.2f} MB | Bias: {mb(summary['total_bias_bytes']):.2f} MB | Total: {mb(summary['total_static_bytes']):.2f} MB")
    print(f"  Dense linear reference: {mb(summary['total_dense_bytes']):.2f} MB")
    if weights_path:
        print(f"  Dense state_dict saved: {weights_path}")


if __name__ == '__main__':
    main()
