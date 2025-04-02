import os
import glob
import argparse
import torch
from Chessgpt import GPT, GPTConfig, load_checkpoint, generate_sample_response
from tokenizer import Tokenizer

# Allow safe unpickling of GPTConfig
torch.serialization.add_safe_globals([GPTConfig])

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the most recent final checkpoint (by creation time) under a run folder."""
    list_ckpt = glob.glob(os.path.join(checkpoint_dir, "run_*", "final", "*.pt"))
    if not list_ckpt:
        return None
    latest = max(list_ckpt, key=os.path.getctime)
    return latest

def main():
    parser = argparse.ArgumentParser(description="Inference for ChessGPT")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint. If not set, the latest final checkpoint is used.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt (chess moves) to generate continuation from")
    parser.add_argument("--max_length", type=int, default=20,
                        help="Maximum generated token length")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of generation samples")
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    
    tokenizer = Tokenizer("vocabs/kaggle2_vocab.txt")
    
    # Use provided checkpoint or find the latest one
    if args.checkpoint is None:
        ckpt_path = find_latest_checkpoint()
        if ckpt_path is None:
            print("No checkpoint found!")
            return
        print(f"No checkpoint specified. Using latest checkpoint: {ckpt_path}")
    else:
        ckpt_path = args.checkpoint

    model, ckpt = load_checkpoint(ckpt_path, device)
    model.eval()
    max_length=len(args.prompt.split())+2
    # Generate sample responses from the given prompt:
    samples = generate_sample_response(model, tokenizer, device,
                                       prompt=args.prompt,
                                       max_length=max_length,
                                       num_return_sequences=args.num_return_sequences)
    for i, sample in enumerate(samples):
        print(f"\n-- Sample {i+1} --\n{sample}\n")

if __name__ == "__main__":
    main()

#python inference.py --prompt "e4 e5 Nf3" --max_length 30 --num_return_sequences 3
#python inference.py --checkpoint "checkpoints/run_20250402_045307/final/model_step_4.pt" --prompt "e4 e5 Nf3" --max_length 5 --num_return_sequences 3