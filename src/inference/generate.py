"""
Boxing-GPT — Inference & Generation
=====================================
Load a trained model and generate text interactively.

Usage:
    python -m src.inference.generate \\
        --checkpoint checkpoints/checkpoint_step50000_best.pt \\
        --tokenizer  data/tokenizer/ \\
        --prompt     "How do I improve my jab?" \\
        --max_tokens 200 \\
        --temperature 0.8 \\
        --top_k 40

Or run in interactive chat mode:
    python -m src.inference.generate --interactive
"""

import argparse
import torch
from ..model.gpt import BoxingGPT
from ..tokenizer.bpe import BPETokenizer


# ─────────────────────────────────────────────
#  Generator class (wrapper for clean usage)
# ─────────────────────────────────────────────

class Generator:
    """
    Wrapper for loading a trained BoxingGPT and generating text.

    Args:
        checkpoint_path : path to .pt checkpoint file
        tokenizer_dir   : path to tokenizer directory
        device          : 'cuda' | 'mps' | 'cpu'
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_dir: str,
        device: str = 'cpu',
    ):
        self.device = device

        # Load tokenizer
        print(f"[Generator] Loading tokenizer from {tokenizer_dir} ...")
        self.tokenizer = BPETokenizer.load(tokenizer_dir)

        # Load model
        print(f"[Generator] Loading model from {checkpoint_path} ...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        cfg = checkpoint['config']
        # cfg here is the training config dict saved in trainer.py
        # Model config is stored separately in config.yaml
        model_cfg = self._load_model_config()

        self.model = BoxingGPT(
            vocab_size=model_cfg['vocab_size'],
            context_length=model_cfg['context_length'],
            n_layers=model_cfg['n_layers'],
            n_heads=model_cfg['n_heads'],
            d_model=model_cfg['d_model'],
            d_ff=model_cfg['d_ff'],
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f"[Generator] Model loaded. Ready to generate.")

    def _load_model_config(self) -> dict:
        """
        Load model configuration from config.yaml.
        
        The checkpoint only stores training config, not model architecture.
        Model config (vocab_size, n_layers, etc.) is in config.yaml.
        """
        import os
        import yaml
        
        # Try to find config.yaml relative to this file or from project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, 'src', 'configs', 'config.yaml')
        
        if not os.path.exists(config_path):
            # Try current working directory
            config_path = 'src/configs/config.yaml'
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        return cfg['model']

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty=1.3,
    ) -> str:
        """
        Generate text given a prompt string.

        Args:
            prompt         : input text prompt
            max_new_tokens : how many new tokens to generate
            temperature    : sampling temperature
            top_k          : top-k sampling
            top_p          : nucleus sampling threshold

        Returns:
            Generated text (prompt + continuation)
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_id,
                repetition_penalty=repetition_penalty,
            )

        # Decode (skip the input prompt tokens)
        generated_ids = output_ids[0].tolist()
        full_text = self.tokenizer.decode(generated_ids)
        return full_text

    def interactive(self) -> None:
        """Run an interactive chat loop in the terminal."""
        print("\n" + "="*60)
        print("  🥊 BoxingGPT — Interactive Mode")
        print("  Type your question or prompt. Ctrl+C to exit.")
        print("="*60 + "\n")

        while True:
            try:
                prompt = input("You: ").strip()
                if not prompt:
                    continue

                print("\nBoxingGPT: ", end="", flush=True)
                response = self.generate(prompt)

                # Print only the generated part (after the prompt)
                # Simple heuristic: the prompt is at the start
                if response.startswith(prompt):
                    print(response[len(prompt):].strip())
                else:
                    print(response)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye, champ! 🥊")
                break


# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BoxingGPT Text Generation")
    parser.add_argument('--checkpoint',   type=str, required=True)
    parser.add_argument('--tokenizer',    type=str, default='data/tokenizer/')
    parser.add_argument('--prompt',       type=str, default='')
    parser.add_argument('--max_tokens',   type=int, default=200)
    parser.add_argument('--temperature',  type=float, default=0.8)
    parser.add_argument('--top_k',        type=int, default=40)
    parser.add_argument('--top_p',        type=float, default=0.9)
    parser.add_argument('--interactive',  action='store_true')
    parser.add_argument('--device',       type=str, default='cpu')
    args = parser.parse_args()

    generator = Generator(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        device=args.device,
    )

    if args.interactive:
        generator.interactive()
    elif args.prompt:
        output = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(output)
    else:
        print("Provide --prompt or --interactive")


if __name__ == '__main__':
    main()
