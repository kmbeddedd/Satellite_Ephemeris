"""
Unified CLI Entrypoint for GNSS Satellite Error Forecasting
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="GNSS Satellite Orbit & Clock Error Forecasting CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model bilstm --data FINAL_Data.csv --output ./gnss_results
  python main.py --model transformer --data FINAL_Data.csv --output ./transformer_results --enable-diffusion
  python main.py --model tune --data FINAL_Data.csv --n-trials 15
        """
    )
    parser.add_argument(
        "--model",
        choices=["bilstm", "transformer", "tune"],
        default="bilstm",
        help="Pipeline to run: 'bilstm' (Keras BiLSTM+GRU), 'transformer' (PyTorch Deep Transformer), or 'tune' (Optuna Tuning)"
    )

    # Pass remaining arguments to the chosen pipeline
    args, remaining_argv = parser.parse_known_args()

    # Forward arguments
    sys.argv = [sys.argv[0]] + remaining_argv

    if args.model == "bilstm":
        from train_bilstm import run_training
        run_training()
    elif args.model == "transformer":
        from train_transformer import run_training
        run_training()
    elif args.model == "tune":
        from tune import run_tuning
        run_tuning()


if __name__ == "__main__":
    main()
