#!/usr/bin/env python3

import os
import argparse

from data import dataset
from lm.tokenizer import SmilesTokenizer
from training import train_lm, train_sae
from lm.model.transformer import Decoder

def main(args):
    project_dir = args.project_dir if args.project_dir is not None else "./MolSAE"

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    if args.train:
        if args.model_type == "lm":
            if args.data_source is None:
                dataset.save_tokenized_ds(project_dir)

            tokenizer = SmilesTokenizer().load(f"{args.data_source}/zinc_tokenizer.pkl")
            
            lm = Decoder(
                d_model=args.model_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                vocab_size=tokenizer.vocab_size
            )

            ds, get_eval_ds = dataset.create_lm_dataset(
                train_bsize=args.batch_size,
                pad_token_id=tokenizer.pad_token_id,
                buffer_size=args.data_bufsize
            )

            train_config = {
                "num_steps": args.steps,
                "learning_rate": args.learning_rate,
                "run_id": args.run_id,
                "ckpt_base_dir": args.checkpoint_dir,
                "project_name": args.project_name,
                "resume": args.resume
            }

            train_lm.train(model=lm, train_ds=ds, get_eval_ds=get_eval_ds, config=train_config)

        elif args.model_type == "sae":
            # TODO!!

            raise NotImplementedError("SAE training is unavailable!")

            # sae = SAE(
            #     input_dim=args.input_dim,
            #     latent_dim=args.latent_dim,
            #     # Add other necessary parameters
            # )

            # ds = dataset.create_sae_dataset(
            #     train_bsize=args.batch_size,
            #     buffer_size=args.bufsize
            # )

            # config = {
            #     "num_steps": args.num_steps,
            #     "learning_rate": args.learning_rate,
            #     "run_id": args.run_id,
            #     "ckpt_base_dir": args.ckpt_base_dir,
            #     "project_name": args.project_name,
            #     "resume": args.resume
            # }

            # train_sae.train(model=sae, train_ds=ds, get_eval_ds=None, config=config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Molecular LMs and SAEs.")

    # select mode
    parser.add_argument("--train", action="store_true", help="Enable training mode.")
    parser.add_argument("--sample", action="store_true", help="Sample from a language model.")

    # set directories
    parser.add_argument("--project-dir", type=str, help="Directory for the project.")
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/checkpoints", help="Directory for storing checkpoints.")
    
    # set training parameters
    parser.add_argument("--model-type", type=str, choices=["lm", "sae"], help="Model type to train.")
    parser.add_argument("--steps", type=int, help="Number of training steps.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--data-source", type=str, help="Directory for existing dataset.")
    parser.add_argument("--batch-size", type=int, help="Batch size for training.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if exists.")
    parser.add_argument("--data-bufsize", type=int, default=50_000, help="Buffer size for dataset.")
    parser.add_argument("--project-name", type=str, default="MolSAE", help="Weights and Biases project name.")
    parser.add_argument("--run-id", type=str, default="default-run", help="Weights and Biases run ID.")

    # set lm parameters
    parser.add_argument("--model-dim", type=int, help="Hidden size for LM model.")
    parser.add_argument("--hidden-dim", type=int, help="Hidden size for LM model.")
    parser.add_argument("--num-layers", type=int, help="Number of layers for LM model.")
    
    #set sae parameters
    parser.add_argument("--input-dim", type=int, help="Input dimension for SAE.")
    parser.add_argument("--latent-dim", type=int, help="Latent dimension for SAE.")
   
