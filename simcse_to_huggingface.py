"""
Convert SimCSE's checkpoints to Huggingface style.
"""

import argparse
import torch
import os
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path of SimCSE checkpoint folder")
    args = parser.parse_args()

    print("SimCSE checkpoint -> Huggingface checkpoint for {}".format(args.path))

    state_dict = torch.load(os.path.join(args.path, "pytorch_model.bin"), map_location=torch.device("cpu"))
    new_state_dict = {}
    for key, param in state_dict.items():
        # Replace "mlp" to "pooler"
        if "mlp" in key:
            key = key.replace("mlp", "pooler")

        # Delete "bert" or "roberta" prefix
        if "bert." in key:
            key = key.replace("bert.", "")
        if "roberta." in key:
            key = key.replace("roberta.", "")

        new_state_dict[key] = param

    torch.save(new_state_dict, os.path.join(args.path, "pytorch_model.bin"))

    # Change architectures in config.json
    config = json.load(open(os.path.join(args.path, "config.json")))
    for i in range(len(config["architectures"])):
        config["architectures"][i] = config["architectures"][i].replace("ForCL", "Model")
    json.dump(config, open(os.path.join(args.path, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
