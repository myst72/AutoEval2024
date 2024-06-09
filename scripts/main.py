import argparse
from tqdm import tqdm
from configloaders import load_config
from dataloaders import load_evaldataset
from templates import load_template
from models import load_model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run evaluation using specified config file.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable development mode")
    return parser.parse_args()

def debug_print(debug_mode, *messages):
    """Print messages only if debug_mode is True."""
    if debug_mode:
        print("🐼", *messages)


def main():
    args = parse_arguments()
    config = load_config(args.config_path)

    # Load
    dataset = load_evaldataset(config["dataset_config"])
    template = load_template(config["template_config"])
    model = load_model(config["model_config"])

    # Generate and Postprocess
    records = []
    for data in tqdm(dataset, desc="Processing Dataset"):
        record = {}
        record["prompt"] = template.create_prompt(data)
        debug_print(args.debug, "prompt:\n", record["prompt"])
        record["generate_text_list"] = model.generate(record["prompt"])
        debug_print(args.debug, "generate_text_list:\n", record["generate_text_list"][0])
        record["postprocess_text_list"] = template.postprocess(record["generate_text_list"], record)
        debug_print(args.debug, "postprocess_text_list:\n", record["postprocess_text_list"][0])
        record["reference"] = template.create_reference(data)
        debug_print(args.debug, "reference:\n", record["reference"])
        records.append(record)

    # TODO: Evaluate
    # TODO: Save

if __name__ == '__main__':
    main()


