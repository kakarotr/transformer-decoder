import platform
from argparse import ArgumentParser, BooleanOptionalAction

from train.base_model import TrainingArguments


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="训练脚本参数")

    for name, field in TrainingArguments.model_fields.items():
        arg_name = f"--{name}"
        arg_type = field.annotation
        default = field.default
        help_text = field.description or ""

        if arg_type is bool:
            parser.add_argument(
                arg_name,
                action=BooleanOptionalAction,
                default=default,
                help=f"{help_text} (default: {default})",
            )
        else:
            parser.add_argument(
                arg_name,
                type=arg_type,  # type: ignore
                default=default,
                help=f"{help_text} (default: {default})",
            )

    return parser


def parse_args() -> TrainingArguments:
    parser = build_parser()
    namespace = parser.parse_args()
    return TrainingArguments(**vars(namespace))


def get_system():
    os_name = platform.system()
    
