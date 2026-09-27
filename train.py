import argparse

from training.pipeline import train_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/detector.joblib")
    parser.add_argument("--public-augmentation", type=int, default=1200)
    parser.add_argument("--extra-csv", action="append", default=[])
    parser.add_argument("--external-test-csv", action="append", default=[])
    parser.add_argument("--stack-folds", type=int, default=4)
    parser.add_argument("--target-fpr", type=float, default=0.02)
    parser.add_argument("--target-ai-miss", type=float, default=0.02)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    train_detector(
        output=arguments.output,
        public_augmentation=arguments.public_augmentation,
        extra_csv=arguments.extra_csv,
        external_test_csv=arguments.external_test_csv,
        stack_folds=arguments.stack_folds,
        target_false_positive_rate=arguments.target_fpr,
        target_ai_missed_as_human_rate=arguments.target_ai_miss,
    )


if __name__ == "__main__":
    main()