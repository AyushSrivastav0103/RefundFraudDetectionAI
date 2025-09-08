import os
import sys


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "ui":
        os.system("streamlit run RefundFraudDetectionAI/ui/app.py")
    elif cmd == "train":
        from RefundFraudDetectionAI.pipeline.train import train_pipeline
        train_pipeline()
    elif cmd == "predict":
        from RefundFraudDetectionAI.pipeline.predict import FraudPipeline
        pipeline = FraudPipeline()
        result = pipeline.run(
            "I never received my package, but tracking shows it was delivered.",
            "Past 3 refund requests in last 2 months for similar reasons.",
        )
        print(result)
    else:
        print("Usage:")
        print("  python -m RefundFraudDetectionAI.main ui")
        print("  python -m RefundFraudDetectionAI.main train")
        print("  python -m RefundFraudDetectionAI.main predict")


if __name__ == "__main__":
    main()

