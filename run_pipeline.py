"""Run the full pipeline: generate -> debate arena -> train."""
from judge_pipeline.generate_data import generate
from judge_pipeline.debate_arena import run_arena
from judge_pipeline.train_judge import train

if __name__ == "__main__":
    generate()
    run_arena()
    train()
    print("\nDone. Serve the judge with:")
    print("  uvicorn judge_pipeline.serve:app --port 8000")
