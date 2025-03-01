import os
import nemo_run as run

if __name__ == "__main__":
    training_job = run.Script(
        inline="""
# This string will get saved to a sh file and executed with bash
# Run any preprocessing commands

# Run the training command
python train.py

# Run any post processing commands
"""
    )

    # Run it locally
    executor = run.LocalExecutor()

    with run.Experiment("nemo_2.0_training_experiment", log_level="INFO") as exp:
        exp.add(training_job, executor=executor, tail_logs=True, name="training")
        # Add more jobs as needed

        # Run the experiment
        exp.run(detach=False)