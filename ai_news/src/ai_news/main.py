#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crew import AiNews

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    # inputs = {
    #     'topic': 'Tourism in Rishikesh',
    #     "date": datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # }

    input_array = [
        {
            'topic': 'AI agents',
            "date": datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        },
        {
            'topic': 'Open AI',
            "date": datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        },
        {
            'topic': 'Hugging Face',
            "date": datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        }
    ]

    try:
        # AiNews().crew().kickoff(inputs=inputs)
        AiNews().crew().kickoff_for_each(inputs=input_array)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

run()