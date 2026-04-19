#!/usr/bin/env python
from pathlib import Path

from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from pydub import AudioSegment
from crews.meeting_minutes_crew.meeting_minutes_crew import MeetingMinutesCrew
from pathlib import Path

load_dotenv()

class MeetingMinutesState(BaseModel):
    transcript: str = ""
    meeting_minutes: str = ""

class MeetingMinutesFlow(Flow[MeetingMinutesState]):

    @start()
    def transcribe_meeting(self):
        print("Generating Transcription...")

        # current file directory
        BASE_DIR = Path(__file__).resolve().parent
        # go 2 directories up
        audio_path = BASE_DIR.parent.parent / "sample_meeting_audio/F_0987_12y7m_1.wav"

        #load model
        model = WhisperModel("base")

        segments, _ = model.transcribe(audio_path)

        transcription = ""
        for segment in segments:
            transcription += segment.text

        self.state.transcript += transcription
        print(transcription)

    @listen(transcribe_meeting)
    def generate_meeting_minutes(self):
        print("Generating Meeting Minutes...")

        crew = MeetingMinutesCrew()
        meeting_minutes = crew.crew().kickoff(inputs={"transcript": self.state.transcript})
        self.state.meeting_minutes = meeting_minutes.raw

def kickoff():
    meeting_minutes_flow = MeetingMinutesFlow()
    meeting_minutes_flow.kickoff()

if __name__ == "__main__":
    kickoff()
