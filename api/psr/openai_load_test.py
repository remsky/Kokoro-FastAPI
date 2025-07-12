import itertools
import logging
import os
import random
from time import strftime

from locust import HttpUser, task, events
from locust.runners import MasterRunner
from loguru import logger
from atomicx import AtomicInt

TEXT_SAMPLES = [
    {"audio_duration": 24, "text": "Andy is from Canada, but with his long blond hair, incredibly mellow demeanor, and wardrobe of faded T-shirts, shorts, and sandals, you would think he grew up surfing on the coast of Malibu. Having worked with Endel Tulving as an undergraduate, Andy reasoned that episodic memory gives us access to a tangible memory from a specific place and time, leading us to feel confident that we are reliving a genuine moment from the past."},
    {"audio_duration": 13, "text": "In contrast, familiarity can be strong, such as the sense of certainty that you have seen something or someone before, or it can be weak, such as a hunch or educated guess; either way, it doesn’t give us anything specific to hold on to."},
    {"audio_duration": 10, "text": "Andy was presenting his research, and I might have been a bit too direct when I voiced my skepticism that familiarity was anything more than weak episodic memory."},
    {"audio_duration": 25, "text": "I anticipated some blowback, but instead of getting defensive, Andy completely disarmed me by responding, “You should be skeptical!” After an afternoon of spirited debate, we decided to team up on an experiment to test the idea that we could identify a brain area responsible for the sense of familiarity. To sweeten the pot, we decided to make a “beer bet”—if his prediction turned out to be right, I’d buy him a beer and vice versa."},
    {"audio_duration":  8, "text": "For instance, if a subject imagined what it would be like to stuff an armadillo in a shoebox, that would create a distinctive episodic memory."},
    {"audio_duration":  5, "text": "(At this point, I’ve lost so many beer bets to Andy, it will take years to pay off my tab.)"},
    {"audio_duration": 23, "text": "We didn’t yet have a working MRI scanner in the lab at Berkeley, but Mark finessed a space for us at a clinical MRI facility in the Martinez VA Medical Center, about halfway between Berkeley and Davis. The Martinez scanner was set up for routine clinical scans and didn’t have state-of-the-art performance specs, so I had to MacGyver it, tweaking our procedures every way I could think of to turn this Ford Pinto into a Ferrari."},
    {"audio_duration":  1, "text": "“Can it fit in a shoebox?”)."},
    {"audio_duration":  9, "text": "Sure enough, activity in the hippocampus spiked when people saw a word and formed a memory that later helped them recall something about 109the context."},
    {"audio_duration": 20, "text": "This argument was based on the fact that animals with damage to the hippocampus, as well as the patients with developmental amnesia such as those studied by Faraneh Vargha-Khadem, seemed to do fine on “recognition memory” tests that required them to tell the difference between objects they had seen before and ones that were new."},
    {"audio_duration":  6, "text": "Rats, like human infants, tend to be more interested in exploring things that are new to them than in things they have seen before."},
    {"audio_duration": 28, "text": "安迪来自加拿大，但凭借他那头长长的金发、非常温和的举止和褪色的T恤、短裤和凉鞋的衣橱，你会认为他是在马里布海岸冲浪长大的。作为本科生与恩德尔·图尔文合作后，安迪推测情景记忆让我们能够访问来自特定地点和时间的具体记忆，使我们感到自信，仿佛正在重温过去的真实时刻。"},
]

VOICES: list[str] = [
    "bf_emma",
    "af_alloy",
    "bf_emma(1)+af_alloy(1)",    
]

HOST = os.environ.get("HOST", "http://localhost:8880")
URL_PATH = os.environ.get("URL_PATH", "/v1/audio/speech")
OUTPUT_FORMAT = os.environ.get("KOKORO_OUTPUT_FORMAT", "opus")

user_id = itertools.count()
concurrency = AtomicInt()

@events.test_start.add_listener
def on_test_start(environment):
    if isinstance(environment.runner, MasterRunner):
        logger.info("A new test is starting.")

@events.test_stop.add_listener
def on_test_stop(environment):
    if isinstance(environment.runner, MasterRunner):
        logger.info(f"The test stopped.")

class KokoroOpenAPiClient(HttpUser):
    def __init__(self, environment):
        super().__init__(environment)
        self.voice: str = random.choice(VOICES)
        self.audio_duration = 0
        self.id = next(user_id)
        logging.info(f"User {self.id} initialized.")

    def wait_time(self):
        return self.audio_duration

    @task
    def call_rest(self):
        current_concurrency = concurrency.inc()
        try:
            sample = random.choice(TEXT_SAMPLES)
            text = sample.get("text")
            self.audio_duration = sample.get("audio_duration")
    
            logging.info(f"{strftime('%X')} User {self.id} is executing, next execution in {self.audio_duration} sec, concurrency: {current_concurrency + 1}.")
    
            headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
            }
    
            body = {
                "input": text, 
                "voice": self.voice, 
                "response_format": OUTPUT_FORMAT, 
                "stream": True, 
                "speed": 1
            }
    
            self.client.post(URL_PATH, headers=headers, json=body, timeout=4)
        finally:
            events.request.fire(
                request_type="Concurrency",
                name="Concurrency metric",
                response_time=concurrency.dec(),
                response_length=0)
