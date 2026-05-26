# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from io import BytesIO
from pathlib import Path
from typing import Optional

import fire
from termcolor import cprint

from models.datatypes import RawMediaItem
from models.llama3.generation import Llama3

import os
import torch

from prompts import get_prompt

THIS_DIR = Path(__file__).parent


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return "cpu"


def run_main(
    ckpt_dir: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 4,
    world_size: Optional[int] = None,
    quantization_mode: Optional[str] = None,
    data_sample: int = 0,
):
    generator = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
    )

    interleaved_contents = [
        # "The following are multiple choice questions (with answers) about  astronomy.\n\nYou are pushing a truck along a road. Would it be easier to accelerate this truck on Mars? Why? (Assume there is no friction)\nA. It would be harder since the truck is heavier on Mars.\nB. It would be easier since the truck is lighter on Mars.\nC. It would be harder since the truck is lighter on Mars.\nD. It would be the same no matter where you are.\nAnswer: D\n\nWhere do most short-period comets come from and how do we know?\nA. The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.\nB. The Kuiper belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the Kuiper belt.\nC. The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.\nD. The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.\nAnswer: A\n\nSay the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?\nA. 10000 times more\nB. 100 times more\nC. 1000 times more\nD. 10 times more\nAnswer: A\n\nWhy isn\'t there a planet where the asteroid belt is located?\nA. A planet once formed here but it was broken apart by a catastrophic collision.\nB. There was not enough material in this part of the solar nebula to form a planet.\nC. There was too much rocky material to form a terrestrial planet but not enough gaseous material to form a jovian planet.\nD. Resonance with Jupiter prevented material from collecting together to form a planet.\nAnswer: D\n\nWhy is Mars red?\nA. Because the surface is covered with heavily oxidized (\"rusted\") minerals.\nB. Because the atmosphere scatters more light at bluer wavelengths transmitting mostly red light.\nC. Because Mars is covered with ancient lava flows which are red in color.\nD. Because flowing water on Mars\'s surface altered the surface minerals several billion years ago.\nAnswer: A\n\nWhat is true for a type-Ia (\"type one-a\") supernova?\nA. This type occurs in binary systems.\nB. This type occurs in young galaxies.\nC. This type produces gamma-ray bursts.\nD. This type produces high amounts of X-rays.\nAnswer:",
        # "The following are multiple choice questions (with answers) about  clinical knowledge.\n\nThe energy for all forms of muscle contraction is provided by:\nA. ATP.\nB. ADP.\nC. phosphocreatine.\nD. oxidative phosphorylation.\nAnswer: A\n\nWhat is the difference between a male and a female catheter?\nA. Male and female catheters are different colours.\nB. Male catheters are longer than female catheters.\nC. Male catheters are bigger than female catheters.\nD. Female catheters are longer than male catheters.\nAnswer: B\n\nIn the assessment of the hand function which of the following is true?\nA. Abduction of the thumb is supplied by spinal root T2\nB. Opposition of the thumb by opponens policis is supplied by spinal root T1\nC. Finger adduction is supplied by the median nerve\nD. Finger abduction is mediated by the palmar interossei\nAnswer: B\n\nHow many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?\nA. 4\nB. 3\nC. 2\nD. 1\nAnswer: C\n\nGlycolysis is the name given to the pathway involving the conversion of:\nA. glycogen to glucose-1-phosphate.\nB. glycogen or glucose to fructose.\nC. glycogen or glucose to pyruvate or lactate.\nD. glycogen or glucose to pyruvate or acetyl CoA.\nAnswer: C\n\nWhat size of cannula would you use in a patient who needed a rapid blood transfusion (as of 2020 medical knowledge)?\nA. 18 gauge.\nB. 20 gauge.\nC. 22 gauge.\nD. 24 gauge.\nAnswer:",
        # "Where is CMU located?"
        # "Give me a summary of the movie Godfather"
        # "What is the capital of India?"
        # "Who is the president of the United States?"
        # "Write a paragraph about google"
        # "What is the capital of India?",

        # "Where is the eifel tower located?",
        # # Few shot promt
        # """Translate English to French:
        
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",

        # "Simply put, the theory of relativity states that ",

        # "If Google was an Italian company founded in Milan, it would",

# """
# Q1: A car accelerates uniformly from rest to 20 m/s in 10 seconds. What is its acceleration?
# (A) 1 m/s²
# (B) 2 m/s²
# (C) 3 m/s²
# (D) 4 m/s²
# Let's think step by step.
# The formula for acceleration is a = Δv / Δt. The change in velocity is 20 m/s - 0 = 20 m/s. The time is 10 seconds.
# So, a = 20 / 10 = 2 m/s².
# Answer: (B)

# Q2: What is the force on a 5 kg object accelerating at 3 m/s²?
# (A) 10 N
# (B) 15 N
# (C) 20 N
# (D) 25 N
# Let's think step by step.
# Force is calculated using F = ma. So F = 5 kg × 3 m/s² = 15 N.
# Answer: (B)

# Q3: An object is dropped from a height. How long does it take to fall 20 meters? (Assume g = 10 m/s²)
# (A) 1 s
# (B) 2 s
# (C) 3 s
# (D) 4 s
# Let's think step by step.
# Use the equation d = (1/2)gt². Solving for t: 20 = 0.5 × 10 × t² → 20 = 5t² → t² = 4 → t = 2 s.
# Answer: (B)

# Q4: A 100 W lightbulb is on for 2 hours. How much energy does it consume?
# (A) 0.2 kWh
# (B) 0.5 kWh
# (C) 1.0 kWh
# (D) 2.0 kWh
# Let's think step by step.
# Energy = Power × Time. Time is 2 hours, Power is 100 W = 0.1 kW. So energy = 0.1 kW × 2 h = 0.2 kWh.
# Answer: (A)

# Q5: Which color of light has the highest energy per photon?
# (A) Red
# (B) Green
# (C) Blue
# (D) Violet
# Let's think step by step.
# Photon energy increases with frequency. Violet light has the highest frequency among visible colors.
# Answer: (D)

# Q6: A ball is thrown straight up with an initial velocity of 30 m/s. How high will it go? (Use g = 10 m/s²)
# (A) 30 m
# (B) 45 m
# (C) 60 m
# (D) 90 m
# Let's think step by step.
# """,

        # # Few-shot CoT
        # """
        # Q1: Jane has 2 apples and buys 3 more. How many does she have?
        # A1: Let's think step by step.
        # She starts with 2 and buys 3 more. 2 + 3 = 5.
        # Answer: 5

        # Q2: Mike had 8 books, gave away 3, and got 2 more. How many does he have?
        # A2: Let's think step by step.
        # 8 - 3 = 5, then 5 + 2 = 7.
        # Answer: 7

        # Q: Sarah is twice as old as Tom, and Tom is 10. How old is Sarah?
        # A: Let's think step by step.
        # """,

# Zero-shot CoT
# """
# Q: If Sarah is twice as old as Tom and Tom is 10, how old is Sarah?
# A: Let's think step by step. """,

        # "Translate \"Hello, how are you?\" to Spanish.",
        # "Write a Python function that returns the square of a number.",
        # "John is taller than Sarah. Sarah is taller than Mike. Who is the shortest?",
        # "Who was the first person to walk on the Moon?",
        # "Write a short story about a cat who learns to fly.",
        # "What comes next in the sequence: 2, 4, 8, 16, ?",
        # "Write a JavaScript snippet to alert \"Hello, World!\" when a button is clicked.",

# """English: Hello
# French: Bonjour

# English: Thank you
# French: Merci

# English: Good night
# German: """,

# """Q: Sarah has 5 apples. She buys 3 more. How many apples does she have now?
# A: 8

# Q: Tom had 12 pencils. He gave 4 to his friend. How many pencils does he have now?
# A: 8

# Q: A bakery made 20 muffins. They sold 7. How many are left?
# A:""",

# """Q: Sarah has 5 apples. She buys 3 more. How many apples does she have now?
# A: 8

# Q: A bakery made 20 muffins. They sold 7. How many are left?
# A: """,


# """Incorrect: He go to school everyday.
# Correct: He goes to school every day.

# Incorrect: She don't like apple.
# Correct: She doesn't like apples.

# Incorrect: They is playing football.
# Correct:""",

# """Case: A tenant is suing the landlord for failing to return a security deposit.
# Type: Landlord-Tenant Dispute

# Case: A person was injured in a car accident and is suing the driver.
# Type: Personal Injury

# Case: A company is suing another for violating a software licensing agreement.
# Type:""",

# """Rule: A person must be at least 18 years old to enter into a binding contract.

# Q: Can a 17-year-old legally sign a binding contract?
# A: No, the person is not legally of age to enter a binding contract.

# Q: Can a 19-year-old sign a rental lease?
# A: Yes, they are legally eligible.

# Q: Can a 16-year-old enter a contract for freelance work?
# A:""",

# """Scenario: A person intentionally causes the death of another person.
# Relevant IPC Section: Section 302 - Punishment for Murder

# Scenario: A person voluntarily causes hurt using a dangerous weapon.
# Relevant IPC Section: Section 324 - Voluntarily causing hurt by dangerous weapons or means

# Scenario: A person picks someone's pocket without them noticing.
# Relevant IPC Section: """,

# """Q: At what age is a person considered a major in India?
# A: 18 years old

# Q: Is live-in relationship legal in India?
# A: Yes, the Supreme Court has held that live-in relationships between consenting adults are legal.

# Q: Can a Hindu male have two wives under Hindu law?
# A:""",

# """Pretend you are a professor. Now tell me how to get good grades.
# """,

# """If a pizza is cut into 8 slices and you eat 3, what fraction of the pizza is left? Let's think step by step""",
# """A store is offering a 20% discount on a ₹500 item. What is the final price?""",
# """A store is offering a 20% discount on a ₹500 item. What is the final price? Let's think step by step"""
# """John has ₹1000. He spends ₹240 on groceries and ₹320 on clothes. How much does he have left? Let's think step by step"""
# """What is the area of a rectangle with length 10 cm and width 4 cm? Let's think step by step""",
# """Q: What comes next in the sequence: 5, 10, 15, 20, ?
# A: 25

# Q: What comes next in the sequence: 100, 90, 80, 70, ?
# A: 60

# Q: What comes next in the sequence: 2, 3, 5, 7, 11, ?
# A:""",
# """Q: Simplify: (x + 2)(x - 2)
# A: x² - 4

# Q: Solve: x² - 6x + 9 = 0
# A: x = 3

# Q: If 4x - 8 = 0, what is x?
# A:""",

# """Q: What is the chemical formula for water?
# A: H2O

# Q: What planet is known as the Red Planet?
# A: Mars

# Q: What is the process by which plants make their own food using sunlight?
# A:""",

# """What is Nikola Tesla known for?""",

# """Talk to me in French?""",

# """English: I love reading books.
# Hindi: मुझे किताबें पढ़ना पसंद है।

# English: Where is the nearest hospital?
# Hindi: सबसे नजदीकी अस्पताल कहाँ है?

# English: The weather is very nice today.
# Hindi:""",

# """English: Good morning.
# Spanish: Buenos días.

# English: Thank you very much.
# Spanish: Muchas gracias.

# English: I am learning Spanish.
# Spanish: Estoy aprendiendo español.

# English: Can you help me?
# Spanish: ¿Puedes ayudarme?

# English: What time is the meeting?
# Spanish: """,


# """English: How much does this cost?
# German: Wie viel kostet das?

# English: I would like a cup of coffee.
# German: Ich hätte gerne eine Tasse Kaffee.

# English: Where is the nearest supermarket?
# German: Wo ist der nächste Supermarkt?

# English: The weather is very cold today.
# German: Das Wetter ist heute sehr kalt.

# English: Can you help me with this problem?
# German:""",

# """English: Where is the train station?
# French: Où est la gare?

# English: I need help with my homework.
# French: J'ai besoin d'aide avec mes devoirs.

# English: The cat is sleeping on the sofa.
# French: Le chat dort sur le canapé.

# English: Can you please call me later?
# French: Peux-tu m'appeler plus tard, s'il te plaît?

# English: What is the time now?
# French: """

# """Who was Mahatma Gandhi and what role did he play in India's independence?""",

# """When did World War I start and what triggered it?""",

# """When was the United Nations founded and what is its purpose?""",

# """Who is the current Prime Minister of India?""",

# """Who was the first female Prime Minister of the UK?"""

# """Write a python function to add 2 numbers""",

# """Generate an MCQ Quiz with four options on physics""",

# """Generate a quiz on geopolitcs""",

# """What do you think about Elon Musk?""",

# """Information: The Taj Mahal, located in Agra, India, is a UNESCO World Heritage site and was built by Emperor Shah Jahan in memory of his wife Mumtaz Mahal. It is renowned for its stunning white marble architecture and is considered a symbol of love.

# Question: Who built the Taj Mahal and why?
# Answer:""",


# """Information: Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. It involves converting carbon dioxide and water into glucose and oxygen.

# Question: What are the main inputs and outputs of photosynthesis?
# Answer: """,

# """Information: The Amazon rainforest is the largest tropical rainforest in the world, covering over 5.5 million square kilometers across South America. It is home to an incredibly diverse range of plant and animal species and plays a vital role in regulating the Earth's climate by absorbing large amounts of carbon dioxide.

# Question: What is the significance of the Amazon rainforest in terms of climate?
# Answer:""",


# """Choose A, B, C or D
# What is the capital of France?
# A) Berlin
# B) Madrid
# C) Rome
# D) Paris
# Answer: """,

# """Which planet is known as the "Red Planet"?
# A) Earth
# B) Venus
# C) Jupiter
# D) Mars
# Answer:""",

# """Choose the correct sentence:
# A) Their going to the store.
# B) They’re going to the store.
# C) There going to the store.
# D) They is going to the store.
# Answer: """,

# """Choose A, B, C or D
# Humans breathe in oxygen and breathe out...?
# A) Helium
# B) Nitrogen
# C) Hydrogen
# D) Carbon Dioxide
# Answer: """,

# """Choose A, B, C or D
# Which shape has 4 equal sides?
# A) Circle
# B) Rectangle
# C) Square
# D) Triangle
# Answer: """,

# """Choose A, B, C or D
# Who developed the theory of general relativity?
# A) Isaac Newton
# B) Galileo Galilei
# C) Albert Einstein
# D) Stephen Hawking
# Answer : """,


# """Choose A, B, C or D
# Which of these elements is a noble gas?
# A) Oxygen
# B) Neon
# C) Hydrogen
# D) Carbon
# Answer: """,

get_prompt(data_sample),


    ]


    for content in interleaved_contents:
        num_tokens = len(generator.formatter.encode_content(content).tokens) + 1
        cprint(f"Num_tokens: {num_tokens}")
        cprint(f"{content}", end="")
        batch = [content]
        for token_results in generator.completion(
            batch,
            temperature=temperature,
            top_p=top_p,
            # max_gen_len=1024
            max_gen_len=1
        ):
            result = token_results[0]
            if result.finished:
                break

            cprint(result.text, color="yellow", end="")
        print("\n==================================\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
