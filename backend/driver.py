from pygpt4all.models.gpt4all_j import GPT4All_J

def new_text_callback(text):
    print(text, end="")

prompt = """The prompt below contains three sources as context and a question to be answered using the context. Write a response to the question using the context.
### Prompt:
Context:
- A short-term holding area is an area containing a holding cell or room, including associated rooms or spaces, where occupants are restrained or detained by security measures not under their control for less than 24 hours.

- Short-term holding areas are permitted to comply with Section 431.

- A short-term holding area is a space that is classified as the main occupancy, provided that certain conditions are met. These conditions include provisions for the release of all restrained or detained occupants, an aggregate area of the short-term holding area that does not exceed 10 percent of the building area, a restrained or detained occupant load of each short-term holding area that does not exceed 20, an aggregate restrained or detained occupant load in short-term holding areas per building that does not exceed 80, compliance with Sections 408.3.7, 408.3.8, 408.4, and 408.7 as applicable for Group I-3 occupancies, requirements of the main occupancy in which short-term holding areas are located, a fire alarm system and automatic smoke detection system complying with Section 907.2.6.3 as applicable to I-3 occupancies, and a separation from other short-term holding areas and adjacent spaces by smoke partitions complying with Section 710.

Question:
What is a short-term holding area?
### Response:"""

model = GPT4All_J("./model/ggml-gpt4all-j-v1.3-groovy.bin")
model.generate(
    prompt, n_predict=100, new_text_callback=new_text_callback
)
