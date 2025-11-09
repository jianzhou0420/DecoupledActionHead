from pathlib import Path
import openai

# Make sure you have set your API key:
# export OPENAI_API_KEY="your_api_key_here"


parts = [
    "Welcome to the demonstration video for ICRA paper Decoupled Action Head: Confining Task knowledge to conditioning layers.",
    "This work related to Behavior Cloning, a data-driven supervised learning approach that has gained much attention for its scaling potential.",
    "And, Diffusion Policy is one of the most influential methods. it was the first to show that predicting continuous action sequences can be highly effective.",

    "However, there are two major problems in this field.",
    "First, the extreme scarcity of data remains a crucial bottleneck. Observation-action paired data is expensive to collect.",
    "Second, the design of neural architectures for manipulation has received little attention.",
    "For example, DP-CNN uses a backbone with over 244 million parameters, which is clearly inefficient and many VLA models directly adapt models from image generation field as action experts, leading to unnecessary large action expert networks.",

    "To address these issues, we propose the Decoupled Action Head training recipe.",
    "The key idea is to pre-train the action head using observation-free, pure action data.",
    "This not only allows us to take advantage of large-scale trajectory data, but also reveals a knowledge confinement property that provides new insights into neural network design.",

    "In detail, the Decoupled Action Head can be understood as a two-stage training paradigm.",
    "Stage one: we train the action generation backbone—the action head—using forward kinematics.",
    "Stage two: we integrate this trained action head back into the diffusion policy, freeze it, and then train only the conditioning modules: the observation encoder and feature modulation projectors.",

    "Our work leads to three main findings.",
    "We first validate the feasibility of the decoupled action head training recipe, proving that an action head pretrained from joint position and end effector pose pairs can be effectively adapted to task-specific training.",
    "We have checked the in-distribution feasibility, out-of-distribution feasibility, and multi-task feasibility on diffusion policy CNN.",

    "Secondly, we notice the performance drop on DP-Transformer, and design a DP-Transformer-FiLM to prove that feature modulation like methods are crucial for decoupled training.",
    "Finally, since task knowledge is confined to conditioning layers, the action head can be simplified.",
    "We propose DP-MLP, a lightweight 4-million-parameter model that runs about 80 percent faster while maintaining similar performance.",
    "All of the experiments are conducted on eight selected tasks with three seeds.",

    "In summary, the Decoupled Action Head provides a new training recipe that leverages observation-free data, improves efficiency, and hints on where to scale for general manipulation policy."
]
# make them together

parts = [" ".join(parts)]

for i, segment in enumerate(parts):
    out = Path(f"video/part_{i}.mp3")
    with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="echo",
        input=segment
    ) as response:
        response.stream_to_file(out)
