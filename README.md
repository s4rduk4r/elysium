# ELYSIUM
`Elysium` is a simple agent to play through the combat situations in `Trails through Daybreak 2`. 
If you want to try it, then you need to have Steam copy of the game. Alternatively, you may modify the code to use one of the many screenshots taken to debug various scenarious. 
Mind you that not all of them were used, so some things are expected not to work at all, or not to work properly.
This project was mainly to learn how the `langgraph` can be used in building agentic systems.

# PREREQUISITES
1. Windows 10
2. Trails through Daybreak 2 (Steam version)
3. Latest [llama.cpp](https://github.com/ggml-org/llama.cpp/releases). I've used `b5425-vulkan`
4. Graphviz. You may turn it off by commenting line in [agent.py](https://github.com/s4rduk4r/elysium/blob/04cd67b575d4ca106654d6facbd33f21a6971ac8/agent.py#L49)
5. [unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit)
6. [Qwen3-30B-A3B-Q4_1](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/blob/main/Qwen3-30B-A3B-Q4_1.gguf)

# INSTALLATION
1. Create new conda environment with Python 3.12
2. `pip install uv`
3. Install latest `pytorch`
4. `uv pip install -r requirements.txt`

After installation you probably need to modify the paths to the models, and `llama-server.exe` executable in files [agent.py](https://github.com/s4rduk4r/elysium/blob/04cd67b575d4ca106654d6facbd33f21a6971ac8/agent.py#L153) 
and [nodes/vlm_wrapper.py](https://github.com/s4rduk4r/elysium/blob/04cd67b575d4ca106654d6facbd33f21a6971ac8/nodes/vlm_wrapper.py#L17)

# USAGE
1. Start agent via `python agent.py`
2. Load save file (you have roughly 40 seconds for that)
3. Wait for agent to profile first 4 characters - their stats are being used to make a decision by LLM reasoner
4. Engage some enemies in combat
5. Observe agent's behaviour

# AGENT'S GRAPH
![](https://raw.githubusercontent.com/s4rduk4r/elysium/main/agent_graph.png)
