This is the code base for our final project for CSC2542: Knowledge Representation and Modern AI. We develop a symbolic epistemic reasoning system based on proper doxastic knowledge bases that can integrate with large language models. We find that this system improves LLM accuracy by up to 20\% on epistemic reasoning tasks.

Our reasoning system utilizes the RP-MEP epistemic planner built by Muise et al in the paper "Planning over multi-agent epistemic states: a classical planning approach"  (https://dl.acm.org/doi/10.5555/2888116.2888179). See https://github.com/QuMuLab/pdkb-planning for the PDKB python module we adapt. 

The LLMs used for this study include LlaMa-2, Llemma, found in Hugging Face's Transformers library, and OpenAI's GPT-3.5-Turbo, and GPT-4. 
