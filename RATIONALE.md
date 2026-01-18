# Repo Rationale

This md will talk through the rationale of this repository.
Mostly because I keep forgetting, over the course of the ~15 years I've been working on it, why the hell I do half the things in this repo.

## Folder Structure
Scripts should contain "one-click reproductions" of all published figures.
This is where you should start if you're interested in tracing back results you may have seen out-there.

Notebooks should provide what the scripts do, with the added benefit of *clean presentation*.
Ideally, these notebooks would just call the scripts specifically, and there'd be a 1-1 between Notebooks and Scripts, but that's ambitious.

## AI
I want a clean separation between the human work and the AI work.

My current thinking is that LLM-AI layer will operate mainly on the *Notebooks* $\pm$ *Scripts* items in order to package draft LaTeX templates.
That is, any LLM-AI inclusion in this repo is just to help address the perfectionist in me and output the most bland, matter-of-fact communication possible for those that constantly demand that.
I have no interest in learning how to write in a dry way, and I think LLM-AI can be useful in that last layer.