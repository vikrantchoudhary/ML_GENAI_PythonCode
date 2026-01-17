import SimpleAgent
# setup
prompt = "you run in a loop of thought, Action, PAUSE, Observarion. Answer the user's question."
agent = SimpleAgent.SimpleAgent(prompt, {"calculator" : SimpleAgent.calculator})
agent("what is 21 times 2 ?")