#Agent components architecture
# The Brain (LLM) : the reasoning engine that decide what to do 
# Planning : the agent break down a complex goal into smaller steps (chain-of-thought)
# Memory: Short-term (conversational history) and long-term (vector Db/RAG)
# Tool (Action space) : External functions the agent can call (like search,calculator,db)
import re
class SimpleAgent:
    def __init__(self,system_prompt,tools):
        self.system = system_prompt
        self.tools = tools
        self.messages = [{"role":"system", "content": system_prompt}]
    
    def __call__(self, message):
        self.messages.append({"role":"user","content":message})
        return self.execute()
    
    def execute(self,max_turns=5):
        #the core agent loop
        for i in range(max_turns):
            # Ask the brain LLM what to do , generally llm_call would be API request to OpenAI/LLM
            response = self.llm_call(self.messages)
            print(f"\n[AI] : {response}")

            #check if the AI wants to take an action , action: tool_name: input_string
            action_match = re.search(r"Action : ([\w_]+): (.*)", response)

            if action_match:
                tool_name, tool_input = action_match.groups()
                if tool_name in self.tools:
                    print(f"---Running tool {tool_name} with input; {tool_input}")
                    observation = self.tools[tool_name](tool_input)

                    # feed the result back to LLM (observations)
                    self.messages.append({"role":"assistant","content": response})
                    self.messages.append({"role":"user","content":f"Observation : {observation}"})
                    continue
                else:
                    # No action found
                    return response
    
    def llm_call(self,messages):
        # Mocking an LLM response for demostration 
        last_msg = messages[-1]["content"]
        print(last_msg)
        if "Observation" in last_msg:
            return "The final answer is 42"
        return "Though: I need to calculate this. Action: calculator : 21 * 2"
    
def calculator(q):
    return eval(q)
    
   
