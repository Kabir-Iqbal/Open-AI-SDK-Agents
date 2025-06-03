import chainlit as cl
import os
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv



load_dotenv()  # load .venv files


gemini_api_key = os.getenv("GEMINI_API_KEY")

# step 1: provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# step 2: Provide the model 
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=provider
)

# step 3: Define the Run configuration
run_config = RunConfig(
    model = model,
    model_provider = provider,        
    tracing_disabled = True      # tracing_disables is used to disable tracing for the agent
)




# Define the Agent
agent = Agent(
    name="Language Translator Agent",
    instructions="""You are Language translator Agent , You can just translate urdu langauge into Englis, 
    or English to Urdu. If you get another query you said Iam a just language translator agent Urdu to English """,
)


# Agent with chainlit

# Define the start message for the agent
@cl.on_chat_start
async def handle_chatt_start():
    cl.user_session.set("history", [])

    await cl.Message(
        content="Welcome to the Language Translator Agent! You can ask me to translate text between Urdu and English."
    ).send()


# Define the message handler for the agent
@cl.on_message
async def handle_message(message : cl.Message):
    
    msg = cl.Message(content='')
    await msg.send()


    # get history from user session
    history = cl.user_session.get("history")
    # append user message to history
    history.append({"role": "user", "content": message.content})
    # get response from agent

    result =  Runner.run_streamed(
        agent,
        run_config=run_config,
        input=history,
    )

   # Send agent response to user with stream
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
            

    # append agent response to history
    history.append({"role" : "assistant", "content" : result.final_output})
    # set history to user session
    cl.user_session.set("history", history)




