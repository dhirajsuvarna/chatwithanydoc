import chainlit as cl 


@cl.on_message
def main(message: str):

    cl.Message(content=f'Received: {message}').send()