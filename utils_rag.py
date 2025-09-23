"""
This file is part of the langchain-kr project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file references code from the following source:
Link: https://github.com/teddylee777/langchain-kr

Original Author: teddylee777
Modifications:
- [2024-07-23]: Added and modified some comments for clarification and added a docstring by jonhyuk0922

"""

from langchain_core.callbacks.base import BaseCallbackHandler
import streamlit as st

class StreamHandler(BaseCallbackHandler):
    def __init__(self,container, initial_text="") -> None:
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token:str,**kwrgs) -> None:
        self.text += token
        self.container.markdown(self.text)

def init_conversation() -> None:
    """session_state messages 초기화
    
    session state 에 데이터 저장 (캐싱)
    왜? Streamlit 은 매번 초기화가 되기때문에 저장해둬야함
    """
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

def print_conversation() -> None:
    """이전 대화를 표시해주는 함수"""

    # 만약 messages 에 기록중인 대화가 있다면 출력해주는 코드
    # 🔹 수정: messages가 딕셔너리이므로 그에 맞게 수정
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for message in st.session_state["messages"]:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
