# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from unittest.mock import patch

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import functions
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_parallel_confirmation_cancels_unfinished_sibling():
  agent = Agent(name='root_agent')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  confirming_call = types.FunctionCall(
      id='confirming-call', name='confirming_tool', args={}
  )
  blocked_call = types.FunctionCall(
      id='blocked-call', name='blocked_tool', args={}
  )
  blocked_started = asyncio.Event()
  blocked_cancelled = asyncio.Event()

  async def execute_call(
      _invocation_context,
      function_call,
      _tools_dict,
      _agent,
      _tool_confirmation,
  ):
    if function_call.id == confirming_call.id:
      await blocked_started.wait()
      return Event(
          author='root_agent',
          actions=EventActions(
              requested_tool_confirmations={
                  confirming_call.id: ToolConfirmation(
                      confirmed=False, hint='Approve'
                  )
              }
          ),
      )
    blocked_started.set()
    try:
      await asyncio.Event().wait()
    except asyncio.CancelledError:
      blocked_cancelled.set()
      raise

  with patch.object(
      functions,
      '_execute_single_function_call_async',
      side_effect=execute_call,
  ):
    response = await asyncio.wait_for(
        functions.handle_function_call_list_async(
            invocation_context,
            [confirming_call, blocked_call],
            {},
        ),
        timeout=0.1,
    )

  assert response.actions.requested_tool_confirmations == {
      confirming_call.id: ToolConfirmation(confirmed=False, hint='Approve')
  }
  assert blocked_cancelled.is_set()


@pytest.mark.asyncio
async def test_parallel_function_calls_with_state_change():
  function_calls = [
      types.Part.from_function_call(
          name='update_session_state',
          args={'key': 'test_key1', 'value': 'test_value1'},
      ),
      types.Part.from_function_call(
          name='update_session_state',
          args={'key': 'test_key2', 'value': 'test_value2'},
      ),
      types.Part.from_function_call(
          name='transfer_to_agent', args={'agent_name': 'test_sub_agent'}
      ),
  ]
  function_responses = [
      types.Part.from_function_response(
          name='update_session_state', response={'result': None}
      ),
      types.Part.from_function_response(
          name='update_session_state', response={'result': None}
      ),
      types.Part.from_function_response(
          name='transfer_to_agent', response={'result': None}
      ),
  ]

  responses: list[types.Content] = [
      function_calls,
      'response1',
  ]
  function_called = 0
  mock_model = testing_utils.MockModel.create(responses=responses)

  async def update_session_state(
      key: str, value: str, tool_context: ToolContext
  ) -> None:
    nonlocal function_called
    function_called += 1
    tool_context.state.update({key: value})
    return

  async def transfer_to_agent(
      agent_name: str, tool_context: ToolContext
  ) -> None:
    nonlocal function_called
    function_called += 1
    tool_context.actions.transfer_to_agent = agent_name
    return

  test_sub_agent = Agent(
      name='test_sub_agent',
  )

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[update_session_state, transfer_to_agent],
      sub_agents=[test_sub_agent],
  )
  runner = testing_utils.TestInMemoryRunner(agent)
  events = await runner.run_async_with_new_session('test')

  # Notice that the following assertion only checks the "contents" part of the events.
  # The "actions" part will be checked later.
  assert testing_utils.simplify_events(events) == [
      ('root_agent', function_calls),
      ('root_agent', function_responses),
      ('test_sub_agent', 'response1'),
  ]

  # Asserts the function calls.
  assert function_called == 3

  # Asserts the actions in response event.
  response_event = events[1]

  assert response_event.actions == EventActions(
      state_delta={
          'test_key1': 'test_value1',
          'test_key2': 'test_value2',
      },
      transfer_to_agent='test_sub_agent',
  )
