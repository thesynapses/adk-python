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
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows import functions
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from ... import testing_utils


def function_call(function_call_id, name, args: dict[str, Any]) -> types.Part:
  part = types.Part.from_function_call(name=name, args=args)
  part.function_call.id = function_call_id
  return part


@pytest.mark.asyncio
async def test_parallel_function_call_error_preserves_completed_sibling():
  id_1 = 'id_1'
  id_2 = 'id_2'
  responses = [
      [
          function_call(id_1, 'fail_tool', {}),
          function_call(id_2, 'sleep_tool', {}),
      ],
      [
          types.Part.from_text(text='final response'),
      ],
  ]

  mock_model = testing_utils.MockModel.create(responses=responses)

  fail_called = False
  sleep_started = False
  sleep_completed = False
  sleep_cancelled = False

  async def fail_tool(tool_context: ToolContext) -> str:
    nonlocal fail_called
    fail_called = True
    raise ValueError('Tool failed intentionally')

  async def sleep_tool(tool_context: ToolContext) -> str:
    nonlocal sleep_started, sleep_completed, sleep_cancelled
    sleep_started = True
    try:
      await asyncio.sleep(0)
      sleep_completed = True
      return 'Tool succeeded'
    except asyncio.CancelledError:
      sleep_cancelled = True
      raise

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[fail_tool, sleep_tool],
  )

  runner = testing_utils.InMemoryRunner(agent)

  with pytest.raises(ValueError, match='Tool failed intentionally'):
    await runner.run_async(
        new_message=types.Content(parts=[types.Part(text='test')]),
    )

  assert fail_called
  assert sleep_started
  assert sleep_completed
  assert not sleep_cancelled
  session = runner.session
  response_ids = {
      response.id
      for event in session.events
      for response in event.get_function_responses()
  }
  assert response_ids == {id_2}
