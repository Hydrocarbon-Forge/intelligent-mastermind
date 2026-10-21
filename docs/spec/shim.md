# API shim: endpoints, streaming, limits (spec excerpt)

Source: tdd.pdf, extracted as text (tables are column-aligned).

```text
  API shim
  The shim makes the human program look like any OpenAI-compatible server to clients. It
  owns request parsing, response framing and the connection with the client, and it makes
  no modeling decisions.

  Endpoints

     Endpoint                                          Behavior

      GET /v1/models                                   Returns one model entry whose id comes from
                                                       config (default human )

      POST /v1/chat/completions                        Creates a request, queues it for the UI, and
                                                       streams or returns the committed text

      GET /health                                      Reports process, database, backend and capability
                                                       status for the UI

     Endpoint                                          Behavior

      GET / and /ws                                    The built UI and its WebSocket

  Request handling
   1. Validate the body with pydantic. Accept message content as a string or as a list of
      parts. Non-text parts become a visible placeholder in the UI.
   2. Reject n greater than 1 with a 400, since one person gives one answer.
   3. Record temperature , top_p , seed and similar fields in the log, and ignore them.
   4. Create the request row, enqueue it, and wait on its completion signal while the person
      works.
   5. Return a streaming or a regular response built from the committed text. Responses
      contain content only, with no reasoning fields.

  Streaming format
  Streaming is the mode in which the connection can show signs of life while a person is
  typing. Each committed token is appended to the stream when its decision is flushed (see
  The gate and calibration).

     data:
     {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1760000000,
                   "model":"human","choices":[{"index":0,"delta":
     {"role":"assistant","content":""},"finish_reason":null}]}

     data: {... "choices":[{"index":0,"delta":
     {"content":"Sure"},"finish_reason":null}]}

     : keepalive

     data: {... "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

     data: [DONE]

        The role chunk is sent as soon as the request is accepted.
        A comment line ( : keepalive ) is sent every heartbeat_seconds . Comment lines are
        part of the SSE format and are ignored by compliant parsers.

        Non-streaming requests cannot send keepalives, so a slow person can hit a client's
        read timeout. The shim logs this when it happens.

  Finishing and limits
         finish_reason is stop when you send, and length when the committed token
        count reaches max_tokens (or max_completion_tokens ). The UI shows the remaining
        budget.
        When the request has stop sequences, the response finishes as soon as the
        committed text ends with one, and the sequence is trimmed from the output.
         usage is computed with the local tokenizer: prompt tokens from the rendered prompt,
        completion tokens from the committed ids. It is included when
        stream_options.include_usage is set.

  Disconnects, concurrency and tools
        The shim polls for client disconnect while waiting. On disconnect the request is
        marked cancelled , the UI is told, and the partial text stays in the log but not in training
        labels.
        The queue holds several requests. The UI works one at a time and shows the others
        with their age. When max_pending is exceeded, new requests get a 503.
        The server binds to 127.0.0.1 by default, with an optional bearer token.
        Tool definitions in a request are shown in the UI for reference. The response is always
        text, and tool_calls are never produced.

  What the backend sees
  The backend never receives the client's request as is. The prompt comes from
  render_chat , which uses the client's messages plus any retrieved knowledge. The UI
  always shows the original messages.

  Tests

     Test                                              Asserts

      test_models_endpoint                             /v1/models returns the configured id in the
                                                       OpenAI list format

     Test                                              Asserts

      test_stream_golden                               The SSE output for a scripted reply matches a
                                                       golden file, including the role chunk, [DONE] and
                                                       keepalive lines

      test_sdk_streaming                               The official OpenAI Python client reads a streamed
                                                       reply to completion

      test_sdk_non_streaming                           The same client gets a complete message with
                                                       finish_reason and usage

      test_n_greater_than_one                          n of 2 returns a 400 in OpenAI error format

      test_params_ignored_but_logged                   temperature and top_p change nothing and
                                                       appear in params_json

      test_stop_sequence_trim                          A stop sequence ends the response and is
                                                       trimmed

      test_max_tokens_length                           Reaching max_tokens finishes with length

      test_disconnect_cancels                          A client disconnect marks the request cancelled
                                                       and notifies the UI

      test_max_pending_503                             Requests beyond max_pending get a 503

      test_tools_shown_not_called                      A request with tools reaches the UI with the
                                                       definitions and returns text only

      test_no_reasoning_fields                         Responses never contain reasoning_content

      test_bearer_token                                With a token configured, requests without it get a
                                                       401

```
