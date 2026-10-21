# Request lifecycle, WebSocket protocol, key bindings, UI components (spec excerpt)

Source: tdd.pdf, extracted as text (tables are column-aligned).

```text
  Request lifecycle and UI

  request lifecycle · 5 lanes, 11 messages

  Messages 3 to 8 repeat for every token position. When the gate auto-selects, messages 5
  and 6 are skipped and the driver goes straight on to the next position. Message 7 only
  happens when your next decision flushes the provisional buffer (see The gate and
  calibration).

  Request states
  A request moves pending , then active , then done , cancelled or error . Within an
  active request each position is either asked (waiting for you) or committed . The driver
  keeps exactly one asked position at a time, and every UI message names the position it
  answers, so a late or duplicate message is ignored.

  WebSocket messages
  All messages are JSON objects with a type field and a request_id .

     Direction             Type                        Payload                         When

     Server to              request_new                Messages, parameters,           A request is queued
     UI                                                tools, age

     Server to              request_active             Prompt tail, max_tokens ,       The request reaches the
     UI                                                retrieved knowledge             front

     Server to              ask                        pos_id , idx , committed        The gate decided to ask
     UI                                                text, space_default ,
                                                       candidates ( id , display
                                                       text, p_you , p_base or
                                                       null, class), other mass,
                                                       explore flag

     Server to              auto_run                   Provisional tokens with         Tokens were auto-selected
     UI                                                display text and confidence

     Server to              rewound                    to_idx                          A rewind was applied
     UI

     Server to              request_done               finish_reason                   The request finished or was
     UI                                                                                cancelled

     Server to              status                     Backend health,                 On change and on a timer
     UI                                                capabilities, scorer version,
                                                       autonomy setting, running
                                                       metrics, knowledge count

     UI to                  sync                       n/a                             The socket connected or
     server                                                                            reconnected. The server
                                                                                       replays request_new ,
                                                                                        request_active and the
                                                                                       current ask

     UI to                  pick                       pos_id , token_id               You chose a listed token
     server

     UI to                  typed                      pos_id , text,                  You finished a word, or a
     server                                            leading_space , final           preview while typing when
                                                                                       final is false

     UI to                  rewind                     to_idx                          You stopped an auto-run
     server

     Direction             Type                        Payload                          When

     UI to                  send                       n/a                              You end the message
     server

     UI to                  decline                    Reason                           You refuse the request, and
     server                                                                             the client gets an error

     UI to                  set_autonomy               delta                            The slider moved
     server

     UI to                  teach                      Text                             You added a note to the
     server                                                                             knowledge store

     UI to                  forget                     Chunk id                         You removed a note
     server

  Candidate lists are sent as ids plus display strings, so the UI never tokenizes anything. The
  one thing it computes itself is the prefix filter as you type, using the display strings it
  already has.

  Key bindings
  The type box always has focus, so typing needs no extra step.

     Key                                                      Action

     Letters and other characters                             Go into the type box

     Space or punctuation                                     Commits the typed word. A typed space is held as
                                                              pending

     Enter                                                    Commits the typed word and a newline

     Ctrl+Enter ( ui.send_key )                               Sends the message

     Alt+1 to Alt+8                                           Picks the candidate in that row

     Up, Down, then Tab                                       Moves the highlight and picks the highlighted
                                                              candidate

     Backspace over ␣                                         Sets leading_space to false for the word being
                                                              typed

     Key                                                 Action

     Backspace with an empty type box                    Removes the last provisional token and asks at that
                                                         position

     Esc                                                 Removes all provisional tokens and asks at your
                                                         last decision

  UI components
  The UI is one React app. A Zustand store is updated only from server messages, because
  the server is the authority. Local state is limited to the type box text and the list highlight.

     Component                            Shows

      RequestList                         Queued requests with their age, and a decline
                                          button

      ConversationView                    The client's original messages and any tool
                                          definitions

      ResponseView                        Committed text, with provisional tokens dimmed
                                          until they are flushed

      CandidateList                       Up to max_rows candidates with ␣ markers and
                                          p_you bars, plus the other mass

      TypeBox                             Your typed word, with the leading-space marker

      StatusBar                           Backend health, scorer version, the autonomy
                                          slider and running metrics

      TeachPanel                          The "Teach it this" box and the stored notes with a
                                          forget button

  Tests
  UI tests use Vitest, Testing Library and jsdom with a mocked WebSocket. Server-side
  protocol tests use the scripted person.

     Test                                              Asserts

      test_candidate_list_rows                         At most max_rows rows render, with ␣ markers,
                                                       p_you bars and the other mass

      test_type_filters_candidates                     Typing narrows the rows by prefix

      test_leading_space_toggle                        The marker follows space_default , and
                                                       backspace over it sends leading_space false

      test_space_commits_word                          A space after typed text sends typed with final
                                                       true and keeps the space pending

      test_enter_commits_newline                       Enter commits the word and a newline

      test_alt_digit_picks                             Alt+2 sends pick for row 2, and a plain digit goes
                                                       into the box

      test_arrow_tab_picks                             Up, Down and Tab pick the highlighted row

      test_send_key                                    The configured send key sends send , and Enter
                                                       alone does not

      test_provisional_display                         Auto-selected tokens appear dimmed and normal
                                                       once flushed

      test_backspace_rewinds_last                      Backspace on an empty box sends rewind to the
                                                       previous index

      test_esc_rewinds_all                             Esc sends rewind to the index of your last
                                                       decision

      test_stale_ask_ignored                           An ask with an older pos_id does not replace the
                                                       current one

      test_reconnect_sync                              After a dropped socket the UI reconnects, sends
                                                       sync and restores the view

      test_slider_sends_autonomy                       Moving the slider sends set_autonomy

      test_teach_and_forget                            Adding and removing a note sends teach and
                                                       forget

      test_ws_happy_path                               The scripted person completes a request through
                                                       ask, pick, typed and send

      test_ws_stale_pos_ignored                        A late pick for an old pos_id changes nothing

     Test                                                Asserts

      test_ws_decline_errors_client                       decline returns an error response to the API
                                                         client

      test_ws_sync_replays_state                          sync replays the queued requests, the active
                                                         request and the current ask
```
