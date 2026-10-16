# Golden utterances

The sentences rehearsed for the stage and used by "Run golden set" (typed, straight into the NLU). Source of truth:
`core/src/commonMain/kotlin/sk/ainet/examples/smarthome/pipeline/GoldenSet.kt`.

| # | Say | Expected function |
|---|---|---|
| 1 | Turn on the kitchen light | set_light |
| 2 | Switch off the bedroom light | set_light |
| 3 | Dim the living room light to twenty percent | set_light |
| 4 | Set the bedroom to twenty one degrees | set_thermostat |
| 5 | Make it nineteen degrees in the whole house | set_thermostat |
| 6 | Close the living room blinds | set_blinds |
| 7 | Open the kitchen blinds halfway | set_blinds |
| 8 | Lock the front door | set_lock |
| 9 | Movie time | set_scene |
| 10 | What is the status of the hallway | get_status |
| 11 | What is the weather like tomorrow | *none* → escalation seam |
| 12 | Order a pizza for dinner | *none* → escalation seam |

Record results per run (device, cpu/gpu, date) in the work log.
