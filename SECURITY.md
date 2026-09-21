# Security

## Reporting

Please report vulnerabilities privately through GitHub's security advisory
form rather than a public issue.

## What this library does with your data

- **Memories are stored where you point it**: a local SQLite file by default,
  or your own MySQL. Nothing is sent anywhere except to the model endpoints you
  configure.
- **Conversation content reaches your model provider.** That is inherent to
  what the library does — building memories requires reading the conversation —
  but it is worth stating plainly, because "local storage" can be misread as
  "nothing leaves the machine".
- **No telemetry.** The library does not phone home, and there is no flag to
  turn off because there is nothing to turn off.
- **Tracing is off unless configured.** With no Langfuse keys set, every tracing
  call is a no-op.

## Handling secrets

API keys come from the environment or a `.env` file, never from code. The `.env`
file is gitignored.

Deleting a user's data is `Memory.reset(user_id=...)`, which removes their rows
from every table. Media files under the data directory are not removed by it —
they are content-addressed and may be shared between users.
