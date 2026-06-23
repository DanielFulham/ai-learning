class ConsoleInputProvider:
    """stdin-backed input provider.

    Production concrete for interactive runs. `demo.py auth` uses this so
    the user actually types the password. Tests and `demo.py all` use
    `ScriptedInputProvider` instead.
    """

    def prompt(self, message: str) -> str:
        return input(message)