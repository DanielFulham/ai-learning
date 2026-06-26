from interfaces.input_provider_interface import InputProviderInterface


class ConsoleInputProvider:
    """Input provider backed by stdlib input().

    The production concrete for InputProviderInterface. Reads from stdin
    via input(), printing the prompt to stdout. C4's F06 adds
    prompt_secret() using getpass.getpass; C3 implements prompt() only.
    """

    def prompt(self, message: str) -> str:
        return input(message)
