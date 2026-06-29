import getpass

from interfaces.input_provider_interface import InputProviderInterface


class ConsoleInputProvider:
    """Input provider backed by stdlib input() and getpass.getpass().

    The production concrete for InputProviderInterface. prompt() reads
    from stdin via input(), echoing the typed text to stdout.
    prompt_secret() reads via getpass.getpass() so the password does not
    echo to the terminal. Both stdlib, no third-party dependency.
    """

    def prompt(self, message: str) -> str:
        return input(message)

    def prompt_secret(self, message: str) -> str:
        return getpass.getpass(message)
