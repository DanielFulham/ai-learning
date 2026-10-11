from interfaces.user_input_interface import UserInputInterface


class ConsoleUserInput(UserInputInterface):
    """Talks to the person through the terminal."""

    def ask(self, prompt: str) -> str:
        return input(prompt).strip()

    def show(self, text: str) -> None:
        print(text)
