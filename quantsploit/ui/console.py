"""
Main interactive console for Quantsploit
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.history import InMemoryHistory
from .display import Display
from .commands import CommandHandler


class Console:
    """
    Interactive console interface for Quantsploit.
    Similar to Metasploit's msfconsole.
    """

    def __init__(self, framework):
        self.framework = framework
        self.display = Display()
        self.command_handler = CommandHandler(framework)
        self.session = PromptSession(history=InMemoryHistory())
        self.running = True

        # Setup auto-completion
        self.completer = self._create_completer()

        # Prompt style
        self.prompt_style = Style.from_dict({
            'prompt': '#00ff00 bold',
            'module': '#ff00ff bold',
        })

    def _create_completer(self) -> WordCompleter:
        """Create command completer"""
        commands = list(self.command_handler.commands.keys())
        return WordCompleter(commands, ignore_case=True)

    def _get_prompt(self) -> str:
        """Generate the prompt string"""
        if self.framework.session.current_module:
            module_name = self.framework.session.current_module.name
            return f"[('class:prompt', 'quantsploit')](" + \
                   f"[('class:module', '{module_name}')]) > "
        else:
            return "[('class:prompt', 'quantsploit')] > "

    def start(self):
        """Start the interactive console"""
        self.display.print_banner()
        self.display.print_info(f"Loaded {len(self.framework.modules)} modules\n")

        while self.running:
            try:
                # Get user input
                prompt_text = self._get_prompt()
                user_input = self.session.prompt(
                    prompt_text,
                    completer=self.completer,
                    style=self.prompt_style
                )

                # Execute command
                self.running = self.command_handler.execute(user_input)

            except KeyboardInterrupt:
                self.display.print("\n")
                self.display.print_warning("Use 'exit' or 'quit' to exit")
                continue

            except EOFError:
                break

            except Exception as e:
                self.display.print_error(f"Error: {str(e)}")

        # Shutdown
        self.framework.shutdown()
        self.display.print_success("Goodbye!")

    def stop(self):
        """Stop the console"""
        self.running = False
