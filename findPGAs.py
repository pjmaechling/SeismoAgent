import time
import sys
from datetime import datetime


class HelloWorldAgent:
    def __init__(self, name="Agent-001", interval=3):
        """
        Initialize the agent with a name and a time interval (in seconds).
        """
        self.name = name
        self.interval = interval
        self.is_running = False

    def perform_task(self):
        """
        The logic the agent performs every cycle.
        """
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] {self.name}: Hello World! I am scanning...")

    def run(self):
        """
        The main loop that keeps the agent alive.
        """
        self.is_running = True
        print(f"--- Starting {self.name} (Press Ctrl+C to stop) ---")

        try:
            while self.is_running:
                self.perform_task()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """
        Clean shutdown procedure.
        """
        self.is_running = False
        print(f"\n[{self.name}]: Shutting down safely. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    # Create an instance of the agent and run it
    my_agent = HelloWorldAgent(name="MyMacAgent")
    my_agent.run()