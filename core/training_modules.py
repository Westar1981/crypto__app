
class TrainingModule:
    def __init__(self, title, content):
        self.title = title
        self.content = content

    def display(self):
        print(f"Training Module: {self.title}")
        print(self.content)

# Example usage
module1 = TrainingModule("Introduction to the Framework", "This module covers the basics of the framework...")
module2 = TrainingModule("Advanced Features", "This module covers advanced features and customization options...")
module1.display()
module2.display()