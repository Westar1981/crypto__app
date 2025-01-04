
class Documentation:
    def __init__(self):
        self.docs = {}

    def add_document(self, title, content):
        self.docs[title] = content

    def update_document(self, title, content):
        if title in self.docs:
            self.docs[title] = content
        else:
            raise ValueError("Document not found")

    def display_document(self, title):
        if title in self.docs:
            print(f"Document: {title}")
            print(self.docs[title])
        else:
            raise ValueError("Document not found")

# Example usage
docs = Documentation()
docs.add_document("Getting Started", "This document covers the basics of getting started with the framework...")
docs.add_document("API Reference", "This document provides detailed information about the API endpoints...")
docs.display_document("Getting Started")
docs.update_document("Getting Started", "Updated content for getting started...")
docs.display_document("Getting Started")