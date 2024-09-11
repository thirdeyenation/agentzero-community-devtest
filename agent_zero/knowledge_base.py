class KnowledgeBase:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def get_latest(self, n):
        return self.items[-n:]

    def create_snapshot(self):
        # Implement snapshot creation logic
        pass