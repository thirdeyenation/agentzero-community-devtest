from queue import Queue

class HumanVerification:
    def __init__(self):
        self.pending_reviews = Queue()

    def add_for_review(self, content: str):
        self.pending_reviews.put(content)

    def review_content(self) -> list:
        approved_content = []
        while not self.pending_reviews.empty():
            content = self.pending_reviews.get()
            if input(f"Approve this content? (y/n): {content}\n").lower() == 'y':
                approved_content.append(content)
        return approved_content