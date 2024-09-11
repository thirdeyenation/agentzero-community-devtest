import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW

class ExperienceDataset(Dataset):
    def __init__(self, experiences):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        return self.experiences[idx]

class ContinualLearningManager:
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.experiences = []

    def add_experience(self, input_text: str, output_text: str):
        self.experiences.append(f"Input: {input_text} Output: {output_text}")

    def adapt_model(self, batch_size=4, num_epochs=3):
        dataset = ExperienceDataset(self.experiences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        self.model.train()
        for epoch in range(num_epochs):
            for batch in dataloader:
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        print(f"Model adapted over {len(self.experiences)} experiences.")

    def generate_response(self, input_text: str) -> str:
        self.model.eval()
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_length=100, num_return_sequences=1, do_sample=True)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Usage
cl_manager = ContinualLearningManager()

# Simulate some experiences
cl_manager.add_experience("What's the capital of France?", "The capital of France is Paris.")
cl_manager.add_experience("How do I make pasta?", "To make pasta, boil water, add salt, cook pasta until al dente, drain, and add sauce.")

# Adapt the model
cl_manager.adapt_model()

# Generate a response using the adapted model
response = cl_manager.generate_response("Tell me about Rome.")
print(response)