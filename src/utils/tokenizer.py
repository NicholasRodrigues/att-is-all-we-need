import numpy as np

np.random.seed(42)

class Tokenizer:
    def __init__(self, text: str):
        self.text = text.lower()
        
    def tokenize(self):
        split_text = self.text.split()
        return split_text

        
# if __name__ == '__main__':
#     dummy_text = 'Life is a journey full of unexpected turns. Every step we take, every decision we make, shapes who we become. Embrace the challenges, cherish the small moments, and always keep moving forward.'
#     tk = Tokenizer(dummy_text)
#     tknzd = tk.tokenize()
#     for i in tknzd:
#         print(i)