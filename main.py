from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-mpnet-base-v2')  # more accurate than MiniLM

def is_answer_correct(correct_answer, user_answer, threshold=0.65):
    embeddings = model.encode([correct_answer, user_answer], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    print(f"Similarity score: {similarity_score:.2f}")
    return similarity_score >= threshold

correct = input("Enter the correct answer: ")
user = input("Enter your answer: ")

if is_answer_correct(correct, user):
    print("✅ Correct!")
else:
    print("❌ Incorrect.")
