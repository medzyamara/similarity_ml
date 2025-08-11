from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('all-mpnet-base-v2')

@app.route("/check_answer", methods=["POST"])
def check_answer():
    data = request.json
    embeddings = model.encode([data["correct_answer"], data["user_answer"]], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    is_correct = similarity_score >= 0.65
    return jsonify({
        "similarity_score": round(similarity_score, 2),
        "is_correct": is_correct
    })

if __name__ == "__main__":
    app.run()
