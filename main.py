from flask import Flask, request, jsonify
import random
import gdown
import numpy as np
import os
from gensim.models import KeyedVectors
import gzip

url = 'https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM'
output = 'GoogleNews-vectors-negative300.bin.gz'
model_file = 'GoogleNews-vectors-negative300.bin'

if not os.path.exists(output):
    print("Downloading the model")
    gdown.download(url, output, quiet=False)

if not os.path.exists(model_file):
    print("Extracting Model")
    with gzip.open(output, 'rb') as gz_file, open(model_file, 'wb') as out_file:
        out_file.write(gz_file.read())

model = KeyedVectors.load_word2vec_format(model_file, binary=True)

app = Flask(__name__)

def predict(word1, sign1, word2, sign2, word3):
  # Perform vector arithmetic: "Beethoven - Music + Painting"
  try:

      # word1 - word2 + word3 = res

      vec1 = model[word1]
      vec2 = model[word2]
      vec3 = model[word3]

      length_vec1 = np.linalg.norm(vec1)
      length_vec2 = np.linalg.norm(vec2)
      length_vec3 = np.linalg.norm(vec3)

      result_vector = vec1 + sign1 * length_vec1/length_vec2 * vec2 + sign2 * length_vec1/length_vec3 * vec3

      # Find the closest words to the resulting vector
      similar_words = model.similar_by_vector(result_vector)
      print(f"Result of '{word1} - {word2} + {word3}':")
      for word, similarity in similar_words:
          same_word = False
          for original_word in [word1, word2, word3]:
            if original_word in word or word in original_word:
              same_word = True
          if(same_word):
            continue;
          else:
            print(f"Result: {word}")
            return word
      
  except KeyError as e:
      print(f"Word not in vocabulary: {e}")
      return "No word found"


def validate_data(data):

    # Expected keys and their types and constraints
    expected_keys = {
        "word1": {"type": str, "max_length": 100},
        "sign1": {"type": int, "allowed_values": [-1, 1]},
        "word2": {"type": str, "max_length": 100},
        "sign2": {"type": int, "allowed_values": [-1, 1]},
        "word3": {"type": str, "max_length": 100}
    }
    
    errors = []
    
    # Validate each key in the expected keys
    for key, specs in expected_keys.items():
        if key in data:
            if not isinstance(data[key], specs["type"]):
                errors.append(f"{key} must be of type {specs['type'].__name__}.")
            elif 'max_length' in specs and len(data[key]) > specs['max_length']:
                errors.append(f"{key} cannot be longer than {specs['max_length']} characters.")
            elif 'allowed_values' in specs and data[key] not in specs['allowed_values']:
                errors.append(f"{key} must be one of {specs['allowed_values']}.")
        else:
            errors.append(f"{key} is missing.")

    return errors

@app.route('/select-word', methods=['POST'])
def select_word():
    data = request.json

    print(data)
    errors = validate_data(data)
    print(errors)

    if(len(errors) == 0):

        predicted_word = predict(data["word1"], data["sign1"],
                                 data["word2"], data["sign2"],
                                 data["word3"])
        print(predicted_word)

        return jsonify({'predicted_word': predicted_word}), 200
    else:
        return jsonify({'error': 'Invalid Data'}), 400

if __name__ == '__main__':
    print("Server Running")
    app.run(host='0.0.0.0', port=8080)

#curl -d '{"word1": "king", "sign1": -1, "word2": "man", "sign2":1, "word3":"woman"}' -H "Content-Type: application/json" -X POST http://127.0.0.1:8080/select-word