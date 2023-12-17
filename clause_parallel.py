from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, Depends, HTTPException
from queue import Queue
from threading import Lock
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Clause:
    def __init__(self):
        self.distinct_types = {0: 'SELECT DISTINCT {} FROM {}', 1: 'SELECT MAX(DISTINCT {}) FROM {}',
                               2: 'SELECT MIN(DISTINCT {}) FROM {}', 3: 'SELECT COUNT(DISTINCT {}) FROM {}',
                               4: 'SELECT SUM(DISTINCT {}) FROM {}', 5: 'SELECT AVG(DISTINCT {}) FROM {}'}

        self.types = {0: 'SELECT {} FROM {}', 1: 'SELECT MAX({}) FROM {}', 2: 'SELECT MIN({}) FROM {}',
                      3: 'SELECT COUNT({}) FROM {}', 4: 'SELECT SUM({}) FROM {}', 5: 'SELECT AVG({}) FROM {}'}

        # Load the Universal Sentence Encoder inside the class
        self.embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

        # Load the Question Classifier model inside the class
        self.model = load_model(os.path.join(os.path.abspath(os.path.dirname(__file__)), "Question_Classifier.h5"))
        
        # Use a lock for synchronization
        self.lock = Lock()

    def get_embeddings(self, x):
        embeddings = self.embed(x)
        return np.asarray(embeddings)

    def predict_batch(self, input_list):
        embeddings = self.get_embeddings(input_list)
        predictions = self.model.predict(embeddings)
        return predictions

    def process_user_question(self, question, inttype=False, summable=False, distinct=False):
        try:
            predictions = self.predict_batch([question])
            max_index = np.argmax(predictions)
            clause = self.distinct_types[max_index] if distinct else self.types[max_index]

            if summable and inttype:
                if 'COUNT' in clause:
                    clause = 'SELECT SUM({}) FROM {}'

            return clause
        except Exception as e:
            logger.error(f"Error processing question: {question}. Error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

    def adapt_parallel(self, user_questions, inttype=False, summable=False, distinct=False):
        with self.lock:
            with ThreadPoolExecutor() as executor:
                clauses = list(executor.map(
                    lambda question: self.process_user_question(question, inttype, summable, distinct),
                    user_questions
                ))
        return clauses

# Queue for handling inputs through FastAPI
input_queue = Queue()
clause_instance = Clause()

# Endpoint to receive user questions
@app.post("/process_question")
async def process_question(request: Request, inttype: bool = False, summable: bool = False, distinct: bool = False):
    try:
        data = await request.json()
        user_question = data["question"]
        input_queue.put((user_question, inttype, summable, distinct))
        return {"status": "Question received and processing started"}
    except Exception as e:
        logger.error(f"Error receiving question. Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Background task to process user questions in parallel
def process_questions_in_background():
    while True:
        if not input_queue.empty():
            user_question, inttype, summable, distinct = input_queue.get()
            try:
                result = clause_instance.adapt_parallel([user_question], inttype, summable, distinct)
                logger.info(f"Processed question: {user_question}, Result: {result}")
            except Exception as e:
                logger.error(f"Error processing question: {user_question}. Error: {str(e)}")

# Start the background task
import asyncio
loop = asyncio.get_event_loop()
loop.create_task(process_questions_in_background())
