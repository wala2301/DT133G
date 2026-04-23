
## Environment & Tools
- Windows 10 on ASUS
- Git version 2.42.0.Windows.2
- Python 3
- FastAPI for building REST APIs 
- Pydantic for data validation 
- scikit-learn for building retrieval models TF-IDF
- rank-bm25 for BM25
- sentence-transformers for Dense retrieval
- ROUGE library for evaluation of answer
- BERTScore library for evaluation of answer
- SciPy (ANOVA, Kruskal-Wallis) for statistical Analysis
- NumPy
- Pytest for system testing 
- Virtual Environment for environment isolation
- Uvicorn for running server

## Quick Start
1. Create and activate a virtual environment.
2. Install dependencies:
	```
	pip install -r requirements.txt
	```
3. Create `.env` from `.env.example` and update values if needed.
4. Start the API:
	```
	uvicorn main:app --reload
	```
5. Run tests:
	```
	python -m pytest -v
	```
6. Run the only retrieval evaluation for each retrieval method:
    ```
    python -m app.evaluation.run_retrieval_eval
    ```
7. Run the evaluation for only the answer:
    ```
   python -m app.evaluation.run_answer_eval
    ```
8. Run the full evaluation for each retrieval method:
   ```
   python -m app.evaluation.run_full_evaluation
   ```
9. Run a statistical comparison between methods:
   ```
    python -m app.evaluation.run_comparison
   ```


## Environment Variables (.env)
The app can run without external LLM credentials, but for full LLM integration set the values in `.env`:

- `LLM_API_URL` (example: `https://api.openai.com/v1/chat/completions`)
- `LLM_API_KEY`
- `LLM_MODEL` (default: `gpt-4o-mini`)
- `LLM_TIMEOUT_SECONDS` (default: `2.5`)
- `RETRIEVAL_METHOD` (default: `tfidf`)
- `Data_DOC_PATH` (default: `data/documents`)
- `RETRIEVAL_TOP_K` (default: `3`)
- `RETRIEVAL_MAX_TOP_K` (default: `10`)
- `LOG_FILE_PATH` (default: `logs/ai_backend.json`)
- `LOG_RETENTION_DAYS` (default: `30`)
- `LOG_CLEANUP_EVERY_N_WRITES` (default: `20`)


## Purpose
The project aims to design and evaluate the backend architecture of a conversational AI system based on 
**``Retrieval-Augmented Generation``** **RAG**, focusing on small corpus scenarios in software engineering.
The primary scientific objective is to generate knowledge about:
- How the choice of retrieval strategy **``TF-IDF, BM25, Dense Retrieval``** affects system quality.
- Whether the use of complex retrieval techniques such as **``Dense Retrieval``** is justified in small corpus.
- The extent of the relationship between retrieval quality and final response quality.

The project also aims to address a known problem in language models, such as `hallucination`, by providing the model 
with context retrieved from reliable sources such as GitHub documentation och Stack Overflow / Stack Exchange API.

### Concrete Goals
- Implement three retrieval strategies within the same RAG system.
- Build an evaluation dataset containing questions, answers, and relevance labels.
- Measure retrieval quality using Precision@k, Recall@k, and NDCG.
- Measure answer quality using ROUGE-L and BERTScore.
- Measure response time for each method.
- Analyze the relationship between retrieval quality and answer quality.
- Compare results using statistical analysis.

A set of measurable and actionable objectives was defined:
##### System Objectives:
- Building an API using FastAPI to receive questions.
- Implementing three retrieval methods **``TF-IDF, BM25, and Dense Retrieval using sentence-transformers``**.
- Linking retrieval to an answer generation model **LLM**.

##### Data Objectives:
Building a corpus from:
- GitHub REST API documentation https://docs.github.com/en/rest
- Stack Exchange API documentation https://api.stackexchange.com

##### Establish clear selection criteria:
- Document length must be from 1 to 3 paragraphs.
- Relevance to a specific programming field.
- Content clarity.

##### Assessment Objectives:
- Evaluate recall using **``Precision@k, Recall@k, and NDCG@k``**.
- Evaluate responses using **``ROUGE-L and BERTScore``**.

##### Statistical Analysis Objectives:
- Test normality using **``Shapiro-Wilk``**, where **ANOVA** is chosen if data are normal, and **Kruskal-Wallis** if data are not normal.
- Calculate effect size using **Cohen's d**.

## Procedures
Modular architecture was used to increase maintainability and ease of future expansion, where the following were separated:
- API layer
- Retrieval layer
- Data models
- Logging
- Tests
#### Design and Implement an API Backend for Receiving Queries (FR1)
A **FastAPI** application was created using **``app = FastAPI(title="AI RAG Backend")``** and the **``router = APIRouter()``** was used to separate the routes from the main file.
Data models were also defined using **Pydantic** in the **``schemas.py``** which ensures automatic data type verification & prevents errors in requests.
An endpoint was implemented using **``@router.post("/ask", response_model=QuestionResponse)``** where the question is received, **``retrieve_context_bundle``** is called to get relevant context, and then **``generate_answer``** is called to produce the final answer which is returned as a JSON response.
After that, the API was tested using TestClient to verify request success, the present of "answer" key and to confirm that the result is text.

#### Development of a Knowledge Base Retrieval Component (FR2)
At the first a **``knowledge base``** related software engineering  information was created by creating a **``document.txt``** file containing software engineering  information, and each line represents an independent document.
Then the documents were loaded and each line was read as a separate document & stored in a list. 
The TF-IDF model was built once when the application was running to improve performance.

````
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(DOCUMENTS)
````
Then the similarity was calculated, the results were sorted and the best results were returned.
The Retrieval component was tested that the function returns a list, the elements are text and **``top_k``** works correctly.

#### Structured Prompt Construction for LLM Request (FR3)
The prompt is constructed in **``build_prompt(question, context)``** in **``app/llm/llm.py``**.
Retrieved context is formatted as a bullet list and combined with the user question.
This grounds the LLM answer in retrieved documents and reduces hallucinated responses.

The LLM call is wrapped with error handling in **``generate_answer()``**. If the LLM is not configured or unavailable,
the system returns the highest-ranked retrieved context instead of failing.

#### Return Generated Response in JSON (FR4)
The endpoint **``@router.post("/ask", response_model=QuestionResponse)``** returns a validated JSON response.
The response model guarantees an **``answer``** field of type string.
This behavior is validated in API tests to ensure the endpoint always returns JSON output.

#### Development of an anonymized conversation recording system for privacy and to protect sensitive data (FR-5)
Using the **``logging.py``** conversations were recorded with sensitive data hidden and stored in **JSON** format, with
a feature to automatically delete old conversations records after 30 days.

The main function to store the conversations is **``def log_conversation()``** where the policy was implemented by first 
deleting old records, then applying the hiding of sensitive data to the question and answer, then creating a new record containing
``timestamp, question_hash, answer_hash, question, answer, retrieval_method, top_k, latency, num_references, retrieved_references``, and then adding the record to the **``ai_backend.json``** file.

For privacy and analysis, the system stores anonymized readable text and hash values together. This allows troubleshooting and evaluation while still protecting sensitive data.

The anonymization or hiding sensitive data has been executed by the **``def anonymize_text(text: str) -> str:``** function
that before saving any 
question or answer, the following are removed:
* Email address → [EMAIL]
* Phone number → [PHONE]
* ID numbers or long numbers → [ID]

A data retention policy has been implemented whereby any record older than 30 days is deleted using **``def delete_old_records(days: int = LOG_RETENTION_DAYS):``**, 
where:
* The cutoff date is calculated.
* Only records newer than 30 days are retained.
* The file is overwritten, removing older records.

Anonymization was tested to check that: 
* Original data is not stored.
* Email address is replaced with [EMAIL]
* Phone number is replaced with [PHONE]
* ID numbers are replaced with [ID]

Then the logging tested where a temporary file is created for testing, log_conversation is called and the file is read.
This test versifies thatOne record exists and anonymization is applied correctly.

In the Retention Policy Test, an old record more than 30 days old is created, a new record is created, delete old records
is executed and verifies that only the new record remains.

#### Configurable Retrieval Parameters (FR-6)
The **``config.py``** file contains all the modifiable system settings, separating them from the application logic ``Separation of Concerns``.

The purpose of this file is to enable system behavior modification without requiring modification of the source code, through the use of environment variables.

In addition, ``top_k`` can be sent in the request body to configure the number of retrieved documents per request.

#### Scalability (NFR-4)
By designing the system architecture using a modular approach, scalability and easy component replacement are ensured.
To support this, a base interface for the retrieval layer, **``BaseRetriever``**, was created using the concept of abstract classes.

This interface enforces a standardized method called **``retrieve``** across all retrieval algorithms, which allows for 
the future addition of new retrieval algorithms while ensuring their compatibility with the rest of the system.

In the initial version of the system, all operations, such as document loading, similarity calculation, and question 
validation, were written as independent functions. While this approach worked correctly, it made it difficult to later 
expand the system or add new retrieval methods.

To ensure scalability, the code was restructured using object-oriented programming, creating the **``retrieval_tfidf.py``**,
which contains all the functions related to the retrieval process using **``TF-IDF``** representation and cosine similarity measurement.

All retrieval-related operations are consolidated into a single component, making the code more organized, 
easier to understand, and easier to maintain. This allows for the future development of different retrieval algorithms, 
such as Embedding Retrievers or Hybrid Retrievers, while maintaining the same **``retrieve``** interface.
This helps meet scalability and component interchangeability requirements.

An intermediate component called **``retrieval_router.py``** was also created, responsible for selecting the appropriate retrieval 
algorithm based on the system settings in **``config.py``**.

The **``config.py``** file serves as a central configuration file to define the retrieval method used in the system, 
which allows for easy modification of the retrieval method simply by adjusting the settings, without needing to change the core application code.

The **``retrieval_utils.py``** load the documents from data and through it, verify the question (input) before passing it to the retrieval system.

The API layer was separated from the retrieval layer. The API only interacts with the **``RetrievalRouter``** and does not need to know 
the details of the retrieval algorithm being used.

In the **``retrieval_bm25.py``** the **``BM25``** retrieval method was implemented which is a probabilistic development of sparse retrieval.


In the **``retrieval_dense.py``** the semantic retrieval using **``embeddings``** was implemented and is used to 
study the effect of transitioning from verbal matching to semantic representation.


The **``metrics.py``** contains the metrics used to evaluate retrieval quality and response quality, where it contains 
all the core evaluation functions used in the project. This included retrieval evaluation metrics such as 
**``Precision@k, Recall@k, and NDCG@k``**, as well as answer evaluation metrics such as **``ROUGE-L and BERTScore``**.

The **``run_retrieval_evaluation.py``** was used to evaluate retrieval quality independently of the generation phase. 
This allowed for the isolation of the retrieval component's performance and the measurement of its ability to retrieve 
correct documents using metrics such as **``Precision@k, Recall@k, and NDCG@k``**, without interference from the 
quality of the linguistic model's output.


The **``run_answer_evaluation.py``** was used to evaluate the quality of the final answers generated by the system. 
In this phase, the appropriate context for each question is first retrieved using the specified retrieval method, 
and then the context is passed to the generation component to produce the answer. The resulting answer is then compared 
to the reference answer using `ROUGE-L` and `BERTScore`, and the final averages for each retrieval method are calculated.


The **``run_full_evaluation.py``** represents the central phase of the evaluation, combining retrieval, generation, 
and quantitative evaluation into a unified procedure. This file loads the questions and the ground truth, runs the 
specified retrieval method, generates the answers, calculates retrieval and answer metrics, and measures retrieval time, 
generation time, and total time. The detailed results and final summaries are then saved in **JSON** and **CSV** files. 
The resulting values are used to check normality and analyze the relationship between retrieval and answer quality.

After evaluating each retrieval method individually, the **``run_comparison.py``** was used to perform a statistical 
comparison between the three methods across all metrics. This file reads the results from `TF-IDF`, `BM25`, and `Dense`, 
then applies an appropriate statistical test to each metric and calculates the effect sizes between each pair of methods 
using `Cohen's d`. The results are saved in **JSON** and **CSV** files for use in the final report.


The **``statistics_analysis.py``** was implemented to perform statistical analysis on the evaluation results. 
It contains the functions necessary to examine data properties, select appropriate statistical tests, and analyze 
effect size. Where through its checks for normality using **``Shapiro-Wilk``** and then selecting the appropriate test 
to compare the methods depending on the nature of the data, where if the data is normal the **``ANOVA``** is used 
but if the data is not normal the **``Kruskal-Wallis``** is used. **``Cohen's d``** was used to estimate the 
effect size between pairs of methods, and correlation analysis `Pearson` or `Spearman` was used to examine the 
relationship between recall quality and response quality.

The evaluation logic is separated into several independent files to improve the clarity of the software structure and 
its reusability. The **``metrics.py``** contains the basic functions for calculating retrieval and answer metrics, 
while **``run_retrieval_evaluation.py``** is dedicated to evaluating retrieval only, and **``run_answer_evaluation.py``** to 
evaluating answer quality. The **``run_full_evaluation.py``** was configured to perform a complete evaluation of a 
single retrieval method, including retrieval, generation, evaluation, and time measurement. 
Subsequently, the **``run_comparison.py``** was used to perform a statistical comparison between the three methods, 
while **``statistics_analysis.py``** collected the necessary statistical tools to examine normality, select the 
appropriate test, and calculate correlation and effect size.


## Discussion
A **REST API** backend has been successfully designed and implemented, capable of:
* Receiving user queries via HTTP POST
* Processing the request
* Returning the answer in JSON format.

**FR2** was successfully implemented where a knowledge base was loaded from a text file **``def load_documents()``**, the texts were converted to 
a digital representation **TF-IDF**, the similarity between the question **``similarities = cosine_similarity(question_vector, doc_vectors)``** and the documents was calculated, and the most relevant documents were retrieved.

**FR3** was successfully implemented by constructing a structured prompt with retrieved context plus the user question before sending it to the LLM.

**FR4** was successfully implemented where the `/ask` endpoint returns the generated answer in a validated JSON response model.

**FR5** was successfully implemented where all conversations are recorded without retaining sensitive data which covers the **NFR-3 hide sensitive data**, and any record saved before 30 days is deleted.
According to the **NFR-7** the program will record **``timestamp, question_hash, answer_hash, question, answer, RETRIEVAL_METHOD, top_k,latency, num_references, retrieved_references``** and this have been successfully achieved.


**FR6** was successfully implemented that all settings are compiled in **``config.py``** to support different operating environments, 
facilitating modification of system settings without changing the code, thus meeting system configurability requirements.

**NFR-1 (Performance)** was implemented by optimizing the request hot path:
* Retrieval ranking is computed once per request and reused for both context and references.
* Logging writes are append-only, while retention cleanup runs periodically instead of rewriting the full log on every request.
* LLM request timeout is configurable (default 2.5 seconds) to keep response latency bounded under normal load.
* A performance test validates average response time for repeated API calls stays below 3 seconds.

**Usability (NFR-2)**
Has been achieved where the system is user-friendly, requiring no prior training, as it features a clear and simple interface that utilizes
artificial intelligence to understand user questions and provide understandable answers.

**Scalability (NFR-4)**
The system enables the easy addition or modification of system components and supports new models or retrieval methods.

By separating the components so they are independent — **``retrieval, llm, logging, config``**, etc. — 
the non-functional requirement **NFR-5 (modular structure where components are separated into independent modules)** is also met,
meaning the components are only accessed through their public interfaces without exposing their internal details.

**Handling Invalid Inputs NFR-6**
has been successfully implemented where the system employs multiple data validation layers to ensure reliability when handling incorrect user input.
User queries are validated before processing to prevent the entry of empty data, queries containing only numbers, or invalid formats.

Additionally, the retrieval module includes exception handling to prevent system crashes if errors occur during vector conversion or similarity calculations.
If no relevant documentation is found, the system displays a clear message to the user instead of a blank or failed response.


The evaluation process was carried out using several independent scripts. `run_full_evaluation.py` was used to 
evaluate each retrieval method individually, and then `run_comparison.py` was used to perform a statistical comparison 
between the three methods. The system was also run via **FastAPI** for testing in an interactive environment, 
and automated tests were used to verify performance requirements.



### Example API Request
```
curl -X POST "http://127.0.0.1:8000/ask" \
	-H "Content-Type: application/json" \
	-d '{"question":"Can I take antibiotics?","top_k":3}'
```
