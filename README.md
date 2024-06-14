[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/NXyI9KHk)
# CS-552 - Final submission

## Running the Project

To test and run the project, please follow the steps outlined below:

- Navigate to the [model](model) folder
- Ensure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```
- Run the file [evaluator.py](model/evaluator.py):
  ```bash
  python3 evaluator.py
  ```

Execution takes about 5 minutes to output a result, with a GPU-L4. During the execution, progress will be printed through informative messages.

## ⚠️ Important Comment About the Evaluator Process
We have uploaded our reference and policy models to Hugging Face under the following public repositories:

- Reference model: [mya-coder/reference-jim-model](https://huggingface.co/mya-coder/reference-jim-model/tree/main)
- Policy model: [mya-coder/mcqa-policy-jim-model](https://huggingface.co/mya-coder/mcqa-policy-jim-model)
- RAG Policy model (same as policy model): [mya-coder/mcqa-policy-jim-model](https://huggingface.co/mya-coder/mcqa-policy-jim-model)


We experienced unusual behavior during Milestone 2, where the evaluator displayed correct metrics using the local checkpoint paths, but switching to the HuggingFace repository ID resulted in 0% accuracy. **This should not be the case in Milestone 3.** However, to prevent similar issues in Milestone 3, the [main_config.yaml](model/main_config.yaml) currently references the HuggingFace repository ID, but provides the corresponding local checkpoint paths as comments. The HuggingFace checkpoints are identical to the local ones, and one can uncomment these lines to switch paths if such issues reappear.
