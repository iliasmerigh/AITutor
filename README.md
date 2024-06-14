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

Execution takes about 5 minutes to output a result, with a GPU-L4.

## ⚠️ Important Comment About the Evaluator Process
We have uploaded our reference and policy models to Hugging Face under the following public repositories:

- Reference model: [mya-coder/reference-jim-model](https://huggingface.co/mya-coder/reference-jim-model/tree/main)
- Policy model: [mya-coder/mcqa-policy-jim-model](https://huggingface.co/mya-coder/mcqa-policy-jim-model)
- RAG Policy model (same as policy model): [mya-coder/mcqa-policy-jim-model](https://huggingface.co/mya-coder/mcqa-policy-jim-model)

However, when executing the [evaluator.py](model/evaluator.py) script and referencing these HF repositories in the [main_config.yaml](model/main_config.yaml) file, we achieve a policy reward accuracy of `0%`. This issue seems to be related to accessing the models. Consequently, we opted to run the [evaluator.py](model/evaluator.py) script by directly accessing the reference and policy models from our Git repository. You can locate the reference model under the [model_SFT_final](model/checkpoints/model_SFT_final) checkpoint and the policy model under [MCQA_model](model/checkpoints/MCQA_model) checkpoint. These folders house the identical models to those we have uploaded to Hugging Face.

For more clarity, we also specify the ID of each model on Hugging Face in the [main_config.yaml](model/main_config.yaml) file.
