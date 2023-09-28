# BERT-Neural-Language-Interface-Explainability-Explorer


Author: Manai Mortadha
__________________________________________________________________________________________________________________

## Description

Welcome to the Neural Language Interface (NLI) Explain project! This repository is dedicated to exploring and explaining the decision-making process of BERT models in the context of Natural Language Inference (NLI) tasks. We employ Feature Interaction methods to shed light on why BERT makes specific predictions in NLI.

### Explainers Included

We offer several feature interaction explainers to understand BERT's decisions in NLI tasks:

1. **IH - Integrated Hessians**: Utilizes the [integrated Hessians method](https://github.com/suinleelab/path_explain) for explanation.

2. **Archipelago**: Incorporates the [Archipelago explainer](https://github.com/mtsang/archipelago) for decision interpretation.

3. **X-Archipelago (Ours)**: Offers a cross-sentence version specially designed for NLI examples with two sentences, building upon the Archipelago explainer.

4. **MAFIA (Ours) - MAsk-based Feature Interaction Attribution**: Introduces our own MAFIA explainer for enhanced feature interaction analysis.

## Requirements

Ensure you have the following Python libraries installed:

```
nltk
pandas
numpy
torch
transformers
```

## How to Use

Follow these steps to effectively use the repository and explore BERT's decisions in NLI tasks:

1. **Download the e-SNLI Dataset**:
   - Visit the [e-SNLI dataset repository](https://github.com/OanaMariaCamburu/e-SNLI/tree/master/dataset).
   - Download the dataset files and place them in the `data/` folder of this repository.

2. **Preprocess the Data**:
   - Run the `python prepare_data.py` script to preprocess the dataset.

3. **Generate Explanations**:
   - For each explainer (lime, IH, Arch, Mask), navigate to the `scripts/` folder.
   - Execute the respective script to generate explanations. These explanations will be stored as `.json` files in the `explanations/` directory (created if not already).

   - For in-depth details on explanation generation, refer to the `explainers/save_explanations.py` file.
   - To explore the implementation of each explainer, navigate to the `explainers/` directory.

4. **Evaluation**:
   - To evaluate the explanations, you can use the `scripts/eval_explanation.sh` script.
   - Modify the script according to your specific evaluation needs and preferences.

Feel free to explore, experiment, and enhance the explanations of BERT's decisions in NLI tasks using this repository.
