# LTRC-IIITH at EHRSQL 2024: Enhancing Reliability of Text-to-SQL Systems through Abstention and Confidence Thresholding

<p align="center">
  <img src="assets/System Workflow.png" alt="System Workflow"/>
</p>

We present our work in the EHRSQL 2024 shared task which tackles reliable text-to-SQL modeling on Electronic Health Records. Our proposed system tackles the task with three modules - abstention module, text-to-SQL generation module, and reliability module. The abstention module identifies whether the question is answerable given the database schema. If the question is answerable, the text-to-SQL generation module generates the SQL query and associated confidence score. The reliability module has two key components - confidence score thresholding, which rejects generations with confidence below a pre-defined level, and error filtering, which identifies and excludes SQL queries that result in execution errors. In the official leaderboard for the task, our system ranks 6th.

### Setting up the Environment

```bash
pip install -r requirements.txt
git clone https://github.com/glee4810/ehrsql-2024
```
Download and preprocess the database following the instructions in the repository ([link](https://github.com/glee4810/ehrsql-2024?tab=readme-ov-file#database))

### Instructions

To run a particular experiment, execute the associated script found in `scripts` from the root directory of the repository.
- Experiment 1: Text-to-SQL Generation (Initial Prompt)
- Experiment 2: Text-to-SQL Generation + Error Filtering
- Experiment 3: Abstention + Text-to-SQL Generation + Error Filtering
- Experiment 4: Abstention (Multi-Task) + Text-to-SQL Generation (Multi-Task) + Error Filtering
- Experiment 5: Abstention (Multi-Task) + Text-to-SQL Generation + Error Filtering + Confidence Thresholding

Example:
```bash
bash scripts/experiment_5.sh
```

### Citation
> LTRC-IIITH at EHRSQL 2024: Enhancing Reliability of Text-to-SQL Systems through Abstention and Confidence Thresholding. Jerrin John Thomas and Pruthwik Mishra and Parameswari Krishnamurthy and Dipti Sharma. Submitted to NAACL-ClinicalNLP 2024.