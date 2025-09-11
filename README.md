# Human Evaluation Interface for AutoRev Task 2

## Setup Instructions

To set up and run the application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Maitreya152/AutoRev_HumanEval_2.git
    cd AutoRev_HumanEval_2
    ```

2.  **Install dependencies:**
    Ensure you have Python installed. Then, install the required packages using pip:
    ```bash
    pip install streamlit
    pip install streamlit[pdf]
    ```

## How to Run the Application

Navigate to the `AutoRev_HumanEval_2` directory and run the Streamlit application:

```bash
streamlit run app.py
```

This command will open the application in your web browser.

## Usage

1.  **Select User**: From the sidebar, choose your username.
2.  **Select Paper ID**: Choose a paper from the dropdown list assigned to your user.
3.  **Review Paper**: The original PDF will be displayed. Below it, you will find different reviews (labeled A, B, C, etc.) with their sections.
4.  **Rate Reviews**: For each point in the Summary, Strengths, Weaknesses, and Questions sections, select a rating.
5.  **Submit Ratings**: Click the "Submit Ratings" button at the bottom of the form. All fields must be rated before submission.

## Output

Your evaluation results will be saved in `evaluation_results.csv` located in the `./data` directory. Please forward this csv file to Maitreya or Ketaki

## Contact

For any questions or issues, please contact Maitreya or Ketaki.
