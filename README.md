# AI CFP Powered by RAG

## Project Setup Instructions

Follow these steps to set up your project environment:

1. **Ensure Conda is Installed**
   - Make sure you have Conda installed on your system. You can download it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if it is not already installed.

2. **Open Terminal and Navigate to Project Directory**
   - Open your terminal or command prompt.
   - Navigate to the project directory:
     ```bash
     cd path/to/your/project
     ```

3. **Create the Conda Environment**
   - Create the Conda environment using the provided YAML file:
     ```bash
     conda env create -f rag_env.yaml
     ```

4. **Activate the Conda Environment**
   - Activate the newly created environment:
     ```bash
     conda activate rag_env
     ```

5. **Set Up Environment Variables**
   - Create a `.env` file in the root directory of the project.
   - Add the following line to the `.env` file, replacing `your_api_key` with your actual OpenAI API key:
     ```plaintext
     OPENAI_API_KEY=your_api_key
     ```

6. **Populate the Chroma Database**
   - Run the following command to populate the Chroma database:
     ```bash
     python populate_database.py --reset
     ```

7. **Query GPT-4o with Relevant Documents from the Chroma Database**
   - Run the following command to query GPT-4o with your prompt and relevant documents:
     ```bash
     python query_data.py "What is a 401k?"
     ```


