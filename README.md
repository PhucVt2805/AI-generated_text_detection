<h1>AI-Generated Text Detection 🤖✍️</h1>

<p>
    <img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python Version">
</p>
<h2>Introduction:</h2>
<hr width="100%" color="#000000">
<br>
<h3>Welcome to the AI-Generated Text Detection project!</h3>
<br>
<p>This project provides a powerful set of tools in Python to check if a piece of text is AI-generated. With the rapid development of large language models, the ability to distinguish between human-written and AI-generated content is becoming increasingly important. This project is designed to address that challenge.</p>
<br>
<p><strong>Key Features:</strong></p>
<ul>
    <li>✅ <strong>Whole Text Detection:</strong> Check if an entire document is likely to be AI-generated.</li>
    <li>🔎 <strong>Sentence-Level Analysis:</strong> Evaluate each individual sentence to determine the likelihood of it being AI-generated.</li>
    <li>🔧 <strong>Model Fine-tuning:</strong> Provides the ability to fine-tune one or more existing detection models on your own dataset to optimize for your specific usage needs.</li>
</ul>

<h2>Repository Structure:</h2>
<hr width="100%" color="#000000">
<br>
<p>The repository is neatly organized with the following main directories and files:</p>
<br>
<pre>
.
├── main.py           # Main script containing examples of how to use the library
├── config/           # Module containing configuration files for the project (e.g., model settings, thresholds)
├── utils/            # Module containing supporting utility functions (e.g., data processing)
├── src/              # Module containing the core source code for the AI detection logic
└── requirements.txt  # List of required Python libraries to run the project
</pre>

<h2>Getting Started:</h2>
<hr width="100%" color="#000000">
<br>
<p>Follow these steps to get the project up and running on your local machine.</p>
<br>
<h3>Prerequisites:</h3>
<ul>
    <li><a href="https://www.python.org/downloads/">Python 3.11</a></li>
</ul>

<h3>Installation:</h3>
<br>
<p>1. Clone the repository to your machine (If this is a Git repo):</p>
<pre>
<code>git clone https://github.com/PhucVt2805/AI-generated_text_detection.git</code>
</pre>
<br>
<p>2. Navigate into the project directory:</p>
<pre>
<code>cd AI-generated_text_detection</code>
</pre>
<br>
<p>2.1 Create a virtual environment (optional but recommended):</p>
<pre>
<code>python -m venv venv</code>
</pre>
<br>
<p>2.2 Activate the virtual environment:</p>
<pre>
<code>source venv/bin/activate  # For Unix-based systems
venv/Scripts/activate  # For Windows</code>
</pre>
<br>
<p>3. Install the required Python libraries. All dependencies are listed in <code>requirements.txt</code>:</p>
<pre>
<code>pip install -r requirements.txt</code>
</pre>
<br>
<p>4. Run the automatic setup script to initialize the environment (e.g., download necessary models):</p>
<pre>
<code>python -m config.auto_setup</code>
</pre>
<p>This script will help you download the necessary resources for the project to function.</p>

<h3>How to Run:</h3>
<hr width="100%" color="#000000">
<br>
<p>After completing the installation and setup, you can run the main script:</p>
<br>
<pre>
<code>python main.py</code>
</pre>
<br>
<p>The <a href="./main.py">main.py</a> file contains example usage demonstrating how to use the AI text detection features and how to perform model fine-tuning. Refer to this file for more details on integrating and using the library in your code.</p>

<h2>Configuration:</h2>
<hr width="100%" color="#000000">
<br>
<p>The <code>config/</code> directory contains files that allow you to customize various settings of the library, such as the type of model used, thresholds for determining AI-generated text, etc. Explore the files in this directory to tailor the project to your needs.</p>

<h2>Fine-tuning:</h2>
<hr width="100%" color="#000000">
<br>
<p>The project supports fine-tuning the detection model on your custom dataset. This is particularly useful if you are working with a specific type of text or want to improve detection performance for a particular domain.</p>
<br>
<p>Please see the instructions and examples in the <a href="./main.py">main.py</a> file for how to prepare your data and perform the fine-tuning process.</p>

<hr width="100%" color="#000000">
<h4>If you have any questions, please contact <b>Phuc2805vt@gmail.com</b></h4>