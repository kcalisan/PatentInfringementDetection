# Patent Infringement Detection

This project aims to detect potential patent infringements using Natural Language Processing (NLP) and Neo4j. The data comes from the Unified Patent Court (UPC) and is processed to find similarities between patent descriptions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Explanation](#explanation)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/kcalisan/PatentInfringementDetection.git
    cd PatentInfringementDetection
    ```

2. Create a virtual environment and activate it:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Update the Neo4j credentials in `main.py` with your Neo4j connection details.

2. Run the script:

    ```bash
    python main.py
    ```

    By default, this will save the results to `similarities.csv`. To create relationships in Neo4j instead, modify the method call in the `main.py` file.

## Explanation

### Fetching Patent Descriptions

The script fetches patent descriptions from a Neo4j database. Each patent node in the database contains a unique patent number and a description of the patent.

### Vectorizing Descriptions

The fetched descriptions are then vectorized using a BERT model. BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art model for NLP tasks. We use the `bert-base-uncased` model from Hugging Face's Transformers library to convert text descriptions into vector representations.

### Calculating Similarities

Once we have the vectors representing the patent descriptions, we calculate the cosine similarity between each pair of vectors. Cosine similarity is a metric used to measure how similar two vectors are. It is defined as the cosine of the angle between the two vectors. A threshold is set to identify potential infringements based on the similarity score.

### Storing Results

The similarities are stored in a CSV file or in the Neo4j database as `POTENTIAL_INFRINGEMENT` relationships, depending on the user’s choice. Storing in Neo4j involves creating relationships between patent nodes with a similarity score.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests.

## License

This project is licensed under the MIT License.
``` &#8203;:citation[oaicite:0]{index=0}&#8203;
