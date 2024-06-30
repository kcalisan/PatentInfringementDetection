import torch
from transformers import BertTokenizer, BertModel
from neo4j import GraphDatabase
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatentInfringementDetection:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def get_patent_descriptions(self):
        logger.info("Fetching patent descriptions from Neo4j")
        query = "MATCH (p:Patent) RETURN p.number AS number, p.description AS description LIMIT 300"
        with self.driver.session() as session:
            result = session.run(query)
            patents = [{"number": record["number"], "description": record["description"]} for record in result]
        logger.info(f"Fetched {len(patents)} patent descriptions")
        return patents

    def vectorize_descriptions(self, descriptions):
        vectors = []
        total_descriptions = len(descriptions)
        logger.info("Vectorizing patent descriptions")
        for idx, description in enumerate(descriptions):
            inputs = self.tokenizer(description, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            vectors.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
            if (idx + 1) % 10 == 0 or (idx + 1) == total_descriptions:
                logger.info(f"Processed {idx + 1}/{total_descriptions} descriptions")
        logger.info("Completed vectorizing patent descriptions")
        return vectors

    def calculate_similarity(self, patents):
        similarities = []
        total_patents = len(patents)
        logger.info("Calculating similarities between patents")
        for i in range(total_patents):
            for j in range(i + 1, total_patents):
                similarity = cosine_similarity([patents[i]['vector']], [patents[j]['vector']], dense_output=True)[0][0]
                if similarity > 0.8:  # Set a threshold for potential infringement
                    similarities.append({
                        "Patent 1": patents[i]['number'],
                        "Patent 2": patents[j]['number'],
                        "Similarity Score": similarity
                    })
            if (i + 1) % 10 == 0 or (i + 1) == total_patents:
                logger.info(f"Calculated similarities for {i + 1}/{total_patents} patents")
        logger.info("Completed similarity calculations")
        return similarities

    def create_potential_infringements(self, similarities):
        logger.info("Creating potential infringement relationships in Neo4j")
        with self.driver.session() as session:
            for idx, similarity in enumerate(similarities):
                query = """
                MATCH (p1:Patent { number: $number1 }), (p2:Patent { number: $number2 })
                MERGE (p1)-[:POTENTIAL_INFRINGEMENT { similarityScore: $score }]->(p2)
                """
                session.run(query, number1=similarity['Patent 1'], number2=similarity['Patent 2'],
                            score=similarity['Similarity Score'])
                if (idx + 1) % 10 == 0 or (idx + 1) == len(similarities):
                    logger.info(f"Created {idx + 1}/{len(similarities)} potential infringement relationships")
        logger.info("Completed creating potential infringement relationships")

    def run(self, save_to_csv=True):
        patents = self.get_patent_descriptions()
        descriptions = [patent['description'] for patent in patents]
        vectors = self.vectorize_descriptions(descriptions)
        for i in range(len(patents)):
            patents[i]['vector'] = vectors[i]
        similarities = self.calculate_similarity(patents)

        if save_to_csv:
            df = pd.DataFrame(similarities)
            df.to_csv('similarities.csv', index=False)
            logger.info("Similarity calculation completed. Results saved to 'similarities.csv'.")
            print(df.to_string(index=False))
        else:
            self.create_potential_infringements(similarities)
            logger.info("Similarity calculation and potential infringement relationships creation completed.")


if __name__ == "__main__":
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "your_password"
    detection = PatentInfringementDetection(neo4j_uri, neo4j_user, neo4j_password)

    # Pass True to save to CSV, or False to create relationships in Neo4j
    detection.run(save_to_csv=True)
