import asyncio
import sys
import os
import boto3
import json
import asyncio
from copy import deepcopy
from dotenv import load_dotenv
from neo4j import GraphDatabase, AsyncGraphDatabase
import aioboto3
import nest_asyncio
from typing import Literal, List, Any
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase
from openai import AzureOpenAI, AsyncAzureOpenAI

load_dotenv()

model_name = "us.meta.llama3-2-11b-instruct-v1:0"
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

async def async_gen_chat(system: str, user: str) -> str:
    session = aioboto3.Session()
    async with session.client(
        "bedrock-runtime",
        aws_access_key_id=os.getenv("AWS_BEDROCK_SA_AK", ""),
        aws_secret_access_key=os.getenv("AWS_BEDROCK_SA_SK", ""),
        region_name=os.getenv("AWS_BEDROCK_META_REGION", "")
    ) as client:
        response = await client.converse(
            modelId=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user}]
                }
            ],
            system=[{"text": system}],
            inferenceConfig={
                "maxTokens": 1024,
                "temperature": 0,
                "stopSequences": []
            }
        )
        return response['output']['message']['content'][0]['text']

client = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_BEDROCK_SA_AK",""),
    aws_secret_access_key=os.getenv("AWS_BEDROCK_SA_SK",""),
    region_name=os.getenv("AWS_BEDROCK_META_REGION","")
)

def gen_chat(system: str, user: str) -> str:
    response = client.converse(
        modelId=model_name,
        messages=[{
            "role": "user",
            "content": [{
                "text": user
            }]
        }],
        system=[{"text": system}],
        inferenceConfig={
            "maxTokens": 1024,
            "temperature": 0,
            "stopSequences": []
        }
    )
    return response['output']['message']['content'][0]['text']

def generate_embedding(input_texts, type: str = "search_document"):
    """
    Generate embeddings using Amazon Bedrock.
    """
    model_id = "cohere.embed-multilingual-v3"
    try:
        # Prepare the payload
        payload = {
            "texts": input_texts,
            "input_type": type,
            "truncate": "START"
        }
        
        # Invoke the model
        response = client.invoke_model(
            modelId=model_id,  # The model ID as configured in Bedrock
            contentType="application/json",  # Content type for the payload
            accept="*/*",  # Accept type for the response
            body=json.dumps(payload)  # Serialize the payload into JSON
        )
        
        # Parse the response
        # print(response['body'].read())
        response_body = json.loads(response['body'].read())
        embedding = response_body.get("embeddings", None)
        
        if embedding is None:
            raise ValueError("No embedding returned in the response.")
        
        return embedding
    
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

class Neo4jHelper:

    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def create_node_if_not_exists(self, label, code, properties):
        query = f"""
        MERGE (n:{label} {{code: $code}})
        SET
        {', '.join([f'n.{key} = ${key}' for key in properties.keys()])}
        RETURN n
        """
        parameters = {"code": code, **properties}
        
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return result.single()  # Return the first result (the created or matched node)
    
    def run_query(self, query):
        with self._driver.session() as session:
            result = session.run(query)
            return result
    
    def execute_query(self, query, **kwargs):
        records, summary, keys = self._driver.execute_query(
            query,
            **kwargs
        )
        return [record.data() for record in records]
class Neo4jHelper:

    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def create_node_if_not_exists(self, label, code, properties):
        query = f"""
        MERGE (n:{label} {{code: $code}})
        SET
        {', '.join([f'n.{key} = ${key}' for key in properties.keys()])}
        RETURN n
        """
        parameters = {"code": code, **properties}
        
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return result.single()  # Return the first result (the created or matched node)
    
    def run_query(self, query):
        with self._driver.session() as session:
            result = session.run(query)
            return result
    
    def execute_query(self, query, **kwargs):
        records, summary, keys = self._driver.execute_query(
            query,
            **kwargs
        )
        return [record.data() for record in records]

class AsyncNeo4jHelper:

    def __init__(self, uri, user, password):
        # Use the async driver
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        # Close the driver asynchronously
        await self._driver.close()

    async def create_node_if_not_exists(self, label, code, properties):
        # Use an async session
        query = f"""
        MERGE (n:{label} {{code: $code}})
        SET
        {', '.join([f'n.{key} = ${key}' for key in properties.keys()])}
        RETURN n
        """
        parameters = {"code": code, **properties}
        
        async with self._driver.session() as session:
            # Run the query asynchronously
            result = await session.run(query, parameters)
            # Fetch the first result asynchronously
            return await result.single()  # Return the first result (the created or matched node)

    async def run_query(self, query):
        async with self._driver.session() as session:
            # Run the query asynchronously
            result = await session.run(query)
            return result  # Note: You'll need to process the result asynchronously

    async def execute_query(self, query, **kwargs):
        records, summary, keys = await self._driver.execute_query(
            query,
            **kwargs
        )
        # Process the records asynchronously
        return [record.data() for record in records]




async def main(user_input: str):
    # Connect to Neo4j
    neo4j_helper = Neo4jHelper(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    async_neo4j_helper = AsyncNeo4jHelper(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # load prompts
    with open("prompt_templates/node_reflection_no_traversal.txt", "r") as t: node_reflection_template = t.read()
    with open("prompt_templates/starter.txt", "r") as t: starter_template = t.read()
    
    def similar_node_by_vector(node_type: Literal["Product", "Brand", "Promotion", "Category"], query: str, threshold: float = 0.7, top_n: int = 10):
        template = """MATCH (p:{node_type}) WITH p,
        vector.similarity.cosine(p.embedding, $embedding) AS score
        WHERE score > {threshold} 
        RETURN labels(p) AS type, p.code AS code, p.title AS title, p.name AS name, p.description AS description, p.elabDescription AS elabDescription, p.elabCountryOfOrigin AS countryOfOrigin, p.stockLevelStatus AS stockLevelStatus, score
        ORDER BY score DESC LIMIT {top_n};"""
        
        result = neo4j_helper.execute_query(
            query=template.format(node_type=node_type, threshold=threshold, top_n=top_n),
            embedding=generate_embedding([query], type="search_query")[0]
        )
        
        return result
    
    all_visited_nodes = []

    starter = json.loads(gen_chat(system=starter_template, user=user_input))
    user_input_embedding = generate_embedding([user_input], type="search_query")[0]
    all_starts = []
    for k, v in starter.items():
        all_possible_starts = similar_node_by_vector(node_type=k, query=", ".join(v))
        # sorted_starts = sorted(all_possible_starts, key=lambda x: x["score"], reverse=True)
        all_starts.extend(all_possible_starts)
    
    class ReflectiveNode:
    
        class Reflection:
            related: bool
            explanation: str = ""
            relevance_score: float = 0
            def __init__(self, **kwargs):  self.__dict__.update(kwargs)
        
        class Relationship:
            direction: str
            relationship: str
            node: 'ReflectiveNode'
            
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
            def to_partial_dict(self) -> dict:
                partial_dict = {
                    "direction": self.direction,
                    "relationship": self.relationship,
                    "code": self.node.code,
                    "name": self.node.name,
                    "type": self.node.type
                }
                # return json.dumps(partial_prompt)
                return partial_dict
            
        code: str
        name: str
        user_query: str
        user_query_embedding: list[float]
        type: Literal["Product", "Category", "Brand", "Promotion"]
        description: str = ""
        relationships: List[Relationship] = []
        previous_node: 'ReflectiveNode' = None
        previous_traversal_reason: str = ""
        n_hop: int = 0
        
        def __init__(self, 
                    code, 
                    name,
                    type, 
                    user_query = "",
                    user_query_embedding = [],
                    description = "",
                    relationships = [],
                    previous_node = None,
                    previous_traversal_reason = "",
                    n_hop = 1,
                    **kwargs):
            all_visited_nodes.append(code)
            self.code = str(code)
            self.name = name
            self.type = type
            self.user_query = user_query
            self.user_query_embedding = user_query_embedding
            self.description = description
            self.previous_node = previous_node
            self.previous_traversal_reason = previous_traversal_reason
            self.n_hop = n_hop
            if relationships: self.relationships = relationships
            self.__dict__.update(kwargs)
        
        def to_partial_dict(self) -> dict:
            partial_dict = {
                "code": self.code,
                "type": self.type,
                "name": self.name,
                "description": self.description
            }
            if self.type == "Product":
                partial_dict["countryOfOrigin"] = getattr(self, "countryOfOrigin", "N/A")
                partial_dict["stockLevelStatus"] = getattr(self, "stockLevelStatus", "N/A")
            return partial_dict
        
        def to_partial_prompt(self) -> str:
            return json.dumps(self.to_partial_dict(), indent=2)
        
        async def set_relationships(self):
            q_code = self.code if self.type == "Promotion" else f"\"{self.code}\""
            # this big query gets all its neighborhoods and filter out the top 5 of each type my similarity to the user's query
            q = f"""
            WITH {self.user_query_embedding} AS queryEmbedding  // Replace with the actual query embedding
            MATCH (n:{self.type} {{code: {q_code}}})
            OPTIONAL MATCH (l)-[r]-(n)
            WHERE l.embedding IS NOT NULL
            WITH l, r, n, vector.similarity.cosine(l.embedding, queryEmbedding) AS similarity
            ORDER BY similarity DESC
            WITH labels(l) AS nodeLabels, l, r, n, similarity
            UNWIND nodeLabels AS nodeLabel
            WITH nodeLabel, l, r, n, similarity
            ORDER BY similarity DESC  // Re-order based on similarity for each label
            WITH nodeLabel, collect({{
                nodeType: nodeLabel,
                code: l.code,
                name: l.name,
                title: l.title,
                description: l.description,
                elabDescription: l.elabDescription,
                countryOfOrigin: l.elabCountryOfOrigin,
                stockLevelStatus: l.stockLevelStatus,
                relationship: r,
                direction: CASE 
                    WHEN (n)-[r]->(l) THEN 'OUTGOING' 
                    WHEN (n)<-[r]-(l) THEN 'INCOMING' 
                    ELSE 'UNDIRECTED' 
                END,
                similarity: similarity
            }}) AS collectedNodes
            WITH nodeLabel, collectedNodes[0..5] AS topNodes
            UNWIND topNodes AS node
            RETURN node.nodeType AS nodeType, node.code AS code, node.name AS name, node.title AS title, node.description AS description, 
                node.elabDescription AS elabDescription, node.countryOfOrigin AS countryOfOrigin, node.stockLevelStatus AS stockLevelStatus, 
                node.relationship AS relationship, node.direction AS direction, node.similarity AS similarity
            """
            db_results = await async_neo4j_helper.execute_query(q)
            self.relationships=[]
            for r in db_results:
                related_node = ReflectiveNode(
                    code=r["code"],
                    name=r["name"] if r["name"] else r["title"],
                    type=r["nodeType"],
                    description=r["elabDescription"] if r["elabDescription"] else r["description"],
                    user_query=self.user_query,
                    user_query_embedding=self.user_query_embedding,
                    previous_node=self,
                    n_hop=self.n_hop + 1
                )
                if r["countryOfOrigin"]: related_node.countryOfOrigin = r["countryOfOrigin"]
                if r["stockLevelStatus"]: related_node.stockLevelStatus = r["stockLevelStatus"]
                self.relationships.append(self.Relationship(
                    direction = r["direction"],
                    relationship = r["relationship"][1],
                    node = related_node
                ))
                
        def match_node(self, type: str, code: str):
            for r in self.relationships:
                if getattr(r.node, 'type', "") == type and getattr(r.node, 'code', "") == code:
                    return r.node
            return None
        
        async def reflect(self) -> Reflection:
            prompt = node_reflection_template
            prompt = prompt.replace("CURRENT_NODE", self.to_partial_prompt())
            prompt = prompt.replace("CURRENT_RELATIONSHIPS", json.dumps([r.to_partial_dict() for r in self.relationships], indent=2))
            gen_result = await async_gen_chat(system=prompt, user=self.user_query)
            try:
                gen_result_dict = json.loads(gen_result)
            except Exception as e:
                print(e)
                return self.Reflection(
                    related=False,
                    relevance_score=0,
                    explanation=""
                )
                
            print(json.dumps(gen_result_dict, indent=2))
            if gen_result_dict["related"]:
                return self.Reflection(
                    related=True,
                    relevance_score=gen_result_dict["relevance_score"],
                    explanation=gen_result_dict["explanation"]
                )
            else:
                return self.Reflection(
                    related=False,
                    relevance_score=gen_result_dict["relevance_score"],
                    explanation=gen_result_dict["explanation"]
                )
    related_nodes = []
    nodes_to_analyze = [
        ReflectiveNode(
            code=item["code"],
            name=item["name"] if item["name"] else item["title"],
            type=item["type"][0],
            user_query=user_input,
            user_query_embedding=user_input_embedding,
            description=item["description"] if item["description"] else item["elabDescription"],
            countryOfOrigin=item["countryOfOrigin"],
            stockLevelStatus=item["stockLevelStatus"],
        ) for item in all_starts
    ]


    async def analyze(node: ReflectiveNode):
        await node.set_relationships()
        reflection = await node.reflect()
        if reflection.related:
            node_dict = node.to_partial_dict()
            node_dict["relationships"] = [r.to_partial_dict() for r in node.relationships]
            related_nodes.append(
                {
                    "node": node_dict,
                    "explanation": reflection.explanation,
                    "relevance_score": reflection.relevance_score
                }
            )
    tasks = [analyze(n) for n in nodes_to_analyze]
    node_analysis_result = await asyncio.gather(*tasks)
    
    
    # filter by relevance
    most_related_nodes = []
    for node in related_nodes:
        if node["relevance_score"] >= 0.6: 
            node.pop("explanation")
            node.pop("relevance_score")
            most_related_nodes.append(node)
        
    print(json.dumps(most_related_nodes, indent=2))
    
    
    provided = json.dumps(most_related_nodes, indent=2) if most_related_nodes else "No nodes fulfill the user's query"
    
    # answer generation
    system_prompt = f"""
    You help answer the user's query by accessing a knowledge graph.

    Below are the relevant nodes and relationships that may help you answer the user:
    {provided}

    """
    print(f"""
Query on all visited nodes:

MATCH (n)
WHERE n.code IN {json.dumps(all_visited_nodes)}
OPTIONAL MATCH (n)-[r]-(m)
WHERE m.code IN {json.dumps(all_visited_nodes)}
RETURN n, r, m
""")
    answer = gen_chat(system=system_prompt, user=user_input)
    print(answer)
    
if __name__ == "__main__":
    user_input = input("Your query: ")
    sys.stdout = open(f"logs/sr/{user_input}.log", "w")
    try:
        asyncio.run(main(user_input))
    finally:
        sys.stdout.close()
        sys.stdout = sys.__stdout__