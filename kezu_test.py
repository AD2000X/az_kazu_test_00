import hydra
from hydra.utils import instantiate

from kazu.data.data import Document
from kazu.pipeline import Pipeline
from kazu.utils.constants import HYDRA_VERSION_BASE
from pathlib import Path
import os

# the hydra config is kept in the model pack
cdir = Path(os.environ["KAZU_MODEL_PACK"]).joinpath("conf")


@hydra.main(
    version_base=HYDRA_VERSION_BASE, config_path=str(cdir), config_name="config"
)
def kazu_test(cfg):
    pipeline: Pipeline = instantiate(cfg.Pipeline)
    text = "EGFR mutations are often implicated in lung cancer"
    doc = Document.create_simple_document(text)
    pipeline([doc])
    print(f"{doc.get_entities()}")


if __name__ == "__main__":
    kazu_test()


# Quickstart

from kazu.data.data import Document, Entity
from kazu.steps.document_post_processing.abbreviation_finder import (
    AbbreviationFinderStep,
)

# creates a document with a single section
doc = Document.create_simple_document(
    "Epidermal Growth Factor Receptor (EGFR) is a gene."
)
# create an Entity for the span "Epidermal Growth Factor Receptor"
entity = Entity.load_contiguous_entity(
    # start and end are the character indices for the entity
    start=0,
    end=len("Epidermal Growth Factor Receptor"),
    namespace="example",
    entity_class="gene",
    match="Epidermal Growth Factor Receptor",
)

# add it to the documents first (and only) section
doc.sections[0].entities.append(entity)

# create an instance of the AbbreviationFinderStep
step = AbbreviationFinderStep()
# a step may fail to process a document, so it returns two lists:
# all the docs, and just the failures
processed, failed = step([doc])
# check that a new entity has been created, attached to the EGFR span
egfr_entity = next(filter(lambda x: x.match == "EGFR", doc.get_entities()))
assert egfr_entity.entity_class == "gene"
print(egfr_entity.match)

# Kazu Data Mode
# from kazu.data.data import Document, Entity
from kazu.steps.document_post_processing.abbreviation_finder import (
    AbbreviationFinderStep,
)

# creates a document with a single section
doc = Document.create_simple_document(
    "Epidermal Growth Factor Receptor (EGFR) is a gene."
)
# create an Entity for the span "Epidermal Growth Factor Receptor"
entity = Entity.load_contiguous_entity(
    # start and end are the character indices for the entity
    start=0,
    end=len("Epidermal Growth Factor Receptor"),
    namespace="example",
    entity_class="gene",
    match="Epidermal Growth Factor Receptor",
)

# add it to the documents first (and only) section
doc.sections[0].entities.append(entity)

# create an instance of the AbbreviationFinderStep
step = AbbreviationFinderStep()
# a step may fail to process a document, so it returns two lists:
# all the docs, and just the failures
processed, failed = step([doc])
# check that a new entity has been created, attached to the EGFR span
egfr_entity = next(filter(lambda x: x.match == "EGFR", doc.get_entities()))
assert egfr_entity.entity_class == "gene"
print(egfr_entity.match)


# THESE ARE EXAMPLE FOR Data Serialization and deserialization

# Serialization: DocumentJsonUtils.doc_to_json_dict()

# from kazu.utils import DocumentJsonUtils
# from kazu.data.data import Document
# assume a Document instance
# doc = Document(text="Example text", entities=[], sections=[])
# Serialize the Document object
# json_dict = DocumentJsonUtils.doc_to_json_dict(doc)
# print(json_dict)

# Deserialization: Document.from_json()
# json_data = {
#     "text": "Example text",
#     "entities": [],
#     "sections": []
# }
#
# # Deserialize JSON data into a Document object
# doc = Document.from_json(json_data)
#
# print(doc.text)  # Output: Example text

# Creating an Object from a Dictionary: from_dict()
# import copy
# from kazu.data.data import Document
#
# # Example dictionary corresponding to a Document object
# dict_data = {
#     "text": "Example text",
#     "entities": [],
#     "sections": []
# }


# Visualising results in Label Studio
# https://astrazeneca.github.io/KAZU/label_studio_integration.html

# Pre-annotate your documents with Kazu
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from kazu.utils.constants import HYDRA_VERSION_BASE
from kazu.pipeline import Pipeline
from kazu.data.data import Document

@hydra.main(version_base=HYDRA_VERSION_BASE, config_path="conf", config_name="config")
def run_docs(cfg: DictConfig) -> None:
    pipeline: Pipeline = instantiate(cfg.Pipeline)
    docs = [Document.create_simple_document(x) for x in ["doc 1 text", "doc 2 text etc"]]
    pipeline(docs)


if __name__ == "__main__":
    run_docs()

# Load your annotations into Label Studio
from kazu.annotation.label_studio import (
    LabelStudioManager,
    LabelStudioAnnotationView,
)
from kazu.data.data import Document

docs: list[Document]

# create the view
view = LabelStudioAnnotationView(
    ner_labels={
        "cell_line": "red",
        "cell_type": "darkblue",
        "disease": "orange",
        "drug": "yellow",
        "gene": "green",
        "species": "purple",
        "anatomy": "pink",
        "molecular_function": "grey",
        "cellular_component": "blue",
        "biological_process": "brown",
    }
)

# if running locally...
url_and_port = "http://localhost:8080"
headers = {
    "Authorization": "Token <your token here>",
    "Content-Type": "application/json",
}

manager = LabelStudioManager(project_name="test", headers=headers, url=url_and_port)
manager.create_linking_project()
manager.update_tasks(docs)
manager.update_view(view=view, docs=docs)

# View/correct annotations in label studio. Once you’re finished, you can export back to Kazu Documents as follows:
from kazu.annotation.label_studio import LabelStudioManager
from kazu.data.data import Document

url_and_port = "http://localhost:8080"
headers = {
    "Authorization": "Token <your token here>",
    "Content-Type": "application/json",
}

manager = LabelStudioManager(project_name="test", headers=headers, url=url_and_port)

docs: list[Document] = manager.export_from_ls()

# Your ‘gold standard’ entities will now be accessible on the kazu.data.data.Section.metadata dictionary with the key: ‘gold_entities’
# For an example of how we integrate label studio into the Kazu acceptance tests, take a look at kazu.annotation.acceptance_test.analyse_full_pipeline()



# The OntologyParser
# https://astrazeneca.github.io/KAZU/ontology_parser.html
# Knowledge Bases are a core component of entity linking
# a lot of value as a vocabulary source for Dictionary based NER
# Kazu OntologyParser

# Writing a Custom Parser
# make a parser for a new datasource, (perhaps for NER or as a new linking target)
# First, OntologyParser.parse_to_dataframe()
import sqlite3  # interact with SQLite database

import pandas as pd

from kazu.ontology_preprocessing.base import (
    OntologyParser,
    DEFAULT_LABEL,
    IDX,
    SYN,
    MAPPING_TYPE,
)


def parse_to_dataframe(self) -> pd.DataFrame:
    """The objective of this method is to create a long, thin pandas dataframe of terms and
    associated metadata.

    We need at the very least, to extract an id and a default label. Normally, we'd also be
    looking to extract any synonyms and the type of mapping as well.
    """

    # fortunately, Chembl comes as an sqlite DB,
    # which lends itself very well to this tabular structure
    conn = sqlite3.connect(self.in_path)    # connect to SQLite database
    query = f"""\
        SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, synonyms AS {SYN},
            syn_type AS {MAPPING_TYPE}
        FROM molecule_dictionary AS md
                 JOIN molecule_synonyms ms ON md.molregno = ms.molregno
        UNION ALL
        SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, pref_name AS {SYN},
            'pref_name' AS {MAPPING_TYPE}
        FROM molecule_dictionary
    """
    df = pd.read_sql(query, conn)
    # eliminate anything without a pref_name, as will be too big otherwise
    df = df.dropna(subset=[DEFAULT_LABEL])

    df.drop_duplicates(inplace=True)

    return df

# Secondly, we need to write the OntologyParser.find_kb() method:
def find_kb(self, string: str) -> str:
    """In our case, this is simple, as everything in the Chembl DB has a chembl identifier.

    Other ontologies may use composite identifiers, e.g. MONDO contains native MONDO_xxxxx
    identifiers as well as HP_xxxxxxx identifiers. In this scenario, we'd need to parse the
    'string' parameter of this method to extract the relevant KB identifier.
    """
    return "CHEMBL"

# The FULL CLASS looks like:
class ChemblOntologyParser(OntologyParser):
    def find_kb(self, string: str) -> str:
        return "CHEMBL"

    def parse_to_dataframe(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.in_path)
    """looking for a chembl indertifier in molecule_dictionary(md) and molecule_synonyms(ms), linking by UNION ALL"""
        query = f"""\
            SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, synonyms AS {SYN},
                syn_type AS {MAPPING_TYPE}
            FROM molecule_dictionary AS md
                     JOIN molecule_synonyms ms ON md.molregno = ms.molregno
            UNION ALL
            SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, pref_name AS {SYN},
                'pref_name' AS {MAPPING_TYPE}
            FROM molecule_dictionary
        """
        df = pd.read_sql(query, conn)
        # eliminate anything without a pref_name, as will be too big otherwise
        df = df.dropna(subset=[DEFAULT_LABEL])

        df.drop_duplicates(inplace=True)

        return df

# Finally, when we want to use our new parser, we need to give it information about what entity class it is associated with:
# We need a string scorer to resolve similar terms.
# Here, we use a trivial example for brevity.
string_scorer = lambda string_1, string_2: 0.75 # evaluate the similarity between string_1 and string_2 and return the same value 0.75
parser = ChemblOntologyParser(      # ChemblOntologyParser is a subclass inherited from OntologyParser, specifically used to parse information from the ChEMBL database
    in_path="path to chembl DB goes here",  # replace to our actual path
    # if used in entity linking, entities with class 'drug'
    # will be associated with this parser
    entity_class="drug",
    name="CHEMBL",  # a globally unique name for the parser
    string_scorer=string_scorer,    # parser use this function to evaluate the similarity between strings
)



# Curating a knowledge base for NER and Linking
# blank on https://astrazeneca.github.io/KAZU/curating_a_knowledgebase.html#



# Scaling with RAY
# blank on https://astrazeneca.github.io/KAZU/scaling_kazu.html



# Kazu as a WebService
# blank on https://astrazeneca.github.io/KAZU/kazu_webservice.html#



# Kazu as a Library
# Dependency conflicts
# We test kazu with the latest version of each of its dependencies as descibed in the pyproject.toml file.
#  If you suspect you are having dependency clash issues,
#  you can view the dependencies a given Kazu model pack was tested with via the tested_dependencies.txt file
#  (located at the top level of a model pack). Try installing the version of the problematic dependency listed here.