from .llm import LLM
from .prompts import HDFS_PROMPT_CORPUS, BGL_PROMPT_CORPUS
from .schemas import EnrichedTemplate

class Enricher:
  def __init__(self, model = None):
    llm = LLM(model=model).get_llm()
    # json_mode uses response_format=json_object, supported by Mistral endpoints.
    # json_schema is not supported on Azure AI Foundry (raises 422).
    self.structured_llm = llm.with_structured_output(EnrichedTemplate, method="json_mode")

  def enrich_corpus_hdfs(self, template: str) -> EnrichedTemplate:
    chain = HDFS_PROMPT_CORPUS | self.structured_llm
    return chain.invoke({"log_template": template})

  def enrich_corpus_bgl(self, template: str) -> EnrichedTemplate:
    chain = BGL_PROMPT_CORPUS | self.structured_llm
    return chain.invoke({"log_template": template})
  


