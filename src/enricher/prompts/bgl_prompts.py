from langchain_core.prompts import ChatPromptTemplate

BGL_PROMPT_CORPUS = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      """
      You are analyst assistant that enriches log templates with additional information about the system.
      For each log template, enrich it with:
      - The system component that generated the log
      - The log level extracted from the template (e.g. INFO, ERROR, etc.)
      - A brief description of the log's likely meaning or purpose in scope of the system's operation
      - Any relevant metadata that can be inferred from the template
      """
    ),
    (
      "human",
      """
      Enrich the following log template with additional information:
      {log_template}
      """
    )
  ]
)
